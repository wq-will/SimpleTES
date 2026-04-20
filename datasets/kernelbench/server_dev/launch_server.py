import asyncio
import os
from random import choice
import sys
from contextlib import asynccontextmanager
from typing import Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from pydantic import BaseModel
from fastapi import FastAPI, Request
import uvicorn

from robust_pool import RobustSubprocessPool
from server_utils import tprint, get_gpu_count


def parse_gpu_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip() != ""]
    if not items:
        raise ValueError("GPU list cannot be empty")

    gpu_ids = []
    for item in items:
        try:
            gpu_id = int(item)
        except ValueError as e:
            raise ValueError(f"Invalid GPU id '{item}' in list '{value}'") from e

        if gpu_id < 0:
            raise ValueError(f"GPU id must be >= 0, got {gpu_id}")

        gpu_ids.append(gpu_id)

    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"GPU list contains duplicates: {value}")

    return gpu_ids

class EvalReq(BaseModel):
    program_src: str
    level_id: int
    task_id: int
    task: str


pool_manager: RobustSubprocessPool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global pool_manager
    tprint("[Server] Starting up...")

    num_compile_workers = getattr(app.state, 'num_compile_workers', 128)
    num_gpus = getattr(app.state, 'num_gpus', None)
    seed_num = getattr(app.state, 'seed_num', 42)
    precision = getattr(app.state, 'precision', 'float32')
    timeout = getattr(app.state, 'timeout', 600.0)
    backend = getattr(app.state, 'backend', 'triton')

    if num_gpus is None:
        num_gpus = get_gpu_count()


    extra_args = getattr(app.state, 'extra_args', {})

    pool_manager = RobustSubprocessPool(
        num_compile_workers=num_compile_workers,
        num_gpus=num_gpus,
        seed_num=seed_num,
        precision=precision,
        timeout=timeout,
        backend=backend,
        extra_args=extra_args,
    )
    await pool_manager.initialize()

    tprint("[Server] Startup complete")

    yield

    tprint("[Server] Shutting down...")
    if pool_manager:
        await pool_manager.shutdown()
    tprint("[Server] Shutdown complete")


app = FastAPI(lifespan=lifespan)


@app.post("/evaluate")
async def eval_kernel(req: EvalReq) -> Dict:
    """Handle evaluation request"""
    global pool_manager

    if pool_manager is None:
        return {"success": False, "error": "Pool not initialized"}

    return await pool_manager.process_request(
        program_src=req.program_src,
        level_id=req.level_id,
        task_id=req.task_id,
        task=req.task,
    )


@app.get("/stats")
async def get_stats() -> Dict:
    """Get pool statistics"""
    global pool_manager
    if pool_manager is None:
        return {"error": "Pool not initialized"}

    return pool_manager.get_stats()


@app.get("/workerstatus")
async def get_worker_status() -> Dict:
    """Get detailed worker queue status"""
    global pool_manager
    if pool_manager is None:
        return {"error": "Pool not initialized"}

    return pool_manager.get_worker_status()


@app.get("/health")
async def health() -> Dict:
    """Health check endpoint"""
    return {"status": "ok"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Robust GPU Kernel Evaluation Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--compile-workers", type=int, default=128,
                        help="Number of compile workers (default: 128), only work for cuda based kernels")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to use (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for evaluation (default: 42)")
    parser.add_argument("--precision", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Precision type for evaluation (default: float32)")
    parser.add_argument("--timeout", type=float, default=1800,
                        help="Timeout for evaluation in seconds (default: 1800)")
    parser.add_argument("--backend", choices=["cuda", "torch", "triton"], default="triton",
                        help="Backend to use for evaluation (default: triton)")
    parser.add_argument("--triton-compiler-gpus", type=str, default="0,1",
                        help="Comma-separated GPU ids for Triton compiler/correctness stage")
    parser.add_argument("--triton-eval-gpus", type=str, default="2,3,4,5,6,7",
                        help="Comma-separated GPU ids for Triton performance eval stage")
    parser.add_argument("--triton-compiler-gpu-concurrency", type=int, choices=[4, 8, 12], default=12,
                        help="Max concurrent CompilerEval tasks per compiler GPU")
    parser.add_argument("--triton-perf-cpu-per-gpu", type=int, default=4,
                        help="CPU cores reserved per Triton perf GPU (default: 4)")
    args = parser.parse_args()

    extra_args = {}
    # triton has no method for AoT compilation, and can only put compilation and correctness together in the compilation worker, because there is no impact on performance
    if args.backend == "triton":
        triton_compiler_gpus = parse_gpu_list(args.triton_compiler_gpus)
        triton_eval_gpus = parse_gpu_list(args.triton_eval_gpus)

        if len(set(triton_compiler_gpus) & set(triton_eval_gpus)) > 0:
            overlap = set(triton_compiler_gpus) & set(triton_eval_gpus)
            overlap_str = ",".join(str(x) for x in sorted(overlap))
            raise ValueError(
                f"Triton compiler GPUs and eval GPUs must not overlap, overlapped: {overlap_str}"
            )

        if args.triton_perf_cpu_per_gpu <= 0:
            raise ValueError("--triton-perf-cpu-per-gpu must be > 0")

        extra_args = {
            "triton_compiler_gpus": triton_compiler_gpus,
            "triton_eval_gpus": triton_eval_gpus,
            "triton_compiler_gpu_concurrency": args.triton_compiler_gpu_concurrency,
            "triton_perf_cpu_per_gpu": args.triton_perf_cpu_per_gpu,
        }

    app.state.num_compile_workers = args.compile_workers
    app.state.num_gpus = args.num_gpus
    app.state.seed_num = args.seed
    app.state.precision = args.precision
    app.state.timeout = args.timeout
    app.state.backend = args.backend
    app.state.extra_args = extra_args

    tprint("=" * 70)
    tprint("Robust GPU Kernel Evaluation Server")
    tprint("=" * 70)
    tprint(f"Host: {args.host}:{args.port}")
    if args.backend == "cuda":
        tprint(f"Compile Workers: {args.compile_workers}")
    else:
        tprint(f"Compile Workers: N/A ({args.backend} doesn't need compilation)")
    if args.num_gpus is not None:
        tprint(f"Eval Workers: {args.num_gpus}")
    else:
        tprint("Eval Workers: auto-detect")
    tprint(f"Seed: {args.seed}")
    tprint(f"Precision: {args.precision}")
    tprint(f"Timeout: {args.timeout}s")
    tprint(f"Backend: {args.backend}")
    if args.backend == "triton":
        tprint(f"Triton Compiler GPUs: {extra_args['triton_compiler_gpus']}")
        tprint(f"Triton Eval GPUs: {extra_args['triton_eval_gpus']}")
        tprint(f"Triton Perf CPU per GPU: {extra_args['triton_perf_cpu_per_gpu']}")
    tprint("Architecture:")
    tprint("  - On-demand subprocess workers")
    tprint("  - Automatic worker restart on crash")
    tprint("  - Shared build cache for Triton/CUDA compilation")
    tprint("=" * 70)
    tprint("Endpoints:")
    tprint("  POST /evaluate      - Evaluate a single kernel")
    tprint("  GET  /stats         - Get pool statistics")
    tprint("  GET  /workerstatus  - Get detailed worker queue status")
    tprint("  GET  /health        - Health check")
    tprint("=" * 70)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
