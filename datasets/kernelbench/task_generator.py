import os
import glob
import re
import ast

KERNEL_BENCH_PATH = os.environ.get("KERNEL_BENCH_PATH", os.path.join(os.path.dirname(__file__), "KernelBench"))

EVALUATOR_TEMPLATE = '''import os
import json
import requests
import yaml

LEVEL_ID = 1
TASK_ID = 1
TASK = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""

    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def read_server_ip_port():
    """
    Read server IP and port from server_info.yaml

    Returns:
        tuple: (ip, port)
    """
    config_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(config_dir, "server_info.yaml")

    default_ip = "localhost"
    default_port = 8000

    if not os.path.exists(config_path):
        print(f"[Client] server_info.yaml not found at {config_path}")
        print(f"[Client] Using default server: {default_ip}:{default_port}")
        return default_ip, default_port

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        ip = config.get("server_ip", default_ip)
        port = config.get("server_port", default_port)

        print(f"[Client] Loaded server config: {ip}:{port}")
        return ip, port

    except Exception as e:
        print(f"[Client] Error reading server_info.yaml: {e}")
        print(f"[Client] Using default server: {default_ip}:{default_port}")
        return default_ip, default_port


def evaluate(program_path):
    """Evaluate a program by sending it to the evaluation server"""
    serv_ip, serv_port = read_server_ip_port()
    server_url = f"http://{serv_ip}:{serv_port}/evaluate"

    program_src = read_file(program_path)
    if not program_src:
        return {
            "compiled": False,
            "correctness": False,
            "metadata": {"client_error": f"Failed to read program from {program_path}"},
            "runtime_stats": {},
            "ref_runtime_stats": {},
            "combined_score": 0,
        }

    request_data = {
        "program_src": program_src,
        "level_id": LEVEL_ID,
        "task_id": TASK_ID,
        "task": TASK,
    }

    try:
        print(f"[Client] Connecting to evaluation server at {server_url}")
        print("[Client] Sending evaluation request...")

        response = requests.post(server_url, json=request_data, timeout=1800)

        print(f"[Client] Received response (status code: {response.status_code})")

        if response.status_code == 200:
            response_data = response.json()
            if response_data.get("success") and response_data.get("result"):
                print("  - Success: True")
                print(f"  - Device ID: {response_data.get('device_id')}")
                print(f"  - Compiled: {response_data.get('result', {}).get('compiled')}")
                print(f"  - Correctness: {response_data.get('result', {}).get('correctness')}")
                print(f"  - Combined Score: {response_data.get('result', {}).get('combined_score', 0):.6f}")
                return response_data["result"]
            else:
                print("  - Success: False")
                print(f"  - Response data: {response_data.get('result', {}).get('metadata', {})}")
                return response_data["result"]

        error_msg = f"HTTP {response.status_code}: {response.text}"
        print(f"  - HTTP Error: {error_msg}")
        return {
            "compiled": False,
            "correctness": False,
            "error": error_msg,
            "error_name": "HTTP Error",
            "combined_score": 0.0,
        }

    except requests.exceptions.ConnectionError:
        return {
            "compiled": False,
            "correctness": False,
            "error": f"Connection refused. Is the server running at {server_url}?",
            "error_name": "Connection Error",
            "combined_score": 0.0,
        }
    except Exception as e:
        return {
            "compiled": False,
            "correctness": False,
            "error": f"Unexpected error: {str(e)}",
            "error_name": type(e).__name__,
            "combined_score": 0.0,
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <program_path>")
        sys.exit(1)

    program_path = sys.argv[1]
    result = evaluate(program_path)

    if result.get("compiled") and result.get("correctness"):
        print("[Client] Evaluation completed successfully!")
        sys.exit(0)

    print("[Client] Evaluation failed")
    if not result.get("compiled"):
        print("  - Compilation failed")
    if not result.get("correctness"):
        print("  - Correctness check failed")
    if "client_error" in result.get("metadata", {}):
        print(f"  - Error: {result['metadata']['client_error']}")
    sys.exit(1)
'''

PROMPT_TEMPLATE = '''You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul + relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

Your task is to rewrite the following Model in {task_name}.py using improved CUDA kernels:
```python
{evolve_block}
```

Here is an example to show you the syntax of inline embedding custom CUDA operators in torch:

```python
from torch.utils.cpp_extension import load_inline

cpp_source="torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"

cuda_source="""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_add = load_inline(
    name='elementwise_add',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['elementwise_add_cuda'],
    extra_cflags =[''],
    extra_ldflags=['']
)
```
'''


def extract_task_id(filename):
    """Extract task ID from filename like '1_Square_matrix_multiplication_.py'"""
    match = re.match(r'^(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None


def add_evolve_markers(source_code):
    """
    Add EVOLVE-BLOCK markers to the source code.
    START marker at the beginning, END marker after the Model class definition.
    Uses AST to find the exact end of the Model class.
    """
    lines = source_code.split('\n')

    # Use AST to find the end of Model class
    model_class_end = -1
    try:
        tree = ast.parse(source_code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Model':
                # node.end_lineno是类定义结束的行号（Python 3.8+）
                if hasattr(node, 'end_lineno'):
                    # 转换为0-based索引
                    model_class_end = node.end_lineno - 1
                else:
                    # 对于旧版本Python，查找最后一个类成员的最后一行
                    last_line = node.lineno - 1
                    for item in node.body:
                        if hasattr(item, 'end_lineno'):
                            last_line = max(last_line, item.end_lineno - 1)
                        elif hasattr(item, 'lineno'):
                            last_line = max(last_line, item.lineno - 1)
                    model_class_end = last_line
                break
    except SyntaxError as e:
        print(f"[Warning] Failed to parse code with AST: {e}")
        # Fallback: search for get_inputs function
        for i, line in enumerate(lines):
            if line.strip().startswith('def get_inputs()'):
                model_class_end = i
                break

        if model_class_end == -1:
            model_class_end = len(lines)

    # Insert markers
    lines.insert(model_class_end + 1, '# EVOLVE-BLOCK-END')
    lines.insert(0, '# EVOLVE-BLOCK-START')

    return '\n'.join(lines)


def extract_evolve_block(source_code):
    """Extract content between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END"""
    lines = source_code.split('\n')
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if '# EVOLVE-BLOCK-START' in line:
            start_idx = i + 1
        elif '# EVOLVE-BLOCK-END' in line:
            end_idx = i
            break

    if start_idx is not None and end_idx is not None:
        return '\n'.join(lines[start_idx:end_idx])
    return ""


def generate_task(level_id, task_file):
    """Generate a single task directory with all required files"""
    task_filename = os.path.basename(task_file)
    task_id = extract_task_id(task_filename)

    if task_id is None:
        print(f"[Warning] Could not extract task ID from {task_filename}, skipping...")
        return

    task_name = task_filename.replace('.py', '')
    task_dir_name = f"level{level_id}_{task_name}"

    os.makedirs(task_dir_name, exist_ok=True)
    print(f"[Generator] Creating task: {task_dir_name}")

    # Read the original source file
    with open(task_file, 'r') as f:
        source_code = f.read()

    # 1. Generate init_program.py with EVOLVE-BLOCK markers
    init_program_path = os.path.join(task_dir_name, 'init_program.py')
    init_program_code = add_evolve_markers(source_code)
    with open(init_program_path, 'w') as f:
        f.write(init_program_code)

    # 2. Generate evaluator.py using regex replacement
    evaluator_path = os.path.join(task_dir_name, 'evaluator.py')
    evaluator_code = EVALUATOR_TEMPLATE
    evaluator_code = re.sub(r'LEVEL_ID\s*=\s*\d+', f'LEVEL_ID = {level_id}', evaluator_code)
    evaluator_code = re.sub(r'TASK_ID\s*=\s*\d+', f'TASK_ID = {task_id}', evaluator_code)
    with open(evaluator_path, 'w') as f:
        f.write(evaluator_code)

    # 3. Extract evolve block and generate prompt
    evolve_block = extract_evolve_block(init_program_code)
    prompt = PROMPT_TEMPLATE
    prompt = re.sub(r'{task_name}', task_name, prompt)
    prompt = re.sub(r'{evolve_block}', evolve_block, prompt)

    prompt_filename = f"level{level_id}_{task_name}.txt"
    prompt_path = os.path.join(task_dir_name, prompt_filename)
    with open(prompt_path, 'w') as f:
        f.write(prompt)

    print(f"[Generator] ✓ Created {task_dir_name}")


def generate_all_tasks():
    """Generate all tasks from KernelBench levels 1-3"""
    # Level 1-2: 100 tasks each, Level 3: 50 tasks
    level_configs = [
        (1, 100),
        (2, 100),
        (3, 50)
    ]

    for level_id, num_tasks in level_configs:
        level_path = os.path.join(KERNEL_BENCH_PATH, f"level{level_id}")

        if not os.path.exists(level_path):
            print(f"[Warning] Level path does not exist: {level_path}")
            continue

        print(f"\n[Generator] Processing level{level_id}: {num_tasks} tasks")

        for task_id in range(1, num_tasks + 1):
            # Find the task file
            task_files = glob.glob(os.path.join(level_path, f"{task_id}_*.py"))

            if not task_files:
                print(f"[Warning] Task {task_id} not found in level{level_id}")
                continue

            task_file = task_files[0]

            try:
                generate_task(level_id, task_file)
            except Exception as e:
                print(f"[Error] Failed to generate task from {task_file}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("OpenEvolve Task Generator")
    print("=" * 60)
    print(f"KernelBench path: {KERNEL_BENCH_PATH}")
    print(f"Output directory: {os.getcwd()}")
    print("=" * 60)

    generate_all_tasks()

    print("\n" + "=" * 60)
    print("[Generator] Task generation complete!")
    print("=" * 60)
