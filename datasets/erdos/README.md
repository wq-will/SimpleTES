# Erdős Minimum Overlap

Find a step function $h : [0, 2] \to [0, 1]$ that **minimises** the overlap integral

$$C_5 = \max_k \int h(x)\,(1 - h(x+k))\,dx$$

subject to $\sum h = n/2$. Evolves the discrete `h`; the evaluator recomputes C₅ and scores via reciprocal.

| Task | What to evolve | Score |
|------|----------------|-------|
| **erdos/erdos_min_overlap** | `construct_h()` in `init_program.py` — returns a 1-D non-negative array with entries in `[0, 1]` | $1 / (10^{-8} + C_5)$ |

Validator is marked `HACK-PROOF DESIGN`: always recomputes C₅, refuses self-reported values. If `sum(h) ≠ n/2`, rescales `h` proportionally; if rescaled values fall outside `[0, 1]`, rejects the solution.

## Requirements

`numpy`. No setup.

## Running

```bash
python main.py \
  --init-program datasets/erdos/erdos_min_overlap/init_program.py \
  --evaluator    datasets/erdos/erdos_min_overlap/evaluator.py \
  --instruction  datasets/erdos/erdos_min_overlap/erdos_min_overlap.txt \
  --model <your-model>
```

Per-evaluation timeout is **1100 s**.
