# Circle Packing in a Unit Square

Pack `n` non-overlapping circles in the unit square to maximise the sum of radii. Evolves the `(n, 3)` array of `(x, y, r)` triples; the evaluator checks geometric constraints and recomputes the sum.

| Task | n | What to evolve |
|------|---|----------------|
| **circle_packing/circle_packing_26** | 26 | `construct_circles()` in `init_program.py` — returns an `(26, 3)` numpy array |
| **circle_packing/circle_packing_32** | 32 | `construct_circles()` in `init_program.py` — returns a `(32, 3)` numpy array |

Validator: shape `(n, 3)`, no `NaN`, non-negative radii, inside the square, no pairwise overlap, all at `1e-12` tolerance. Score = recomputed sum of radii; self-reported scores are ignored.

## Requirements

`numpy`. No setup.

## Running

```bash
python main.py \
  --init-program datasets/circle_packing/circle_packing_26/init_program.py \
  --evaluator    datasets/circle_packing/circle_packing_26/evaluator.py \
  --instruction  datasets/circle_packing/circle_packing_26/circle_packing_26.txt \
  --model <your-model>
```

Per-evaluation timeout is **530 s** (`EVALUATOR_TIMEOUT_SECONDS`); both subtasks use the same default.
