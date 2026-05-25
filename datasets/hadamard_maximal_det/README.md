# Hadamard Maximum Determinant

Construct an `n × n` ±1 matrix `H` maximising `|det(H)|`. `n = 29` is not a Hadamard order, so the score is normalised against the theoretical maximum.

<table>
<thead>
<tr>
  <th align="left">Task</th>
  <th align="center">n</th>
  <th align="left">What to evolve</th>
  <th align="left"><code>combined_score</code></th>
  <th align="center">TIMEOUT</th>
</tr>
</thead>
<tbody>
<tr>
  <td><b>hadamard_maximal_det/<br/>hadamard_maximal_det_29</b></td>
  <td align="center">29</td>
  <td><code>construct_hadamard_matrix(n=29)</code> — returns a <code>(29, 29)</code> matrix</td>
  <td>|det(H)| / 29<sup>29/2</sup></td>
  <td align="center">350 s</td>
</tr>
</tbody>
</table>

`29^(29/2)` is the Hadamard bound on the determinant of any 29 × 29 matrix with entries in `[-1, 1]` (defined as `THEORETICAL_MAX` in `evaluator.py`). Score is a ratio in `[0, 1]`; an actual Hadamard matrix would score 1.0.

Validator: shape `(29, 29)`, no `NaN`, every entry exactly ±1. Determinant is recomputed via the Bareiss algorithm in exact integer arithmetic.

## Requirements

`numpy`. No setup.

## Running

```bash
python main.py \
  --init-program datasets/hadamard_maximal_det/hadamard_maximal_det_29/init_program.py \
  --evaluator    datasets/hadamard_maximal_det/hadamard_maximal_det_29/evaluator.py \
  --instruction  datasets/hadamard_maximal_det/hadamard_maximal_det_29/hadamard_maximal_det_29.txt \
  --model <your-model>
```
