# Sum-Difference Problem

Find a finite set of integers `A` that yields the strongest lower bound on the constant `C` in

```
|A + A| / |A|   ≤   ( |A - A| / |A| ) ^ C.
```

Evolves `A`; the evaluator recomputes `|A+A|`, `|A-A|`, and the bound from `A`.

<table>
<thead>
<tr>
  <th align="left">Task</th>
  <th align="left">What to evolve</th>
  <th align="left"><code>combined_score</code></th>
  <th align="center">TIMEOUT</th>
</tr>
</thead>
<tbody>
<tr>
  <td><b>sums_diffs/sums_diffs</b></td>
  <td><code>construct_set()</code> in <code>init_program.py</code> — returns a finite set of integers</td>
  <td>C(A) = log(|A+A| / |A|)&nbsp;/&nbsp;log(|A-A| / |A|)<br/><sub>(higher is better)</sub></td>
  <td align="center">180 s</td>
</tr>
</tbody>
</table>

Both `|A+A|/|A| > 1` and `|A-A|/|A| > 1` are required; otherwise the candidate is rejected.

Validator: `A` is a set of integers, `2 ≤ |A| ≤ 512` after dedup, elements in `[-10⁶, 10⁶]`, integer tolerance `1e-9`. The candidate may attach a self-reported `C` to `run_code()`'s return value; the validator always recomputes from `A` and warns on mismatch > `REPORTED_ATOL`.

## Requirements

Python stdlib. No setup.

## Running

```bash
python main.py \
  --init-program datasets/sums_diffs/sums_diffs/init_program.py \
  --evaluator    datasets/sums_diffs/sums_diffs/evaluator.py \
  --instruction  datasets/sums_diffs/sums_diffs/sums_diffs.txt \
  --model <your-model>
```
