# Autocorrelation Inequalities

Three extremal-analysis tasks on a non-negative function `f` discretised over `[-1/4, 1/4]`. Each subtask has a different direction (min / max) and a different scoring formula; the engine always maximises `combined_score`.

<table>
<thead>
<tr>
  <th align="left">Subtask</th>
  <th align="center">Direction</th>
  <th align="left">Quantity</th>
  <th align="left"><code>combined_score</code></th>
  <th align="center">TIMEOUT</th>
</tr>
</thead>
<tbody>
<tr>
  <td><b>autocorrelation_first</b></td>
  <td align="center"><b>minimise</b></td>
  <td>C₁ = 2n · max(f * f) / (∑f)²</td>
  <td>1 / (10⁻⁸ + C₁)</td>
  <td align="center">1100 s</td>
</tr>
<tr>
  <td><b>autocorrelation_second</b></td>
  <td align="center"><b>maximise</b></td>
  <td>C₂ = ‖f*f‖₂² / (‖f*f‖₁ · ‖f*f‖<sub>∞</sub>)</td>
  <td>C₂ (raw, no transform)</td>
  <td align="center">1100 s</td>
</tr>
<tr>
  <td><b>autocorrelation_third</b></td>
  <td align="center"><b>minimise</b></td>
  <td>C₃ = 2n · max(|conv(f, f)|) / (∑f)²</td>
  <td>BENCHMARK / C₃</td>
  <td align="center">70 s</td>
</tr>
</tbody>
</table>

Notes:

- `autocorrelation_second` is the only maximisation target; score is raw C₂. (The instruction defines `R(f) ≤ C₂`; the evaluator calls the computed value `c2_value` and uses it as the score directly.)
- `autocorrelation_third` divides by `BENCHMARK = 1.4556427953745406`, the AlphaEvolve C₃ baseline (the reference point in the SimpleTES paper). Matching AlphaEvolve scores 1.0; beating it scores > 1.0.
- The evaluator always recomputes the score and warns when the candidate's self-reported value disagrees by more than `1e-4`.

## Requirements

Python with `numpy` (and the LP solver declared in `requirements.txt`). No external setup.

## Running

```bash
python main.py \
  --init-program datasets/autocorrelation/<subtask>/init_program.py \
  --evaluator    datasets/autocorrelation/<subtask>/evaluator.py \
  --instruction  datasets/autocorrelation/<subtask>/<subtask>.txt \
  --model <your-model>
```

Each subtask uses its own `TIMEOUT_SECONDS` default (above); override per-run via `EVALUATOR_TIMEOUT_SECONDS=<n>`.
