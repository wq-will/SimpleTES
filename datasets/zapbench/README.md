# ZAPBench Whole-Brain Forecasting

Predict the next 32/16/8/4/1 whole-brain activity frames from a 4-frame context. The evaluator scores predictions by negative mean absolute error, so higher is better.

## Setup

From the repository root, clone the public ZAPBench dependency and prepare the cached datasets:

```bash
mkdir -p third_party
git clone https://github.com/google-research/zapbench.git third_party/zapbench
git -C third_party/zapbench checkout b08d584e3ba80125788ac915eec63a2e4e11467b
bash datasets/zapbench/whole_brain_forecasting/setup.sh
datasets/zapbench/whole_brain_forecasting/.venv/bin/python \
  datasets/zapbench/whole_brain_forecasting/prepare_data.py
```

## Run

```bash
python main.py \
  --init-program datasets/zapbench/whole_brain_forecasting/init_program.py \
  --evaluator datasets/zapbench/whole_brain_forecasting/evaluator.py \
  --instruction datasets/zapbench/whole_brain_forecasting/zapbench.txt \
  --eval-venv datasets/zapbench/whole_brain_forecasting/.venv \
  --model <your-model>
```