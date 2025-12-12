# Repository Guidelines

## Project Structure & Module Organization
- `d3llm/d3llm_DREAM` and `d3llm/d3llm_LLaDA`: training pipelines; `distill_1_data_prepare` builds pseudo-trajectory data, `distill_2_training` runs main distillation.
- `chat/`: quick chat entrypoints for Dream and LLaDA models.
- `eval_scripts/`: benchmark runners (GSM8K, MATH, HumanEval, MBPP, long-context) used once models are placed at the paths referenced inside each script.
- `baseline/`: baseline implementations (e.g., Fast_dLLM_v1).
- `utils/`: vendored evaluation and serving tools (lm-evaluation-harness, DreamCoder eval, serving helpers); most tests live here. Assets and leaderboard data sit in `asset/` and `AUP_leaderboard/`.

## Build, Test, and Development Commands
- Install: `pip install -r requirements.txt` (match pinned versions such as transformers==4.49.0, flash_attn==2.7.4.post1).
- Chat smoke tests: `python chat/chat_d3llm_dream.py` or `python chat/chat_d3llm_llada.py` once checkpoints are available locally/HF.
- Training: `deepspeed --num_gpus=4 d3llm/d3llm_DREAM/distill_2_training/d3llm_dream_train.py` (swap for the LLaDA script as needed).
- Evaluation: `bash eval_scripts/dream_gsm8k_cot.sh` or `bash eval_scripts/llada_math.sh`; update model/data paths inside the scripts before running.
- API sample: `python api-example.py` to start the FastAPI server after pointing `model_path` to your checkpoint.

## Coding Style & Naming Conventions
- Python 3.10+ with 4-space indent; prefer type hints and module-level docstrings on public utilities.
- Format with `black` before pushing; keep lines readable (≤120 chars) and avoid committing large artifacts.
- Naming: files/modules in `snake_case`, classes in `PascalCase`, functions/variables in `snake_case`, constants in `UPPER_SNAKE_CASE`.
- Parameterize new scripts via argparse/config files rather than hard-coding paths or hyperparameters.

## Testing Guidelines
- Core suites live in `utils/lm-evaluation-harness/tests`; use `pytest utils/lm-evaluation-harness/tests -k "<scope>"` for targeted runs (full run is heavy).
- For DreamCoder utilities, run `pytest utils/utils_DreamCoder` when touching sanitizers or evaluation logic.
- When adding training/chat changes, include a lightweight regression check (short diffusion step or single chat round) and note expected GPU/CPU needs in the PR.

## Commit & Pull Request Guidelines
- Use short, imperative commit messages (e.g., “Add Dream chat fallback”), consistent with the existing history.
- In PRs, describe intent, key changes, and reproduction commands (train/eval/chat/API) plus hardware and model path assumptions.
- Link related issues/benchmarks; attach metrics tables or logs when altering training/evaluation; add screenshots only for user-facing CLI/UI tweaks.
- Keep diffs focused and call out any new data/model artifacts or config expectations explicitly.
