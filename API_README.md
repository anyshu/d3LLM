# d3LLM API Server

FastAPI wrapper for d3LLM chat-style completions with OpenAI-compatible routes.

## Install
```bash
pip install -r api_requirements.txt
```
`api_requirements.txt` links to the pinned project `requirements.txt` plus FastAPI/uvicorn extras.

## Configure
Set via `.env` or shell:
```
# Model sources (HF repo id or local path)
D3LLM_MODEL_PATH_DREAM=/data/models/d3llm/Dream
D3LLM_MODEL_PATH_LLADA=/data/models/d3llm/LLaDA

# Device per model (cuda:N | cpu | mps)
D3LLM_DEVICE_DREAM=cuda:0
D3LLM_DEVICE_LLADA=cuda:1

# Startup behavior
D3LLM_PRELOAD=1                # preload on startup (default 1)
D3LLM_WARMUP_PROMPT=Hello      # warmup prompt
D3LLM_WARMUP_TOKENS=8          # warmup length
D3LLM_COMPILE=1                # torch.compile Dream backend (set 0 to disable)

# Server
API_HOST=0.0.0.0
API_PORT=18091
```
Fallback: `DEVICE` sets the default for both models if per-model vars are omitted.
Generation knobs mirror the chat reference scripts; see `api_server.py` for optional overrides such as `D3LLM_DREAM_STEPS` or `D3LLM_LLADA_BLOCK_SIZE`.

## Run
```bash
source .env  # optional
uvicorn api_server:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-18091}
```
On startup (when `D3LLM_PRELOAD=1`), each model is loaded to its assigned device and run through a short warmup generate.

## Endpoints
- `GET /health` → status, default device, per-model device mapping, loaded models.
- `GET /v1/models` → available model IDs (`d3LLM-Dream`, `d3LLM-LLaDA`).
- `POST /v1/chat/completions` → OpenAI-style chat completion (set `stream:true` for SSE).

Example:
```bash
curl -X POST http://localhost:18091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "d3LLM-Dream",
    "messages": [{"role": "user", "content": "Write a Python hello world."}],
    "max_tokens": 128,
    "temperature": 0.3,
    "extra_body": {
      "block_length": 64,
      "diffusion_threshold": 0.6,
      "diffusion_steps": 192
    }
  }'
```

Stream with `curl -N` (keeps the connection open for Server-Sent Events):
```bash
curl -N -X POST http://localhost:18091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "d3LLM-Dream",
    "messages": [{"role": "user", "content": "Say hi in streaming mode."}],
    "stream": true
  }'
```

Notes:
- Defaults match the chat entrypoints: `max_tokens`=256 and `temperature`=0.0 unless overridden.
- `extra_body` maps OpenAI-style fields to generation kwargs: `block_length→block_size`, `diffusion_threshold→threshold`, `diffusion_steps→steps`.
- If a specified CUDA device is unavailable, the server returns 503 with details.
- Memory use stacks if both models share the same GPU; split devices to avoid OOM.
