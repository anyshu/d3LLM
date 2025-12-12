"""
Lightweight FastAPI server for d3LLM chat-style completions.
- Supports d3LLM-Dream and d3LLM-LLaDA via Hugging Face style checkpoints.
- Endpoints: /health, /v1/models, /v1/chat/completions (OpenAI-compatible schema, non-stream).
Run with: uvicorn api_server:app --host 0.0.0.0 --port 18091
Environment overrides:
  D3LLM_MODEL_PATH_DREAM, D3LLM_MODEL_PATH_LLADA, API_HOST, API_PORT, DEVICE
"""

import os
import time
import uuid
from typing import Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


# Model registry; override with env vars to point at local checkpoints.
MODEL_REGISTRY: Dict[str, str] = {
    "d3LLM-Dream": os.getenv("D3LLM_MODEL_PATH_DREAM", "d3LLM/d3LLM_Dream"),
    "d3LLM-LLaDA": os.getenv("D3LLM_MODEL_PATH_LLADA", "d3LLM/d3LLM_LLaDA"),
}


def resolve_device_default() -> str:
    if os.getenv("DEVICE"):
        return os.getenv("DEVICE")
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str):
    if device.startswith("cuda"):
        dev = torch.device(device)
        major, _ = torch.cuda.get_device_capability(dev)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


DEFAULT_DEVICE = resolve_device_default()
MODEL_DEVICES: Dict[str, str] = {
    "d3LLM-Dream": os.getenv("D3LLM_DEVICE_DREAM", DEFAULT_DEVICE),
    "d3LLM-LLaDA": os.getenv("D3LLM_DEVICE_LLADA", DEFAULT_DEVICE),
}

PRELOAD_MODELS = env_flag("D3LLM_PRELOAD", "1")
WARMUP_PROMPT = os.getenv("D3LLM_WARMUP_PROMPT", "Hello from d3LLM")
WARMUP_TOKENS = int(os.getenv("D3LLM_WARMUP_TOKENS", "8"))


class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (system|user|assistant)")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model ID, e.g., d3LLM-Dream or d3LLM-LLaDA")
    messages: List[Message]
    max_tokens: int = Field(512, description="Maximum new tokens to generate")
    temperature: float = Field(0.7, ge=0.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling top-p")
    stream: bool = Field(False, description="Streaming not supported in this minimal server")


class Choice(BaseModel):
    index: int
    finish_reason: str
    message: Dict[str, str]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: str = "d3llm-fastapi"


class ModelBundle:
    def __init__(self, model_id: str, model_path: str, device: str):
        self.model_id = model_id
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None

    def ensure_loaded(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise HTTPException(
                    status_code=503,
                    detail=f"CUDA not available but requested for {self.model_id}",
                )
            # Validate device index if provided, e.g., cuda:1.
            try:
                torch.cuda.get_device_capability(torch.device(self.device))
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"Invalid CUDA device for {self.model_id}: {self.device} ({exc})",
                )

        torch_dtype = resolve_dtype(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()


_MODEL_CACHE: Dict[str, ModelBundle] = {}


def get_model_bundle(model_id: str) -> ModelBundle:
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    if model_id not in _MODEL_CACHE:
        _MODEL_CACHE[model_id] = ModelBundle(
            model_id, MODEL_REGISTRY[model_id], MODEL_DEVICES[model_id]
        )
    bundle = _MODEL_CACHE[model_id]
    bundle.ensure_loaded()
    return bundle


def render_prompt(tokenizer, messages: List[Message]) -> str:
    message_dicts = [m.model_dump() for m in messages]
    try:
        return tokenizer.apply_chat_template(
            message_dicts, add_generation_prompt=True, tokenize=False
        )
    except Exception:
        joined = "\n".join(f"{m['role']}: {m['content']}" for m in message_dicts)
        return f"{joined}\nassistant:"


def resolve_pad_token_id(tokenizer) -> int:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise HTTPException(status_code=500, detail="Model lacks a pad/eos token.")
    return pad_token_id


def generate_response(
    bundle: ModelBundle,
    request: ChatCompletionRequest,
) -> Tuple[str, int, int]:
    tokenizer = bundle.tokenizer
    model = bundle.model
    prompt = render_prompt(tokenizer, request.messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(bundle.device)
    prompt_len = inputs["input_ids"].shape[-1]
    pad_token_id = resolve_pad_token_id(tokenizer)

    do_sample = request.temperature > 0
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
        )

    generated = output_ids[:, prompt_len:]
    text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    completion_tokens = generated.shape[-1]
    return text, prompt_len, completion_tokens


app = FastAPI(title="d3LLM API", version="0.1.0")


@app.on_event("startup")
async def preload_models() -> None:
    if not PRELOAD_MODELS:
        return
    for model_id in MODEL_REGISTRY:
        try:
            bundle = get_model_bundle(model_id)
            tokenizer = bundle.tokenizer
            pad_token_id = resolve_pad_token_id(tokenizer)
            prompt = render_prompt(
                tokenizer, [Message(role="user", content=WARMUP_PROMPT)]
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(bundle.device)
            with torch.no_grad():
                bundle.model.generate(
                    **inputs,
                    max_new_tokens=WARMUP_TOKENS,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )
            print(f"[preload] {model_id} ready on {bundle.device}")
        except Exception as exc:
            print(f"[preload] failed for {model_id}: {exc}")


@app.get("/health")
async def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "default_device": DEFAULT_DEVICE,
        "model_devices": MODEL_DEVICES,
        "available_models": list(MODEL_REGISTRY.keys()),
        "loaded_models": list(_MODEL_CACHE.keys()),
    }


@app.get("/v1/models")
async def list_models() -> Dict[str, List[Dict[str, str]]]:
    data = []
    for model_id in MODEL_REGISTRY:
        data.append(
            {"id": model_id, "object": "model", "owned_by": "d3LLM", "permission": []}
        )
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    if request.stream:
        raise HTTPException(
            status_code=400, detail="Streaming is not supported in this server."
        )
    bundle = get_model_bundle(request.model)
    start = time.time()
    content, prompt_tokens, completion_tokens = generate_response(bundle, request)
    elapsed = time.time() - start

    choice = Choice(
        index=0,
        finish_reason="stop",
        message={"role": "assistant", "content": content},
    )
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[choice],
        usage=usage,
    )
    # Attach latency for debugging.
    response_dict = response.model_dump()
    response_dict["latency_sec"] = round(elapsed, 3)
    return response_dict


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", os.getenv("PORT", 18091)))
    uvicorn.run("api_server:app", host=host, port=port, log_level="info")
