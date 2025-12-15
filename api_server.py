"""
FastAPI server for d3LLM chat completions.
- Supports d3LLM-Dream and d3LLM-LLaDA (logic mirrors chat/chat_d3llm_dream.py and chat/chat_d3llm_llada.py).
- Endpoints: /health, /v1/models, /v1/chat/completions (OpenAI-compatible schema with SSE streaming).
Run with: uvicorn api_server:app --host 0.0.0.0 --port 18091
Environment overrides:
  D3LLM_MODEL_PATH_DREAM, D3LLM_MODEL_PATH_LLADA, API_HOST, API_PORT, DEVICE
  D3LLM_COMPILE (1|0) to toggle torch.compile for Dream backend.
"""

import json
import os
import time
import types
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoTokenizer

from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamGenerationMixin
from d3llm.d3llm_LLaDA.d3llm_llada_generate_util import generate_multi_block_kv_cache
from utils.utils_Dream.model.configuration_dream import DreamConfig
from utils.utils_Dream.model.modeling_dream import DreamModel
from utils.utils_LLaDA.model.modeling_llada import LLaDAModelLM


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
ENABLE_COMPILE = env_flag("D3LLM_COMPILE", "1")

DREAM_GENERATION_DEFAULTS: Dict[str, object] = {
    "steps": int(os.getenv("D3LLM_DREAM_STEPS", "256")),
    "block_size": int(os.getenv("D3LLM_DREAM_BLOCK_SIZE", "32")),
    "block_add_threshold": float(os.getenv("D3LLM_DREAM_BLOCK_ADD_THRESHOLD", "0.1")),
    "decoded_token_threshold": float(
        os.getenv("D3LLM_DREAM_DECODED_TOKEN_THRESHOLD", "0.95")
    ),
    "threshold": float(os.getenv("D3LLM_DREAM_THRESHOLD", "0.5")),
    "cache_delay_iter": int(os.getenv("D3LLM_DREAM_CACHE_DELAY_ITER", "10000")),
    "alg": os.getenv("D3LLM_DREAM_ALG", "entropy_threshold"),
}
DREAM_DEFAULT_MAX_NEW_TOKENS = int(os.getenv("D3LLM_DREAM_MAX_TOKENS", "256"))

LLADA_GENERATION_DEFAULTS: Dict[str, object] = {
    "steps": int(os.getenv("D3LLM_LLADA_STEPS", "256")),
    "block_size": int(os.getenv("D3LLM_LLADA_BLOCK_SIZE", "32")),
    "block_add_threshold": float(
        os.getenv("D3LLM_LLADA_BLOCK_ADD_THRESHOLD", "0.1")
    ),
    "decoded_token_threshold": float(
        os.getenv("D3LLM_LLADA_DECODED_TOKEN_THRESHOLD", "0.95")
    ),
    "threshold": float(os.getenv("D3LLM_LLADA_THRESHOLD", "0.5")),
    "cache_delay_iter": int(os.getenv("D3LLM_LLADA_CACHE_DELAY_ITER", "2")),
    "remasking": os.getenv("D3LLM_LLADA_REMASKING", "low_confidence"),
    "mask_id": int(os.getenv("D3LLM_LLADA_MASK_ID", "126336")),
}
LLADA_DEFAULT_MAX_NEW_TOKENS = int(os.getenv("D3LLM_LLADA_MAX_TOKENS", "256"))


class Message(BaseModel):
    role: str = Field(
        ..., description="Role of the message sender (system|user|assistant)"
    )
    content: str = Field(..., description="Message content")


class ExtraBody(BaseModel):
    diffusion_steps: Optional[int] = Field(
        default=None, description="Override steps -> generation steps"
    )
    diffusion_threshold: Optional[float] = Field(
        default=None, description="Override threshold -> entropy threshold"
    )
    block_length: Optional[int] = Field(
        default=None, description="Override block_length -> block_size"
    )


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model ID, e.g., d3LLM-Dream or d3LLM-LLaDA")
    messages: List[Message]
    max_tokens: int = Field(256, description="Maximum new tokens to generate")
    temperature: float = Field(0.0, ge=0.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling top-p")
    stream: bool = Field(False, description="Stream responses as Server-Sent Events")
    extra_body: Optional[ExtraBody] = Field(
        default=None,
        description="Optional overrides: block_length->block_size, diffusion_threshold->threshold, diffusion_steps->steps",
    )


class Choice(BaseModel):
    index: int
    finish_reason: Optional[str] = None
    message: Optional[Dict[str, str]] = None
    delta: Optional[Dict[str, str]] = None


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
    usage: Optional[Usage] = None
    system_fingerprint: str = "d3llm-fastapi"


def validate_device_available(device: str, model_id: str) -> None:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise HTTPException(
                status_code=503,
                detail=f"CUDA not available but requested for {model_id}",
            )
        try:
            torch.cuda.get_device_capability(torch.device(device))
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Invalid CUDA device for {model_id}: {device} ({exc})",
            )


def trim_generated_ids(token_ids: List[int], stop_ids: List[int]) -> List[int]:
    stop_set = set(stop_ids)
    for idx, tid in enumerate(token_ids):
        if tid in stop_set:
            return token_ids[:idx]
    return token_ids


def render_prompt(tokenizer, messages: List[Message]) -> str:
    message_dicts = [m.model_dump() for m in messages]
    try:
        return tokenizer.apply_chat_template(
            message_dicts, add_generation_prompt=True, tokenize=False
        )
    except Exception:
        joined = "\n".join(f"{m['role']}: {m['content']}" for m in message_dicts)
        return f"{joined}\nassistant:"


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    token_ids: List[int]


class BaseChatModel:
    def __init__(self, model_id: str, model_path: str, device: str):
        self.model_id = model_id
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.mask_token_id: Optional[int] = None

    def ensure_loaded(self) -> None:
        raise NotImplementedError

    def warmup(self, prompt: str, max_new_tokens: int) -> None:
        request = ChatCompletionRequest(
            model=self.model_id,
            messages=[Message(role="user", content=prompt)],
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        self.generate(request)

    def generate(self, request: ChatCompletionRequest) -> GenerationResult:
        raise NotImplementedError


def apply_extra_overrides(gen_kwargs: Dict[str, object], request: ChatCompletionRequest) -> None:
    """Map OpenAI-style extra_body fields to generation kwargs."""
    extra = request.extra_body
    if not extra:
        return
    if extra.block_length is not None:
        gen_kwargs["block_size"] = extra.block_length
    if extra.diffusion_threshold is not None:
        gen_kwargs["threshold"] = extra.diffusion_threshold
    if extra.diffusion_steps is not None:
        gen_kwargs["steps"] = extra.diffusion_steps


class DreamChatModel(BaseChatModel):
    def ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        validate_device_available(self.device, self.model_id)
        torch_dtype = resolve_dtype(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model_config = DreamConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        try:
            model_config._attn_implementation = "flash_attention_2"
        except Exception:
            pass

        model = DreamModel.from_pretrained(
            self.model_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        model = model.to(self.device).eval()

        if ENABLE_COMPILE and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as exc:
                print(f"[dream] torch.compile disabled: {exc}")

        model.generate_multi_block = types.MethodType(
            DreamGenerationMixin.generate_multi_block, model
        )
        model._sample_multi_block = types.MethodType(
            DreamGenerationMixin._sample_multi_block, model
        )
        model._sample_multi_block_kv_cache = types.MethodType(
            DreamGenerationMixin._sample_multi_block_kv_cache, model
        )
        model._prepare_inputs = types.MethodType(
            DreamGenerationMixin._prepare_inputs, model
        )

        self.mask_token_id = (
            getattr(model_config, "mask_token_id", None)
            or getattr(model.generation_config, "mask_token_id", None)
            or getattr(self.tokenizer, "mask_token_id", None)
        )
        self.model = model

    def generate(self, request: ChatCompletionRequest) -> GenerationResult:
        self.ensure_loaded()
        prompt = render_prompt(self.tokenizer, request.messages)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(
            self.device
        )
        prompt_len = input_ids.shape[-1]

        gen_kwargs = dict(DREAM_GENERATION_DEFAULTS)
        gen_kwargs["max_new_tokens"] = request.max_tokens or DREAM_DEFAULT_MAX_NEW_TOKENS
        gen_kwargs["temperature"] = request.temperature
        gen_kwargs["top_p"] = request.top_p
        gen_kwargs["return_dict_in_generate"] = True
        apply_extra_overrides(gen_kwargs, request)

        with torch.no_grad():
            output, _ = self.model.generate_multi_block(input_ids, **gen_kwargs)

        sequences = output.sequences if hasattr(output, "sequences") else output
        generated_ids = sequences[:, prompt_len:].tolist()[0]
        stop_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.mask_token_id,
        ]
        generated_ids = trim_generated_ids(generated_ids, [i for i in stop_ids if i is not None])

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completion_tokens = len(generated_ids)
        return GenerationResult(text, prompt_len, completion_tokens, generated_ids)


class LLaDAChatModel(BaseChatModel):
    def ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        validate_device_available(self.device, self.model_id)
        torch_dtype = resolve_dtype(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(self.model_path)
        config.flash_attention = True

        self.model = (
            LLaDAModelLM.from_pretrained(
                self.model_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            .to(self.device)
            .eval()
        )
        self.mask_token_id = LLADA_GENERATION_DEFAULTS["mask_id"]

    def generate(self, request: ChatCompletionRequest) -> GenerationResult:
        self.ensure_loaded()
        prompt = render_prompt(self.tokenizer, request.messages)
        input_ids = self.tokenizer(prompt)["input_ids"]
        input_ids = torch.tensor(input_ids, device=self.device).unsqueeze(0)
        prompt_len = input_ids.shape[-1]

        gen_kwargs = dict(LLADA_GENERATION_DEFAULTS)
        gen_kwargs["max_new_tokens"] = request.max_tokens or LLADA_DEFAULT_MAX_NEW_TOKENS
        gen_kwargs["temperature"] = request.temperature
        apply_extra_overrides(gen_kwargs, request)

        with torch.no_grad():
            output, _ = generate_multi_block_kv_cache(
                self.model, input_ids, **gen_kwargs
            )

        generated_ids = output[0, prompt_len:].tolist()
        stop_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.mask_token_id,
        ]
        generated_ids = trim_generated_ids(generated_ids, [i for i in stop_ids if i is not None])

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completion_tokens = len(generated_ids)
        return GenerationResult(text, prompt_len, completion_tokens, generated_ids)


BACKEND_BUILDERS = {
    "d3LLM-Dream": DreamChatModel,
    "d3LLM-LLaDA": LLaDAChatModel,
}

_MODEL_CACHE: Dict[str, BaseChatModel] = {}


def get_model_bundle(model_id: str) -> BaseChatModel:
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    if model_id not in _MODEL_CACHE:
        backend_cls = BACKEND_BUILDERS.get(model_id)
        if backend_cls is None:
            raise HTTPException(status_code=404, detail=f"No backend for model: {model_id}")
        _MODEL_CACHE[model_id] = backend_cls(
            model_id, MODEL_REGISTRY[model_id], MODEL_DEVICES[model_id]
        )
    bundle = _MODEL_CACHE[model_id]
    bundle.ensure_loaded()
    return bundle


def generate_response(
    bundle: BaseChatModel,
    request: ChatCompletionRequest,
) -> GenerationResult:
    return bundle.generate(request)


def format_stream_chunk(
    completion_id: str,
    created: int,
    model_id: str,
    delta: Dict[str, str],
    finish_reason: Optional[str],
    usage: Optional[Usage] = None,
    extra_fields: Optional[Dict[str, object]] = None,
) -> str:
    payload: Dict[str, object] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
        "system_fingerprint": "d3llm-fastapi",
    }
    if usage is not None:
        payload["usage"] = usage.model_dump()
    if extra_fields:
        payload.update(extra_fields)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stream_chat_response(
    bundle: BaseChatModel, request: ChatCompletionRequest
):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    start = time.time()
    try:
        result = generate_response(bundle, request)
        elapsed = time.time() - start
        usage = Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        )

        yield format_stream_chunk(
            completion_id,
            created,
            request.model,
            delta={"role": "assistant"},
            finish_reason=None,
        )

        for token_id in result.token_ids:
            token_text = bundle.tokenizer.decode([token_id], skip_special_tokens=True)
            if not token_text:
                continue
            yield format_stream_chunk(
                completion_id,
                created,
                request.model,
                delta={"content": token_text},
                finish_reason=None,
            )

        yield format_stream_chunk(
            completion_id,
            created,
            request.model,
            delta={},
            finish_reason="stop",
            usage=usage,
            extra_fields={"latency_sec": round(elapsed, 3)},
        )
        yield "data: [DONE]\n\n"
    except Exception as exc:
        error_payload = {
            "error": {
                "message": str(exc),
                "type": "internal_error",
            }
        }
        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"


app = FastAPI(title="d3LLM API", version="0.1.0")


@app.on_event("startup")
async def preload_models() -> None:
    if not PRELOAD_MODELS:
        return
    for model_id in MODEL_REGISTRY:
        try:
            bundle = get_model_bundle(model_id)
            bundle.warmup(WARMUP_PROMPT, WARMUP_TOKENS)
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
) -> ChatCompletionResponse | StreamingResponse:
    bundle = get_model_bundle(request.model)
    if request.stream:
        return StreamingResponse(
            stream_chat_response(bundle, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    start = time.time()
    result = generate_response(bundle, request)
    elapsed = time.time() - start

    choice = Choice(
        index=0,
        finish_reason="stop",
        message={"role": "assistant", "content": result.text},
    )
    usage = Usage(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.prompt_tokens + result.completion_tokens,
    )
    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[choice],
        usage=usage,
    )
    response_dict = response.model_dump()
    response_dict["latency_sec"] = round(elapsed, 3)
    return response_dict


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", os.getenv("PORT", 18091)))
    uvicorn.run("api_server:app", host=host, port=port, log_level="info")
