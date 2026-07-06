from __future__ import annotations

import argparse
import json
import os
import threading
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field


DEFAULT_MODEL_DIR = "/home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct"


class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str = "你是一个严谨、简洁的中文助手。"
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=0.1)


class GenerateResponse(BaseModel):
    prompt: str
    system_prompt: str
    generated_text: str
    input_tokens: int
    output_tokens: int
    role_name: str
    model_dir: str
    device_map: str
    dtype: str
    input_device: str


def _dtype_from_name(name: str):
    import torch

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


class GeneralQwenRunner:
    def __init__(
        self,
        *,
        model_dir: str,
        role_name: str,
        device_map: str,
        dtype: str,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_dir = model_dir
        self.role_name = role_name
        self.device_map = device_map
        self.dtype = dtype
        self.lock = threading.Lock()

        torch_dtype = _dtype_from_name(dtype)
        print(f"Loading tokenizer from {model_dir}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        print(
            f"Loading Qwen role={role_name} from {model_dir} "
            f"with device_map={device_map}, dtype={dtype}...",
            flush=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.input_device = next(self.model.get_input_embeddings().parameters()).device
        print(f"General Qwen service loaded: role={role_name}, input_device={self.input_device}", flush=True)

    def _build_inputs(self, request: GenerateRequest):
        import torch

        messages = [
            {"role": "system", "content": request.system_prompt.strip()},
            {"role": "user", "content": request.prompt.strip()},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            text = f"{request.system_prompt.strip()}\n\n用户：{request.prompt.strip()}\n助手："
            input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(self.input_device)
        attention_mask = torch.ones_like(input_ids, device=self.input_device)
        return input_ids, attention_mask

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        import torch

        input_ids, attention_mask = self._build_inputs(request)
        do_sample = request.temperature > 0
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": request.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": request.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = request.temperature
            generate_kwargs["top_p"] = request.top_p

        with torch.inference_mode(), self.lock:
            output_ids = self.model.generate(**generate_kwargs)

        generated_ids = output_ids[0, input_ids.shape[-1] :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return GenerateResponse(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            generated_text=generated_text,
            input_tokens=int(input_ids.shape[-1]),
            output_tokens=int(generated_ids.shape[-1]),
            role_name=self.role_name,
            model_dir=self.model_dir,
            device_map=self.device_map,
            dtype=self.dtype,
            input_device=str(self.input_device),
        )

    def stream_generate(self, request: GenerateRequest):
        import torch
        from transformers import TextIteratorStreamer

        input_ids, attention_mask = self._build_inputs(request)
        do_sample = request.temperature > 0
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": request.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": request.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        if do_sample:
            generate_kwargs["temperature"] = request.temperature
            generate_kwargs["top_p"] = request.top_p

        error_holder: list[str] = []

        def _run_generate() -> None:
            try:
                with torch.inference_mode(), self.lock:
                    self.model.generate(**generate_kwargs)
            except Exception as exc:  # pragma: no cover - surfaced through stream
                error_holder.append(repr(exc))

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()

        yield {
            "type": "meta",
            "role_name": self.role_name,
            "input_tokens": int(input_ids.shape[-1]),
            "model_dir": self.model_dir,
            "device_map": self.device_map,
            "dtype": self.dtype,
            "input_device": str(self.input_device),
        }
        generated_text_parts: list[str] = []
        for text in streamer:
            if text:
                generated_text_parts.append(text)
                yield {"type": "chunk", "text": text}
        thread.join()
        if error_holder:
            yield {"type": "error", "detail": error_holder[0]}
            return
        generated_text = "".join(generated_text_parts)
        output_ids = self.tokenizer(generated_text, add_special_tokens=False)["input_ids"]
        yield {
            "type": "done",
            "generated_text": generated_text.strip(),
            "input_tokens": int(input_ids.shape[-1]),
            "output_tokens": len(output_ids),
            "role_name": self.role_name,
            "model_dir": self.model_dir,
            "device_map": self.device_map,
            "dtype": self.dtype,
            "input_device": str(self.input_device),
        }


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>General Qwen</title>
  <style>
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f6f7f9; color:#1f2933; }
    main { width:min(980px,calc(100vw - 32px)); margin:24px auto; display:grid; gap:14px; }
    section { background:#fff; border:1px solid #d8dee8; border-radius:8px; padding:16px; }
    label { display:block; margin-bottom:6px; color:#667085; font-size:13px; }
    textarea,input { width:100%; box-sizing:border-box; border:1px solid #d8dee8; border-radius:6px; padding:10px; font:inherit; }
    textarea { min-height:140px; resize:vertical; line-height:1.5; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    button { height:42px; border:0; border-radius:6px; background:#0f766e; color:#fff; font:inherit; font-weight:650; cursor:pointer; }
    pre { white-space:pre-wrap; word-break:break-word; background:#f8fafc; border:1px solid #d8dee8; border-radius:6px; padding:12px; }
  </style>
</head>
<body>
  <main>
    <section>
      <label>System Prompt</label>
      <textarea id="systemPrompt">你是一个严谨、简洁的中文助手。</textarea>
      <label>Prompt</label>
      <textarea id="prompt">请用三句话说明什么是服务层多智能体。</textarea>
      <div class="row">
        <div><label>max_new_tokens</label><input id="maxNewTokens" type="number" value="256" min="1" max="2048" /></div>
        <div><label>temperature</label><input id="temperature" type="number" value="0.0" min="0" step="0.1" /></div>
      </div>
      <button id="send">发送</button>
    </section>
    <section>
      <label>输出</label>
      <pre id="answer"></pre>
    </section>
  </main>
  <script>
    document.getElementById("send").addEventListener("click", async () => {
      const answer = document.getElementById("answer");
      answer.textContent = "生成中...";
      const payload = {
        system_prompt: document.getElementById("systemPrompt").value,
        prompt: document.getElementById("prompt").value,
        max_new_tokens: Number(document.getElementById("maxNewTokens").value),
        temperature: Number(document.getElementById("temperature").value)
      };
      const res = await fetch("/generate", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
      const data = await res.json();
      answer.textContent = res.ok ? data.generated_text : JSON.stringify(data, null, 2);
    });
  </script>
</body>
</html>
"""


def create_app(runner: GeneralQwenRunner) -> FastAPI:
    app = FastAPI(title="General Qwen API", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "service": "general_qwen",
            "role_name": runner.role_name,
            "model_dir": runner.model_dir,
            "device_map": runner.device_map,
            "dtype": runner.dtype,
            "input_device": str(runner.input_device),
        }

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest):
        return runner.generate(request)

    @app.post("/generate/stream")
    def generate_stream(request: GenerateRequest):
        def _events():
            for event in runner.stream_generate(request):
                yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

        return StreamingResponse(_events(), media_type="application/x-ndjson")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--role-name", default="qwen")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8131)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    runner = GeneralQwenRunner(
        model_dir=args.model_dir,
        role_name=args.role_name,
        device_map=args.device_map,
        dtype=args.dtype,
    )
    app = create_app(runner)
    print(f"Serving General Qwen role={args.role_name} on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
