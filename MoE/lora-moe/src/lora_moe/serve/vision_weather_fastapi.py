from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from peft import PeftModel
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_moe.components import FeatureToSoftPromptProjector, FrozenWeatherVisionEncoder
from lora_moe.train.vision_weather_lora import dtype_from_name


DEFAULT_MODEL_DIR = "/home/wdz/BT/MoE/models/Qwen2.5-14B-Instruct"
DEFAULT_OUTPUT_DIR = "/home/wdz/BT/MoE/lora-moe/outputs/vision_weather_lora_qv_v1"
DEFAULT_VISION_WEIGHTS = (
    "/home/wdz/BT/Stage1/vision_weather/weights/"
    "20260605_182036_weather_cls_more_cloudy_gpu_30ep_best_model.pt"
)
LABEL_ZH = {
    "sunny": "晴天",
    "cloudy": "多云",
    "rain": "下雨",
}


class GenerateRequest(BaseModel):
    prompt: str = "请识别这张图片的天气，只回答一句中文结论。"
    image_base64: str
    image_name: Optional[str] = None
    max_new_tokens: int = Field(default=32, ge=1, le=256)
    temperature: float = Field(default=0.0, ge=0.0)


class GenerateResponse(BaseModel):
    prompt: str
    model_prompt: str
    generated_text: str
    pred_label_from_text: Optional[str] = None
    pred_label_zh_from_text: Optional[str] = None
    input_soft_tokens: int
    output_tokens: int
    image_received: bool
    image_name: Optional[str] = None
    image_bytes: int = 0
    modality_status: str
    vision_encoder: dict
    artifacts: dict


def resolve_artifacts(output_dir: str, adapter_dir: str, projector_path: str, use_best: bool) -> tuple[Path, Path]:
    root = Path(output_dir).expanduser()
    resolved_adapter = Path(adapter_dir).expanduser() if adapter_dir else root / ("best/adapter" if use_best else "adapter")
    resolved_projector = (
        Path(projector_path).expanduser() if projector_path else root / ("best/projector.pt" if use_best else "projector.pt")
    )
    if not resolved_adapter.exists():
        raise FileNotFoundError(f"adapter not found: {resolved_adapter}")
    if not resolved_projector.exists():
        raise FileNotFoundError(f"projector not found: {resolved_projector}")
    return resolved_adapter, resolved_projector


def preprocess_pil_image(img: Image.Image, image_size: int) -> torch.Tensor:
    img = img.convert("RGB")
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(3, 1, 1)
    return (x - mean) / std


def parse_prediction(text: str) -> Optional[str]:
    lowered = text.lower()
    if "多云" in text or "cloudy" in lowered:
        return "cloudy"
    if "晴天" in text or "晴朗" in text or "sunny" in lowered:
        return "sunny"
    if "下雨" in text or "雨天" in text or "rain" in lowered:
        return "rain"
    return None


def token_ids(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def build_prompt_ids(tokenizer, prompt: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    user_prompt = prompt.strip() or "请识别这张图片的天气，只回答一句中文结论。"
    prefix = (
        "<|im_start|>system\n"
        "你是气象图像识别助手。请根据用户上传的图片判断天气。"
        "回答要自然、简洁，除非用户明确要求英文，否则使用中文。"
        "不要提到视觉token、编码器、特征、模型内部表示等实现细节。\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<图片>"
    )
    suffix = f"\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    return token_ids(tokenizer, prefix), token_ids(tokenizer, suffix), prefix + suffix


class VisionWeatherRunner:
    def __init__(
        self,
        model_dir: str,
        output_dir: str,
        adapter_dir: str,
        projector_path: str,
        vision_weights: str,
        use_best: bool,
        device_map: str,
        dtype: str,
    ) -> None:
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.device_map = device_map
        self.dtype = dtype
        self.adapter_dir, self.projector_path = resolve_artifacts(output_dir, adapter_dir, projector_path, use_best)
        self.lock = threading.Lock()

        torch_dtype = dtype_from_name(dtype)
        print(f"Loading tokenizer from {model_dir}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        print(f"Loading Qwen base from {model_dir} with device_map={device_map}, dtype={dtype}...", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        print(f"Loading LoRA adapter from {self.adapter_dir}...", flush=True)
        self.model = PeftModel.from_pretrained(base_model, self.adapter_dir)
        self.model.eval()
        self.input_device = next(self.model.get_input_embeddings().parameters()).device

        projector_ckpt = torch.load(self.projector_path, map_location="cpu")
        self.projector = FeatureToSoftPromptProjector(
            input_dim=int(projector_ckpt["input_dim"]),
            hidden_dim=int(projector_ckpt["hidden_dim"]),
            output_dim=self.model.config.hidden_size,
            num_tokens=int(projector_ckpt["num_tokens"]),
            dropout=0.0,
        )
        self.projector.load_state_dict(projector_ckpt["state_dict"])
        self.projector.to(self.input_device, dtype=torch_dtype)
        self.projector.eval()

        self.vision_weights = vision_weights or str(projector_ckpt["vision_weights"])
        print(f"Loading frozen weather vision encoder from {self.vision_weights}...", flush=True)
        self.vision_encoder = FrozenWeatherVisionEncoder(
            weights=self.vision_weights,
            device=self.input_device,
            freeze=True,
        )
        self.vision_encoder.eval()
        self.num_visual_tokens = int(projector_ckpt["num_tokens"])
        print("Vision-weather LoRA service loaded.", flush=True)

    @torch.inference_mode()
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        raw = base64.b64decode(request.image_base64)
        with Image.open(io.BytesIO(raw)) as img:
            pixel_values = preprocess_pil_image(img, self.vision_encoder.image_size)

        pixel_values = pixel_values.unsqueeze(0).to(self.input_device, dtype=torch.float32)
        features = self.vision_encoder(pixel_values)
        visual_embeds = self.projector(features.to(self.input_device, dtype=next(self.projector.parameters()).dtype))
        visual_embeds = visual_embeds.to(dtype=self.model.get_input_embeddings().weight.dtype)

        prefix_ids, suffix_ids, model_prompt = build_prompt_ids(self.tokenizer, request.prompt)
        embedding = self.model.get_input_embeddings()
        prefix_embeds = embedding(prefix_ids.to(self.input_device)).unsqueeze(0)
        suffix_embeds = embedding(suffix_ids.to(self.input_device)).unsqueeze(0)
        inputs_embeds = torch.cat([prefix_embeds, visual_embeds, suffix_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.input_device, dtype=torch.long)

        do_sample = request.temperature > 0
        with self.lock:
            output_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=request.max_new_tokens,
                do_sample=do_sample,
                temperature=request.temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        cls = self.vision_encoder.classify(pixel_values)
        probs = {
            name: round(float(cls["probs"][0, idx].detach().cpu()), 6)
            for idx, name in enumerate(self.vision_encoder.class_names)
        }
        pred_label = parse_prediction(generated_text)
        return GenerateResponse(
            prompt=request.prompt,
            model_prompt=model_prompt,
            generated_text=generated_text,
            pred_label_from_text=pred_label,
            pred_label_zh_from_text=LABEL_ZH.get(pred_label) if pred_label else None,
            input_soft_tokens=self.num_visual_tokens,
            output_tokens=int(output_ids.shape[-1]),
            image_received=True,
            image_name=request.image_name,
            image_bytes=len(raw),
            modality_status="vision_encoder_projector_lora_connected",
            vision_encoder={
                "pred_label": cls["pred_label"][0],
                "pred_label_zh": LABEL_ZH.get(cls["pred_label"][0], cls["pred_label"][0]),
                "probs": probs,
                "feature_dim": self.vision_encoder.out_dim,
                "image_size": self.vision_encoder.image_size,
                "vision_weights": self.vision_weights,
            },
            artifacts={
                "model_dir": self.model_dir,
                "adapter_dir": str(self.adapter_dir),
                "projector_path": str(self.projector_path),
                "device_map": self.device_map,
                "dtype": self.dtype,
            },
        )


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LoRA-MoE 视觉天气推理</title>
  <style>
    :root { --bg:#f6f7f9; --panel:#fff; --text:#1f2933; --muted:#6b7280; --line:#d8dee8; --accent:#0f766e; --accent-strong:#0b5f59; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }
    h1 { width:min(1120px,calc(100vw - 32px)); margin:24px auto 0; font-size:22px; font-weight:650; }
    main { width:min(1120px,calc(100vw - 32px)); margin:24px auto; display:grid; grid-template-columns:380px 1fr; gap:16px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }
    label { display:block; font-size:13px; color:var(--muted); margin-bottom:6px; }
    textarea,input[type="number"] { width:100%; border:1px solid var(--line); border-radius:6px; padding:10px; font:inherit; background:#fff; }
    textarea { min-height:120px; resize:vertical; line-height:1.5; }
    .field { margin-top:14px; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }
    button { width:100%; height:42px; margin-top:16px; border:0; border-radius:6px; background:var(--accent); color:#fff; font:inherit; font-weight:650; cursor:pointer; }
    button:hover { background:var(--accent-strong); }
    button:disabled { cursor:wait; opacity:.65; }
    .small-button { width:auto; height:34px; margin-top:0; padding:0 12px; background:#eef2f7; color:var(--text); }
    .small-button:hover { background:#e4e9f1; }
    #image { display:none; }
    .tools { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-top:10px; }
    .file-name { color:var(--muted); font-size:13px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; max-width:180px; }
    .preview { width:100%; min-height:140px; margin-top:10px; border:1px dashed var(--line); border-radius:6px; display:grid; place-items:center; overflow:hidden; color:var(--muted); background:#fbfcfd; }
    .preview img { width:100%; max-height:260px; object-fit:contain; display:block; }
    .answer { min-height:420px; white-space:pre-wrap; line-height:1.6; border:1px solid var(--line); border-radius:6px; padding:14px; background:#fbfcfd; }
    .meta { margin-top:12px; color:var(--muted); font-size:13px; line-height:1.6; white-space:pre-wrap; }
    .status { margin-top:12px; color:var(--muted); font-size:13px; min-height:20px; }
    @media (max-width:820px) { main { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <h1>LoRA-MoE 视觉天气推理</h1>
  <main>
    <section>
      <div class="field" style="margin-top:0">
        <label for="prompt">输入内容</label>
        <textarea id="prompt">请识别这张图片的天气，只回答一句中文结论。</textarea>
      </div>
      <div class="field">
        <label>图片</label>
        <input id="image" type="file" accept="image/*" />
        <div class="preview" id="preview">未选择图片</div>
        <div class="tools">
          <div>
            <button class="small-button" id="chooseImage" type="button">选择图片</button>
            <button class="small-button" id="clearImage" type="button">清除</button>
          </div>
          <div class="file-name" id="fileName">无图片</div>
        </div>
      </div>
      <div class="row">
        <div>
          <label for="maxNewTokens">输出 tokens</label>
          <input id="maxNewTokens" type="number" min="1" max="256" value="32" />
        </div>
        <div>
          <label for="temperature">temperature</label>
          <input id="temperature" type="number" min="0" step="0.1" value="0.0" />
        </div>
      </div>
      <button id="send">发送</button>
      <div class="status" id="status"></div>
    </section>
    <section>
      <label>模型输出</label>
      <div class="answer" id="answer"></div>
      <div class="meta" id="meta"></div>
    </section>
  </main>
  <script>
    const imageInput = document.getElementById("image");
    const chooseImage = document.getElementById("chooseImage");
    const clearImage = document.getElementById("clearImage");
    const fileName = document.getElementById("fileName");
    const preview = document.getElementById("preview");
    const send = document.getElementById("send");
    const statusEl = document.getElementById("status");
    const answer = document.getElementById("answer");
    const meta = document.getElementById("meta");
    let imageBase64 = null;
    let imageName = null;
    chooseImage.addEventListener("click", () => imageInput.click());
    clearImage.addEventListener("click", () => {
      imageInput.value = ""; imageBase64 = null; imageName = null;
      fileName.textContent = "无图片"; preview.textContent = "未选择图片";
    });
    imageInput.addEventListener("change", () => {
      const file = imageInput.files[0]; imageBase64 = null; imageName = null;
      if (!file) { fileName.textContent = "无图片"; preview.textContent = "未选择图片"; return; }
      imageName = file.name; fileName.textContent = file.name;
      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = reader.result;
        imageBase64 = String(dataUrl).split(",")[1] || "";
        preview.innerHTML = "";
        const img = document.createElement("img");
        img.src = dataUrl; img.alt = file.name; preview.appendChild(img);
      };
      reader.readAsDataURL(file);
    });
    send.addEventListener("click", async () => {
      if (!imageBase64) { statusEl.textContent = "请先选择图片"; return; }
      send.disabled = true; statusEl.textContent = "生成中..."; answer.textContent = ""; meta.textContent = "";
      try {
        const payload = {
          prompt: document.getElementById("prompt").value,
          image_base64: imageBase64,
          image_name: imageName,
          max_new_tokens: Number(document.getElementById("maxNewTokens").value),
          temperature: Number(document.getElementById("temperature").value)
        };
        const res = await fetch("/generate", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
        answer.textContent = data.generated_text || "";
        meta.textContent =
          `pred_label_from_text: ${data.pred_label_from_text || ""}\n` +
          `pred_label_zh_from_text: ${data.pred_label_zh_from_text || ""}\n` +
          `output_tokens: ${data.output_tokens}\n` +
          `input_soft_tokens: ${data.input_soft_tokens}\n` +
          `modality_status: ${data.modality_status}\n` +
          `image_name: ${data.image_name || ""}\n` +
          `image_bytes: ${data.image_bytes || 0}\n\n` +
          `vision_encoder:\n${JSON.stringify(data.vision_encoder || {}, null, 2)}\n\n` +
          `artifacts:\n${JSON.stringify(data.artifacts || {}, null, 2)}`;
        statusEl.textContent = "完成";
      } catch (err) {
        statusEl.textContent = "请求失败"; answer.textContent = String(err);
      } finally {
        send.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def create_app(runner: VisionWeatherRunner) -> FastAPI:
    app = FastAPI(title="LoRA-MoE Vision Weather API", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_dir": runner.model_dir,
            "adapter_dir": str(runner.adapter_dir),
            "projector_path": str(runner.projector_path),
            "vision_weights": runner.vision_weights,
            "device_map": runner.device_map,
            "dtype": runner.dtype,
            "input_device": str(runner.input_device),
            "num_visual_tokens": runner.num_visual_tokens,
            "vision_classes": runner.vision_encoder.class_names,
        }

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest):
        return runner.generate(request)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--projector-path", default="")
    parser.add_argument("--vision-weights", default=DEFAULT_VISION_WEIGHTS)
    parser.add_argument("--use-best", action="store_true", default=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    runner = VisionWeatherRunner(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        adapter_dir=args.adapter_dir,
        projector_path=args.projector_path,
        vision_weights=args.vision_weights,
        use_best=args.use_best,
        device_map=args.device_map,
        dtype=args.dtype,
    )
    app = create_app(runner)
    print(f"Serving LoRA-MoE vision-weather FastAPI on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
