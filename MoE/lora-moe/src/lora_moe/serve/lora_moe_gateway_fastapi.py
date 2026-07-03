from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


DEFAULT_VISION_URL = "http://127.0.0.1:8121"
DEFAULT_STAGE1_URL = "http://127.0.0.1:8122"


class GenerateRequest(BaseModel):
    prompt: str = "请根据当前输入判断应该调用哪个专家，并给出结果。"
    image_base64: Optional[str] = None
    image_name: Optional[str] = None
    max_new_tokens: int = Field(default=80, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0)
    task_mode: str = "auto"


class GatewayResponse(BaseModel):
    prompt: str
    route: str
    expert_url: str
    generated_text: str
    modality_status: str
    elapsed_ms: float
    raw_response: dict[str, Any]


class RouteDebugResponse(BaseModel):
    route: str
    reason: str
    expert_url: str


def _json_request(method: str, url: str, payload: Optional[dict[str, Any]], timeout_s: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"expert service unreachable: {url}; {exc}") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"expert service timeout: {url}") from exc


def _strip_data_url_prefix(image_base64: Optional[str]) -> Optional[str]:
    if not image_base64:
        return None
    if "," in image_base64 and image_base64.split(",", 1)[0].startswith("data:"):
        return image_base64.split(",", 1)[1]
    return image_base64


class GatewayRunner:
    def __init__(
        self,
        *,
        vision_url: str,
        stage1_url: str,
        request_timeout_s: float,
    ) -> None:
        self.vision_url = vision_url.rstrip("/")
        self.stage1_url = stage1_url.rstrip("/")
        self.request_timeout_s = float(request_timeout_s)

    def route(self, request: GenerateRequest) -> tuple[str, str]:
        mode = (request.task_mode or "auto").strip().lower()
        has_image = bool(_strip_data_url_prefix(request.image_base64))
        prompt = (request.prompt or "").lower()

        if mode in {"vision", "image", "vision_weather", "weather_image"}:
            return "vision_weather", "task_mode forced vision"
        if mode in {"stage1", "rain", "rainfall", "inversion", "link"}:
            return "stage1_rain", "task_mode forced stage1"
        if mode in {"text", "general", "none", "off"}:
            return "stage1_text", "task_mode forced text fallback"

        if has_image:
            return "vision_weather", "image provided"

        stage1_keywords = (
            "反演",
            "链路",
            "当前降雨",
            "当前雨量",
            "过境",
            "卫星",
            "rainfall inversion",
            "satellite pass",
            "link",
        )
        if any(keyword in prompt for keyword in stage1_keywords):
            return "stage1_rain", "stage1 keyword matched"

        return "stage1_text", "no expert-specific signal; use text fallback"

    def route_debug(self, request: GenerateRequest) -> RouteDebugResponse:
        route, reason = self.route(request)
        return RouteDebugResponse(
            route=route,
            reason=reason,
            expert_url=self.vision_url if route == "vision_weather" else self.stage1_url,
        )

    def generate(self, request: GenerateRequest) -> GatewayResponse:
        started = time.perf_counter()
        route, _ = self.route(request)

        if route == "vision_weather":
            image_base64 = _strip_data_url_prefix(request.image_base64)
            if not image_base64:
                raise HTTPException(status_code=400, detail="vision_weather expert requires image_base64")
            expert_url = self.vision_url
            payload = {
                "prompt": request.prompt,
                "image_base64": image_base64,
                "image_name": request.image_name,
                "max_new_tokens": min(request.max_new_tokens, 256),
                "temperature": request.temperature,
            }
        elif route == "stage1_rain":
            expert_url = self.stage1_url
            payload = {
                "prompt": request.prompt,
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "task_mode": "stage1",
            }
        else:
            expert_url = self.stage1_url
            payload = {
                "prompt": request.prompt,
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "task_mode": "text",
            }

        raw = _json_request("POST", f"{expert_url}/generate", payload, self.request_timeout_s)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return GatewayResponse(
            prompt=request.prompt,
            route=route,
            expert_url=expert_url,
            generated_text=str(raw.get("generated_text", "")),
            modality_status=str(raw.get("modality_status", route)),
            elapsed_ms=elapsed_ms,
            raw_response=raw,
        )

    def health(self) -> dict[str, Any]:
        experts = {}
        for name, base_url in (("vision_weather", self.vision_url), ("stage1_rain", self.stage1_url)):
            try:
                experts[name] = _json_request("GET", f"{base_url}/health", None, min(self.request_timeout_s, 10.0))
            except HTTPException as exc:
                experts[name] = {"status": "error", "detail": exc.detail}
        return {
            "status": "ok",
            "service": "lora_moe_gateway",
            "vision_url": self.vision_url,
            "stage1_url": self.stage1_url,
            "experts": experts,
        }

    def stage1_latest(self) -> dict[str, Any]:
        return _json_request("GET", f"{self.stage1_url}/stage1/latest", None, self.request_timeout_s)

    def stage1_tick(self) -> dict[str, Any]:
        return _json_request("POST", f"{self.stage1_url}/stage1/tick", {}, self.request_timeout_s)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LoRA-MoE Gateway</title>
  <style>
    :root { --bg:#f5f6f8; --panel:#fff; --text:#1f2933; --muted:#667085; --line:#d8dee8; --accent:#0f766e; --accent-strong:#0b5f59; --danger:#b42318; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }
    header { width:min(1160px,calc(100vw - 32px)); margin:22px auto 10px; display:flex; justify-content:space-between; align-items:center; gap:12px; }
    h1 { margin:0; font-size:22px; font-weight:650; }
    main { width:min(1160px,calc(100vw - 32px)); margin:0 auto 24px; display:grid; grid-template-columns:390px 1fr; gap:16px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }
    label { display:block; font-size:13px; color:var(--muted); margin-bottom:6px; }
    textarea,input,select { width:100%; border:1px solid var(--line); border-radius:6px; padding:10px; font:inherit; background:#fff; }
    textarea { min-height:150px; resize:vertical; line-height:1.5; }
    .field { margin-top:14px; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    button { height:42px; border:0; border-radius:6px; background:var(--accent); color:#fff; font:inherit; font-weight:650; cursor:pointer; padding:0 14px; }
    button:hover { background:var(--accent-strong); }
    button:disabled { cursor:wait; opacity:.65; }
    .secondary { background:#344054; }
    .status { color:var(--muted); font-size:13px; }
    .answer { white-space:pre-wrap; line-height:1.65; min-height:160px; font-size:15px; }
    pre { white-space:pre-wrap; word-break:break-word; background:#f8fafc; border:1px solid var(--line); border-radius:6px; padding:12px; max-height:520px; overflow:auto; font-size:12px; }
    img { display:block; width:100%; max-height:260px; object-fit:contain; border:1px solid var(--line); border-radius:6px; background:#f8fafc; }
    @media (max-width: 860px) { main { grid-template-columns:1fr; } header { align-items:flex-start; flex-direction:column; } }
  </style>
</head>
<body>
  <header>
    <h1>LoRA-MoE Gateway</h1>
    <div class="status" id="status">就绪</div>
  </header>
  <main>
    <section>
      <label for="prompt">输入</label>
      <textarea id="prompt">请根据当前输入给出结果。</textarea>

      <div class="field">
        <label for="image">图片</label>
        <input id="image" type="file" accept="image/*" />
      </div>
      <div class="field" id="previewBox" style="display:none">
        <img id="preview" alt="" />
      </div>

      <div class="row field">
        <div>
          <label for="taskMode">专家</label>
          <select id="taskMode">
            <option value="auto">自动</option>
            <option value="vision">视觉天气</option>
            <option value="stage1">链路反演</option>
            <option value="text">纯文本</option>
          </select>
        </div>
        <div>
          <label for="tokens">输出长度</label>
          <input id="tokens" type="number" value="80" min="1" max="512" />
        </div>
      </div>

      <div class="row field">
        <button id="send">发送</button>
        <button class="secondary" id="refresh">刷新状态</button>
      </div>
    </section>

    <section>
      <label>回答</label>
      <div class="answer" id="answer"></div>
      <div class="field">
        <label>调试信息</label>
        <pre id="meta"></pre>
      </div>
    </section>
  </main>

  <script>
    let imageBase64 = "";
    let imageName = "";
    const statusEl = document.getElementById("status");
    const answerEl = document.getElementById("answer");
    const metaEl = document.getElementById("meta");
    const imageEl = document.getElementById("image");
    const preview = document.getElementById("preview");
    const previewBox = document.getElementById("previewBox");

    imageEl.addEventListener("change", () => {
      const file = imageEl.files && imageEl.files[0];
      imageBase64 = "";
      imageName = "";
      previewBox.style.display = "none";
      if (!file) return;
      imageName = file.name;
      const reader = new FileReader();
      reader.onload = () => {
        const result = String(reader.result || "");
        imageBase64 = result.includes(",") ? result.split(",", 2)[1] : result;
        preview.src = result;
        previewBox.style.display = "block";
      };
      reader.readAsDataURL(file);
    });

    async function refreshHealth() {
      statusEl.textContent = "检查服务...";
      try {
        const res = await fetch("/health");
        const data = await res.json();
        metaEl.textContent = JSON.stringify(data, null, 2);
        statusEl.textContent = "就绪";
      } catch (err) {
        statusEl.textContent = "状态检查失败";
        metaEl.textContent = String(err);
      }
    }

    document.getElementById("refresh").addEventListener("click", refreshHealth);

    document.getElementById("send").addEventListener("click", async () => {
      const payload = {
        prompt: document.getElementById("prompt").value,
        image_base64: imageBase64 || null,
        image_name: imageName || null,
        task_mode: document.getElementById("taskMode").value,
        max_new_tokens: Number(document.getElementById("tokens").value || 80),
        temperature: 0
      };
      statusEl.textContent = "推理中...";
      answerEl.textContent = "";
      metaEl.textContent = "";
      document.getElementById("send").disabled = true;
      try {
        const res = await fetch("/generate", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
        const data = await res.json();
        if (!res.ok) throw new Error(JSON.stringify(data));
        answerEl.textContent = data.generated_text || "";
        metaEl.textContent = JSON.stringify(data, null, 2);
        statusEl.textContent = `完成：${data.route}`;
      } catch (err) {
        statusEl.textContent = "请求失败";
        answerEl.textContent = String(err);
      } finally {
        document.getElementById("send").disabled = false;
      }
    });

    refreshHealth();
  </script>
</body>
</html>
"""


def create_app(runner: GatewayRunner) -> FastAPI:
    app = FastAPI(title="LoRA-MoE Gateway API", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/health")
    def health():
        return runner.health()

    @app.get("/experts")
    def experts():
        return runner.health()["experts"]

    @app.post("/route/debug", response_model=RouteDebugResponse)
    def route_debug(request: GenerateRequest):
        return runner.route_debug(request)

    @app.get("/stage1/latest")
    def stage1_latest():
        return runner.stage1_latest()

    @app.post("/stage1/tick")
    def stage1_tick():
        return runner.stage1_tick()

    @app.post("/generate", response_model=GatewayResponse)
    def generate(request: GenerateRequest):
        return runner.generate(request)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a single frontend/API gateway for multiple LoRA-MoE experts.")
    parser.add_argument("--vision-url", default=DEFAULT_VISION_URL)
    parser.add_argument("--stage1-url", default=DEFAULT_STAGE1_URL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    runner = GatewayRunner(
        vision_url=args.vision_url,
        stage1_url=args.stage1_url,
        request_timeout_s=args.request_timeout_s,
    )
    app = create_app(runner)
    print(f"Serving LoRA-MoE Gateway on http://{args.host}:{args.port}", flush=True)
    print(f"vision_weather -> {args.vision_url}", flush=True)
    print(f"stage1_rain    -> {args.stage1_url}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
