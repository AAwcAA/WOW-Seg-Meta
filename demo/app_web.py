"""
FastAPI-based Web UI for WOW-Seg.

Web flow:
1. Upload image via raw-binary POST to /api/upload_raw
2. Submit mode + prompt coordinates to /run

This avoids multipart parsing for the upload step.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
WOW_EVAL_DIR = REPO_ROOT / "wow_eval"

for _path in (REPO_ROOT, WOW_EVAL_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

from batch_pipeline import (
    DEFAULT_MAX_NEW_TOKENS,
    classify_masks_in_image,
    classify_single_mask,
    render_mask_overlay,
)
from sam_helpers import build_sam_bundle, predict_from_box, predict_from_points
from wow_inference import WowSegPredictor

APP_WEB_VERSION = "webui-v5"
WOW_SEG_CITATION = """@inproceedings{li2026wowseg,
      title={{WOW}-Seg: A Word-free Open World Segmentation Model},
      author={Danyang Li and Tianhao Wu and Bin Lin and Zhenyuan Chen and Yang Zhang and Yuxuan Li and Ming-Ming Cheng and Xiang Li},
      booktitle={The Fourteenth International Conference on Learning Representations},
      year={2026},
      url={https://openreview.net/forum?id=AyJPSnE1bq}
}"""


def ensure_rgb_uint8(img):
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))
    if not isinstance(img, np.ndarray):
        raise ValueError(f"Unsupported image type: {type(img)}")
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def array_to_data_url(array: np.ndarray, mode: Optional[str] = None):
    if mode is None:
        mode = "RGBA" if array.ndim == 3 and array.shape[2] == 4 else "RGB"
    image = Image.fromarray(array, mode=mode)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_summary_text(results):
    if not results:
        return "No results."
    lines = []
    for result in results:
        bbox = ",".join(str(v) for v in result["bbox_xyxy"])
        lines.append(f"{result['index']:02d}. {result['category']} | area={result['area']} | bbox={bbox}")
    return "\n".join(lines)


def build_status_text(results, mode):
    if not results:
        return f"{mode}: no masks found."
    return f"{mode}: processed {len(results)} mask(s)."


def render_cards(results):
    blocks = []
    for result in results:
        crop = array_to_data_url(result["crop_rgba"], mode="RGBA")
        category = html.escape(result["category"])
        bbox = ",".join(str(v) for v in result["bbox_xyxy"])
        blocks.append(
            f"""
            <div class="res-item">
              <img src="{crop}" alt="{category}">
              <div class="res-info">
                <div class="res-tag">{result['index']}. {category}</div>
                <div style="color:#64748b">area={result['area']}</div>
                <div style="color:#64748b">bbox={html.escape(bbox)}</div>
              </div>
            </div>
            """
        )
    return "\n".join(blocks)


def html_page_response(content: str):
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def render_page(
    *,
    session_id: str = "",
    image_data: str = "",
    overlay_data: str = "",
    status: str = "Load one image to start.",
    summary: str = "No results yet.",
    cards_html: str = "",
    mode: str = "auto",
    point: Optional[dict] = None,
    box: Optional[dict] = None,
):
    status_html = html.escape(status)
    summary_html = html.escape(summary)
    citation_html = html.escape(WOW_SEG_CITATION)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>WOW-Seg | Multi-Modal Segmentation UI</title>
  <style>
    :root {{ --primary: #4f46e5; --sidebar-w: 320px; --border: #e2e8f0; }}
    body {{ margin: 0; display: flex; height: 100vh; font-family: 'Segoe UI', system-ui, sans-serif; background: #f8fafc; overflow: hidden; }}
    .sidebar {{ width: var(--sidebar-w); background: white; border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 24px; gap: 20px; box-shadow: 2px 0 5px rgba(0,0,0,0.02); }}
    .logo {{ font-size: 24px; font-weight: 800; color: var(--primary); margin-bottom: 10px; }}
    .section {{ display: flex; flex-direction: column; gap: 8px; }}
    .label {{ font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
    .main {{ flex: 1; padding: 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 24px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .card {{ background: white; border-radius: 12px; border: 1px solid var(--border); padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    canvas, .res-img {{ width: 100%; border-radius: 8px; background: #000; aspect-ratio: 4/3; object-fit: contain; cursor: crosshair; display: block; }}
    input[type="file"], select, button {{ width: 100%; padding: 10px; border-radius: 8px; border: 1px solid var(--border); font-size: 14px; }}
    button {{ background: var(--primary); color: white; border: none; font-weight: 600; cursor: pointer; transition: 0.2s; }}
    button:hover {{ background: #4338ca; }}
    button:disabled {{ background: #cbd5e1; cursor: not-allowed; }}
    .secondary-btn {{ background: white; color: #333; border: 1px solid var(--border); margin-top: 8px; }}
    .status-bar {{ padding: 12px; background: #f1f5f9; border-radius: 8px; border-left: 4px solid var(--primary); font-family: monospace; font-size: 13px; white-space: pre-wrap; }}
    .res-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 14px;
      align-items: start;
    }}
    .res-item {{
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      background: #fff;
      transition: 0.2s;
      display: flex;
      flex-direction: column;
      min-height: 240px;
    }}
    .res-item img {{
      width: 100%;
      height: 160px;
      object-fit: contain;
      background: #f1f5f9;
      display: block;
      border-bottom: 1px solid var(--border);
    }}
    .res-info {{
      padding: 10px;
      font-size: 12px;
      line-height: 1.45;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .res-tag {{ font-weight: 700; color: var(--primary); }}
    .citation-box {{
      padding: 12px;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 8px;
      font-family: Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      color: #334155;
      white-space: pre-wrap;
      word-break: break-word;
      min-height: 240px;
      max-height: 380px;
      overflow-y: auto;
    }}
    .tiny-note {{
      font-size: 11px;
      color: #94a3b8;
      line-height: 1.45;
    }}
    @media (max-width: 1100px) {{
      body {{ display: block; height: auto; overflow: auto; }}
      .sidebar {{ width: auto; border-right: none; border-bottom: 1px solid var(--border); }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="sidebar">
    <div class="logo">WOW-Seg demo</div>
    <div class="section">
      <span class="label">1. Source Image</span>
      <input id="imageFile" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" />
      <button id="uploadButton" type="button">Upload & Load</button>
    </div>
    <form id="runForm" method="post" action="/run">
      <input id="sessionIdInput" name="session_id" type="hidden" value="{html.escape(session_id)}" />
      <input id="pointXInput" name="point_x" type="hidden" value="{'' if point is None else point.get('x', '')}" />
      <input id="pointYInput" name="point_y" type="hidden" value="{'' if point is None else point.get('y', '')}" />
      <input id="boxX0Input" name="box_x0" type="hidden" value="{'' if box is None else box.get('x0', '')}" />
      <input id="boxY0Input" name="box_y0" type="hidden" value="{'' if box is None else box.get('y0', '')}" />
      <input id="boxX1Input" name="box_x1" type="hidden" value="{'' if box is None else box.get('x1', '')}" />
      <input id="boxY1Input" name="box_y1" type="hidden" value="{'' if box is None else box.get('y1', '')}" />
      <div class="section">
        <span class="label">2. Inference Mode</span>
        <select id="modeSelect" name="mode">
          <option value="auto" {"selected" if mode=="auto" else ""}>Automatic (Full Image)</option>
          <option value="point" {"selected" if mode=="point" else ""}>Point Prompt</option>
          <option value="box" {"selected" if mode=="box" else ""}>Box Prompt</option>
        </select>
      </div>
      <div class="section" style="margin-top:10px;">
        <button id="runButton" type="submit" {"disabled" if not session_id else ""}>Run Model</button>
        <button id="clearButton" class="secondary-btn" type="button">Clear Selection</button>
      </div>
    </form>
    <div class="section">
      <span class="label">Guide</span>
      <div class="status-bar" id="modeHint"></div>
    </div>
    <div class="section">
      <span class="label">Reference</span>
      <button id="copyCitationBtn" type="button">Copy Citation</button>
      <div id="citationBox" class="citation-box">{citation_html}</div>
      <div class="tiny-note">Click the button above to copy the BibTeX citation.</div>
    </div>
    <div style="margin-top:auto; font-size:11px; color:#94a3b8;">
      Version: {APP_WEB_VERSION}<br>
      Session: {html.escape(session_id) if session_id else "Not Loaded"}
    </div>
  </div>

  <div class="main">
    <div class="status-bar" id="statusBox">{status_html}</div>
    <div class="grid">
      <div class="card">
        <span class="label" style="display:block; margin-bottom:10px;">Input Canvas</span>
        <canvas id="inputCanvas" width="1024" height="768"></canvas>
      </div>
      <div class="card">
        <span class="label" style="display:block; margin-bottom:10px;">Result Preview</span>
        <img class="res-img" id="overlayImage" src="{overlay_data}" alt="Segmentation result">
      </div>
    </div>
    <div class="card">
      <span class="label" style="display:block; margin-bottom:10px;">Mask Summary</span>
      <div id="summaryBox" class="status-bar">{summary_html}</div>
    </div>
    <div class="card">
      <span class="label" style="display:block; margin-bottom:10px;">Detected Objects</span>
      <div id="resultsGrid" class="res-grid">{cards_html if cards_html else '<div style="color:#94a3b8; padding:20px;">No objects detected.</div>'}</div>
    </div>
  </div>

  <script>
    const hints = {{
      auto: "Auto: click Run to segment the whole image and classify all masks.",
      point: "Point Prompt: click once on the image, then click Run.",
      box: "BBox Prompt: drag one box on the image, then click Run."
    }};

    let sessionId = {json.dumps(session_id)};
    let imageData = {json.dumps(image_data)};
    let point = {json.dumps(point)};
    let box = {json.dumps(box)};

    const imageInput = document.getElementById("imageFile");
    const uploadButton = document.getElementById("uploadButton");
    const modeSelect = document.getElementById("modeSelect");
    const modeHint = document.getElementById("modeHint");
    const runForm = document.getElementById("runForm");
    const runButton = document.getElementById("runButton");
    const clearButton = document.getElementById("clearButton");
    const copyCitationBtn = document.getElementById("copyCitationBtn");
    const sessionIdInput = document.getElementById("sessionIdInput");
    const pointXInput = document.getElementById("pointXInput");
    const pointYInput = document.getElementById("pointYInput");
    const boxX0Input = document.getElementById("boxX0Input");
    const boxY0Input = document.getElementById("boxY0Input");
    const boxX1Input = document.getElementById("boxX1Input");
    const boxY1Input = document.getElementById("boxY1Input");
    const canvas = document.getElementById("inputCanvas");
    const ctx = canvas.getContext("2d");
    const overlayImage = document.getElementById("overlayImage");
    const statusBox = document.getElementById("statusBox");
    const summaryBox = document.getElementById("summaryBox");

    let imageElement = new Image();
    let drawWidth = 0;
    let drawHeight = 0;
    let drawOffsetX = 0;
    let drawOffsetY = 0;
    let dragStart = null;
    let dragCurrent = null;

    function updateModeHint() {{
      modeHint.textContent = hints[modeSelect.value] || "";
    }}

    function syncPromptInputs() {{
      sessionIdInput.value = sessionId || "";
      pointXInput.value = point ? point.x : "";
      pointYInput.value = point ? point.y : "";
      boxX0Input.value = box ? box.x0 : "";
      boxY0Input.value = box ? box.y0 : "";
      boxX1Input.value = box ? box.x1 : "";
      boxY1Input.value = box ? box.y1 : "";
      runButton.disabled = !sessionId;
    }}

    function clearPrompt() {{
      point = null;
      box = null;
      dragStart = null;
      dragCurrent = null;
      syncPromptInputs();
      redrawCanvas();
    }}

    function normalizeBox(b) {{
      return {{
        x0: Math.min(b.x0, b.x1),
        y0: Math.min(b.y0, b.y1),
        x1: Math.max(b.x0, b.x1),
        y1: Math.max(b.y0, b.y1)
      }};
    }}

    function imageToCanvas(x, y) {{
      const cx = drawOffsetX + x * drawWidth / imageElement.naturalWidth;
      const cy = drawOffsetY + y * drawHeight / imageElement.naturalHeight;
      return [cx, cy];
    }}

    function canvasToImage(clientX, clientY) {{
      if (!imageElement.src) return null;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = (clientX - rect.left) * scaleX;
      const y = (clientY - rect.top) * scaleY;
      if (x < drawOffsetX || y < drawOffsetY || x > drawOffsetX + drawWidth || y > drawOffsetY + drawHeight) {{
        return null;
      }}
      const imgX = Math.round((x - drawOffsetX) * imageElement.naturalWidth / drawWidth);
      const imgY = Math.round((y - drawOffsetY) * imageElement.naturalHeight / drawHeight);
      return {{
        x: Math.max(0, Math.min(imageElement.naturalWidth - 1, imgX)),
        y: Math.max(0, Math.min(imageElement.naturalHeight - 1, imgY))
      }};
    }}

    function redrawCanvas() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#0a101c";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (!imageElement.src) return;

      const scale = Math.min(canvas.width / imageElement.naturalWidth, canvas.height / imageElement.naturalHeight);
      drawWidth = Math.round(imageElement.naturalWidth * scale);
      drawHeight = Math.round(imageElement.naturalHeight * scale);
      drawOffsetX = Math.round((canvas.width - drawWidth) / 2);
      drawOffsetY = Math.round((canvas.height - drawHeight) / 2);
      ctx.drawImage(imageElement, drawOffsetX, drawOffsetY, drawWidth, drawHeight);

      if (point) {{
        const [cx, cy] = imageToCanvas(point.x, point.y);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(cx, cy, 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = "#ef4444";
        ctx.beginPath();
        ctx.arc(cx, cy, 6, 0, Math.PI * 2);
        ctx.fill();
      }}

      const liveBox = box || (dragStart && dragCurrent ? normalizeBox({{
        x0: dragStart.x, y0: dragStart.y, x1: dragCurrent.x, y1: dragCurrent.y
      }}) : null);

      if (liveBox) {{
        const c0 = imageToCanvas(liveBox.x0, liveBox.y0);
        const c1 = imageToCanvas(liveBox.x1, liveBox.y1);
        ctx.lineWidth = 3;
        ctx.strokeStyle = "#4f46e5";
        ctx.strokeRect(c0[0], c0[1], c1[0] - c0[0], c1[1] - c0[1]);
        ctx.fillStyle = "rgba(79, 70, 229, 0.15)";
        ctx.fillRect(c0[0], c0[1], c1[0] - c0[0], c1[1] - c0[1]);
      }}
    }}

    uploadButton.addEventListener("click", async () => {{
      const file = imageInput.files[0];
      if (!file) {{
        alert("Please choose an image first.");
        return;
      }}
      uploadButton.disabled = true;
      uploadButton.textContent = "Uploading...";
      try {{
        const response = await fetch(`/api/upload_raw?filename=${{encodeURIComponent(file.name)}}`, {{
          method: "POST",
          headers: {{ "Content-Type": "application/octet-stream" }},
          body: file
        }});
        const data = await response.json();
        if (!response.ok) {{
          throw new Error(data.detail || "Upload failed");
        }}
        sessionId = data.session_id;
        imageData = data.image_data;
        imageElement = new Image();
        imageElement.onload = redrawCanvas;
        imageElement.src = imageData;
        overlayImage.src = "";
        point = null;
        box = null;
        dragStart = null;
        dragCurrent = null;
        statusBox.textContent = data.status;
        summaryBox.textContent = "No results yet.";
        document.getElementById("resultsGrid").innerHTML = '<div style="color:#94a3b8; padding:20px;">No objects detected.</div>';
        syncPromptInputs();
      }} catch (error) {{
        statusBox.textContent = `Error: ${{error.message}}`;
      }} finally {{
        uploadButton.disabled = false;
        uploadButton.textContent = "Upload";
      }}
    }});

    modeSelect.addEventListener("change", () => {{
      updateModeHint();
      clearPrompt();
    }});

    clearButton.addEventListener("click", () => {{
      clearPrompt();
      statusBox.textContent = "Prompt selection cleared.";
    }});

    copyCitationBtn.addEventListener("click", async () => {{
      const citationText = {json.dumps(WOW_SEG_CITATION)};
      try {{
        await navigator.clipboard.writeText(citationText);
        statusBox.textContent = "Citation copied to clipboard.";
      }} catch (error) {{
        statusBox.textContent = "Failed to copy citation. Please copy it manually.";
      }}
    }});

    canvas.addEventListener("click", (event) => {{
      if (modeSelect.value !== "point" || !sessionId) return;
      const coords = canvasToImage(event.clientX, event.clientY);
      if (!coords) return;
      point = coords;
      box = null;
      syncPromptInputs();
      redrawCanvas();
    }});

    canvas.addEventListener("mousedown", (event) => {{
      if (modeSelect.value !== "box" || !sessionId) return;
      const coords = canvasToImage(event.clientX, event.clientY);
      if (!coords) return;
      dragStart = coords;
      dragCurrent = coords;
      point = null;
      box = null;
      syncPromptInputs();
      redrawCanvas();
    }});

    canvas.addEventListener("mousemove", (event) => {{
      if (modeSelect.value !== "box" || !dragStart) return;
      const coords = canvasToImage(event.clientX, event.clientY);
      if (!coords) return;
      dragCurrent = coords;
      redrawCanvas();
    }});

    window.addEventListener("mouseup", (event) => {{
      if (modeSelect.value !== "box" || !dragStart) return;
      const coords = canvasToImage(event.clientX, event.clientY) || dragCurrent;
      if (!coords) {{
        dragStart = null;
        dragCurrent = null;
        redrawCanvas();
        return;
      }}
      box = normalizeBox({{ x0: dragStart.x, y0: dragStart.y, x1: coords.x, y1: coords.y }});
      dragStart = null;
      dragCurrent = null;
      syncPromptInputs();
      redrawCanvas();
    }});

    runForm.addEventListener("submit", (event) => {{
      if (!sessionId) {{
        event.preventDefault();
        alert("Please upload an image first.");
        return;
      }}
      if (modeSelect.value === "point" && !point) {{
        event.preventDefault();
        alert("Please click one point first.");
        return;
      }}
      if (modeSelect.value === "box" && !box) {{
        event.preventDefault();
        alert("Please drag one box first.");
        return;
      }}
      runButton.disabled = true;
      runButton.textContent = "Running...";
    }});

    updateModeHint();
    syncPromptInputs();
    if (imageData) {{
      imageElement = new Image();
      imageElement.onload = redrawCanvas;
      imageElement.src = imageData;
    }} else {{
      redrawCanvas();
    }}
  </script>
</body>
</html>
"""


def create_app(args):
    app = FastAPI(title="WOW-Seg Web UI")

    device = args.device if torch.cuda.is_available() else "cpu"
    print("Loading SAM...")
    sam_predictor, mask_generator = build_sam_bundle(
        checkpoint_path=args.sam_ckpt,
        model_type=args.sam_model_type,
        device=device,
    )
    print("Loading WOW-Seg...")
    wow = WowSegPredictor(args.wow_model, device=device)

    sessions: Dict[str, Dict[str, object]] = {}
    inference_lock = threading.Lock()

    def run_auto(image_rgb: np.ndarray):
        return classify_masks_in_image(
            image_rgb=image_rgb,
            mask_generator=mask_generator,
            wow=wow,
            min_mask_area=args.min_mask_area,
            max_masks=args.max_masks,
            max_new_tokens=args.max_new_tokens,
        )

    def run_point(image_rgb: np.ndarray, point_x: int, point_y: int):
        mask = predict_from_points(sam_predictor, image_rgb, [((point_x, point_y), 1)])
        result = classify_single_mask(
            image_rgb=image_rgb,
            mask=mask,
            wow=wow,
            index=1,
            max_new_tokens=args.max_new_tokens,
        )
        return [result] if result is not None else []

    def run_box(image_rgb: np.ndarray, box_xyxy):
        mask = predict_from_box(sam_predictor, image_rgb, np.array(box_xyxy, dtype=int))
        result = classify_single_mask(
            image_rgb=image_rgb,
            mask=mask,
            wow=wow,
            index=1,
            max_new_tokens=args.max_new_tokens,
        )
        return [result] if result is not None else []

    @app.get("/", response_class=HTMLResponse)
    def index():
        return html_page_response(render_page())

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.post("/api/upload_raw")
    async def upload_raw(request: Request, filename: str = "uploaded_image"):
        try:
            raw = await request.body()
            print(f"[web] POST /api/upload_raw filename={filename} bytes={len(raw)}")
            image_rgb = ensure_rgb_uint8(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:
            return JSONResponse({"detail": f"Invalid image: {exc}"}, status_code=400)

        session_id = str(uuid.uuid4())
        image_data = array_to_data_url(image_rgb)
        sessions[session_id] = {
            "image_rgb": image_rgb,
            "image_data": image_data,
            "filename": filename,
        }
        return JSONResponse(
            {
                "session_id": session_id,
                "image_data": image_data,
                "status": f"Loaded image: {filename}. Choose a mode and click Run.",
            }
        )

    @app.post("/run", response_class=HTMLResponse)
    async def run(request: Request):
        form = await request.form()
        print(f"[web] POST /run fields={list(form.keys())}")

        def parse_optional_int(value):
            if value in (None, ""):
                return None
            return int(value)

        session_id = str(form.get("session_id", "") or "")
        mode = str(form.get("mode", "auto") or "auto")
        point_x = parse_optional_int(form.get("point_x"))
        point_y = parse_optional_int(form.get("point_y"))
        box_x0 = parse_optional_int(form.get("box_x0"))
        box_y0 = parse_optional_int(form.get("box_y0"))
        box_x1 = parse_optional_int(form.get("box_x1"))
        box_y1 = parse_optional_int(form.get("box_y1"))

        session = sessions.get(session_id)
        if session is None:
            return html_page_response(
                render_page(status="Error: session not found. Please upload the image again.", summary="No results yet.")
            )

        image_rgb = session["image_rgb"]
        image_data = session["image_data"]
        mode = (mode or "auto").strip().lower()
        point = None
        box = None

        try:
            with inference_lock:
                if mode == "auto":
                    results = run_auto(image_rgb)
                elif mode == "point":
                    if point_x is None or point_y is None:
                        raise ValueError("Point coordinates are required for point mode.")
                    point = {"x": int(point_x), "y": int(point_y)}
                    results = run_point(image_rgb, point["x"], point["y"])
                elif mode == "box":
                    if None in (box_x0, box_y0, box_x1, box_y1):
                        raise ValueError("Box coordinates are required for box mode.")
                    box = {"x0": int(box_x0), "y0": int(box_y0), "x1": int(box_x1), "y1": int(box_y1)}
                    results = run_box(image_rgb, (box["x0"], box["y0"], box["x1"], box["y1"]))
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
        except Exception as exc:
            return html_page_response(
                render_page(
                    session_id=session_id,
                    image_data=image_data,
                    status=f"Error: {exc}",
                    summary="No results.",
                    mode=mode,
                    point=point,
                    box=box,
                )
            )

        overlay = render_mask_overlay(image_rgb, results) if results else image_rgb
        return html_page_response(
            render_page(
                session_id=session_id,
                image_data=image_data,
                overlay_data=array_to_data_url(overlay),
                status=build_status_text(results, mode.capitalize()),
                summary=build_summary_text(results),
                cards_html=render_cards(results),
                mode=mode,
                point=point,
                box=box,
            )
        )

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="WOW-Seg FastAPI Web UI")
    parser.add_argument("--wow_model", type=str, required=True, help="Path to the WOW-Seg checkpoint.")
    parser.add_argument(
        "--sam_ckpt",
        type=str,
        default=str(DEMO_DIR / "weights" / "sam" / "sam_vit_h_4b8939.pth"),
        help="Path to the SAM checkpoint.",
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model type matching the checkpoint.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min_mask_area", type=int, default=128)
    parser.add_argument("--max_masks", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, reload=False, workers=1)
