import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image
import numpy as np

# ── Classes ──────────────────────────────────────────────────────────────────
CLASS_META = [
    ("Cloth",    "cloth",    "#3b82f6"),
    ("N95",      "n95",      "#8b5cf6"),
    ("No Mask",  "nomask",   "#f97316"),
    ("Surgical", "surgical", "#14b8a6"),
]

# ── Model ─────────────────────────────────────────────────────────────────────
model = torch.load('./models/modelAug6k.pth', weights_only=False)
model.eval()

image_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5211, 0.4858, 0.4651), (0.2889, 0.2824, 0.2880)),
])

# ── HTML helpers ──────────────────────────────────────────────────────────────
def build_result_html(predicted: int, confidence: float, probs: list) -> str:
    is_masked = predicted != 2
    card_class = "masked" if is_masked else "no-mask"
    label_class = "masked" if is_masked else "no-mask"
    label_text = (
        f"{CLASS_META[predicted][0]} Mask Detected" if is_masked else "No Mask Detected"
    )
    verdict_icon = "✓" if is_masked else "✕"

    bars_html = ""
    for i, (name, css_class, _) in enumerate(CLASS_META):
        pct = probs[i] * 100
        active_style = "font-weight:700;color:#f1f5f9;" if i == predicted else ""
        bars_html += f"""
        <div class="prob-row">
            <span class="prob-label" style="{active_style}">{name}</span>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill {css_class}" style="width:{pct:.2f}%"></div>
            </div>
            <span class="prob-pct">{pct:.1f}%</span>
        </div>"""

    return f"""
<div class="result-card {card_class}">
    <p class="result-label {label_class}">
        <span style="font-size:1.35rem;">{verdict_icon}</span>&nbsp;{label_text}
    </p>
    <p class="result-confidence">Confidence: <strong>{confidence:.1f}%</strong></p>
    <div class="prob-section">
        <h4>All class probabilities</h4>
        {bars_html}
    </div>
</div>"""

PLACEHOLDER_HTML = """
<div class="result-placeholder">
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0
                 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0
                 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
    <p>Upload a photo or use your camera,<br>then click <strong>Analyze Image</strong></p>
</div>"""

# ── Classify ───────────────────────────────────────────────────────────────────
def classify(image):
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    tensor = image_transforms(image).float().unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    # Model outputs 6 logits; only the first 4 correspond to named classes
    probs_all = F.softmax(output, dim=1)[0]
    probs4 = probs_all[:4]
    probs4 = (probs4 / probs4.sum()).tolist()

    predicted = int(torch.tensor(probs4).argmax().item())
    confidence = probs4[predicted] * 100

    return build_result_html(predicted, confidence, probs4)

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
#app-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 55%, #4a90d9 100%);
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(30,58,95,0.22);
}
#app-header h1 {
    color: #fff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    letter-spacing: -0.3px;
}
#app-header .subtitle {
    color: rgba(255,255,255,0.82);
    font-size: 0.93rem;
    line-height: 1.55;
    margin: 0 auto;
    max-width: 560px;
}
#app-header .badges {
    margin-top: 14px;
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}
#app-header .badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.9);
    font-weight: 500;
}
#submit-btn {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 13px 0 !important;
    cursor: pointer !important;
    transition: opacity 0.18s !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
}
#submit-btn:hover { opacity: 0.86 !important; }
#result-panel {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 24px;
    min-height: 320px;
}
.result-card { border-radius: 12px; padding: 20px 22px; margin: 0; }
.result-card.masked  { background: #052e16; border-left: 5px solid #22c55e; }
.result-card.no-mask { background: #1c0a00; border-left: 5px solid #f97316; }
.result-label { font-size: 1.45rem; font-weight: 800; margin: 0 0 6px 0; line-height: 1.3; letter-spacing: -0.2px; }
.result-label.masked  { color: #86efac; }
.result-label.no-mask { color: #fdba74; }
.result-confidence { font-size: 0.88rem; color: #94a3b8; margin: 0 0 18px 0; }
.result-confidence strong { color: #e2e8f0; }
.prob-section h4 {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #475569;
    margin: 0 0 10px 0;
}
.prob-row { display:flex; align-items:center; margin-bottom:8px; gap:8px; }
.prob-label { font-size:0.82rem; color:#94a3b8; width:68px; flex-shrink:0; }
.prob-bar-bg { flex:1; height:8px; background:#1e293b; border-radius:4px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:4px; }
.prob-bar-fill.cloth    { background:#3b82f6; }
.prob-bar-fill.n95      { background:#8b5cf6; }
.prob-bar-fill.nomask   { background:#f97316; }
.prob-bar-fill.surgical { background:#14b8a6; }
.prob-pct { font-size:0.78rem; color:#e2e8f0; width:38px; text-align:right; flex-shrink:0; font-weight:600; }
.result-placeholder {
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; min-height:240px;
    color:#334155; text-align:center; gap:14px;
}
.result-placeholder svg { width:52px; height:52px; opacity:0.4; }
.result-placeholder p   { font-size:0.88rem; margin:0; line-height:1.55; color:#475569; }
#app-footer {
    text-align:center;
    padding: 20px 0 6px;
    font-size: 0.78rem;
    color: #94a3b8;
    border-top: 1px solid #e2e8f0;
    margin-top: 28px;
    line-height: 1.7;
}
#app-footer a { color: #2d6a9f; text-decoration: none; }
#app-footer a:hover { text-decoration: underline; }
"""

# ── Layout ─────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="Face Mask Detector") as demo:

    gr.HTML("""
    <div id="app-header">
        <h1>😷 Face Mask Detector</h1>
        <p class="subtitle">
            Detect whether a person is wearing a <strong>Cloth</strong>,
            <strong>N95</strong>, or <strong>Surgical</strong> mask — or no mask at all.
            Works best with a clear frontal photo where the face is fully visible.
        </p>
        <div class="badges">
            <span class="badge">4 Classes</span>
            <span class="badge">79.6% Accuracy</span>
            <span class="badge">Trained on 6,071 images</span>
            <span class="badge">CNN · PyTorch</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=5):
            image_input = gr.Image(
                type="numpy",
                label="Upload or capture a photo",
                height=320,
            )
            submit_btn = gr.Button(
                "Analyze Image",
                elem_id="submit-btn",
                variant="primary",
            )

        with gr.Column(scale=5):
            gr.Markdown("### Result", elem_classes=["result-heading"])
            result_output = gr.HTML(
                value=PLACEHOLDER_HTML,
                elem_id="result-panel",
            )

    gr.HTML("""
    <div id="app-footer">
        Originally a 2022 Introduction to AI group assignment at Concordia University —
        revamped in 2026 by <a href="https://samchuang.ca" target="_blank">Samuel Chuang</a>
        &nbsp;·&nbsp;
        <a href="https://github.com/SamuelChuang98/AI-Face-Mask-Detector" target="_blank">GitHub</a>
    </div>
    """)

    submit_btn.click(
        fn=classify,
        inputs=[image_input],
        outputs=[result_output],
        api_name="predict",
    )

demo.launch()
