import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Brain Tumor AI Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:    #060d14;
    --bg-secondary:  #0c1c2c;
    --bg-card:       rgba(10, 30, 50, 0.85);
    --accent-blue:   #00aaff;
    --accent-cyan:   #00e5ff;
    --accent-teal:   #00ffd5;
    --text-primary:  #e8f4ff;
    --text-muted:    #7aaccc;
    --border-glow:   rgba(0, 170, 255, 0.35);
    --shadow-glow:   0 0 30px rgba(0, 200, 255, 0.18);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(ellipse at 20% 50%, #0a1f35 0%, #060d14 60%),
                radial-gradient(ellipse at 80% 20%, #091826 0%, transparent 50%);
    background-color: var(--bg-primary);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071523 0%, #0a1e30 100%);
    border-right: 1px solid var(--border-glow);
}

section[data-testid="stSidebar"] .stRadio label {
    color: var(--text-primary) !important;
    font-family: 'Exo 2', sans-serif;
    font-size: 14px;
    padding: 6px 0;
}

/* ── Headers ── */
h1 { font-family: 'Orbitron', monospace !important; font-weight: 900 !important; }
h2, h3, h4 { font-family: 'Orbitron', monospace !important; font-weight: 700 !important; }

/* ── Hero Title ── */
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(1.8rem, 4vw, 3rem);
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #00aaff, #00e5ff, #00ffd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
    margin-bottom: 0.3rem;
    text-shadow: none;
}

.hero-subtitle {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.95rem;
    font-family: 'Exo 2', sans-serif;
    font-weight: 300;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── Divider ── */
.glow-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-cyan), transparent);
    margin: 1.5rem 0;
    opacity: 0.7;
}

/* ── Cards ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-glow);
    border-radius: 16px;
    padding: 24px;
    box-shadow: var(--shadow-glow), inset 0 1px 0 rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}

/* ── Metric Boxes ── */
.metric-box {
    background: linear-gradient(135deg, rgba(0,170,255,0.12), rgba(0,229,255,0.06));
    border: 1px solid rgba(0,200,255,0.3);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 12px;
}

.metric-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: 'Exo 2', sans-serif;
    margin-bottom: 6px;
}

.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent-cyan);
}

/* ── Tumor Class Badge ── */
.tumor-badge {
    display: inline-block;
    padding: 8px 22px;
    border-radius: 50px;
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1px;
    margin: 10px 0;
    text-align: center;
    width: 100%;
}

.badge-danger {
    background: linear-gradient(135deg, rgba(255,60,60,0.2), rgba(220,30,30,0.1));
    border: 1px solid rgba(255,80,80,0.5);
    color: #ff6b6b;
}

.badge-warning {
    background: linear-gradient(135deg, rgba(255,180,0,0.2), rgba(200,140,0,0.1));
    border: 1px solid rgba(255,200,0,0.5);
    color: #ffd060;
}

.badge-success {
    background: linear-gradient(135deg, rgba(0,220,120,0.2), rgba(0,180,90,0.1));
    border: 1px solid rgba(0,220,130,0.5);
    color: #00e58a;
}

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    background: rgba(0, 170, 255, 0.04) !important;
    border: 2px dashed rgba(0, 170, 255, 0.3) !important;
    border-radius: 14px !important;
    padding: 10px !important;
    transition: border-color 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 229, 255, 0.6) !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #0077cc, #00aaff, #00e5ff);
    color: #ffffff;
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border: none;
    border-radius: 50px;
    padding: 14px 36px;
    width: 100%;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(0, 170, 255, 0.4);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    box-shadow: 0 6px 30px rgba(0, 229, 255, 0.6);
    transform: translateY(-2px);
}

/* ── Progress Bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #0077cc, #00e5ff, #00ffd5) !important;
    border-radius: 50px !important;
}

.stProgress {
    background: rgba(0, 170, 255, 0.1) !important;
    border-radius: 50px !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--accent-cyan) !important;
}

/* ── Sidebar Model Selector Label ── */
.sidebar-section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 10px;
}

/* ── Info Panel Tags ── */
.info-tag {
    display: inline-block;
    background: rgba(0, 170, 255, 0.1);
    border: 1px solid rgba(0, 170, 255, 0.25);
    color: var(--accent-blue);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-family: 'Exo 2', sans-serif;
    margin: 3px 2px;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.78rem;
    font-family: 'Exo 2', sans-serif;
    letter-spacing: 2px;
    padding: 20px 0 10px;
}

/* ── Warning / Error overrides ── */
.stAlert {
    border-radius: 12px !important;
}

/* ── Image caption ── */
[data-testid="caption"] {
    color: var(--text-muted) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.8rem !important;
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)


# ===================== HELPERS =====================

def ensure_rgb(img_array: np.ndarray) -> np.ndarray:
    """Convert any image to a 3-channel uint8 RGB array."""
    if img_array is None:
        raise ValueError("Image array is None.")
    # Handle grayscale
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    # Handle RGBA / 4-channel
    elif img_array.ndim == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    # Ensure uint8
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
    return img_array


def get_badge_class(label: str) -> str:
    label_lower = label.lower()
    if "no tumor" in label_lower:
        return "badge-success"
    elif "meningioma" in label_lower:
        return "badge-warning"
    else:
        return "badge-danger"


# ===================== LOAD MODELS =====================
@st.cache_resource(show_spinner="Loading AI models…")
def load_models():
    classification_model = tf.keras.models.load_model("EfficientNet_best_Model.keras")
    segmentation_model   = tf.keras.models.load_model("unet_final1.keras", compile=False)
    detection_model      = YOLO("Yolov8_best_model.pt")
    return classification_model, segmentation_model, detection_model

classification_model, segmentation_model, detection_model = load_models()

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 10px 0 20px;'>
            <span style='font-family:"Orbitron",monospace; font-size:1.5rem;
                         font-weight:900; color:#00e5ff; letter-spacing:2px;'>
                🧠 AI SUITE
            </span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section-label">Select Analysis Mode</p>', unsafe_allow_html=True)

    model_choice = st.radio(
        label="",
        options=["Classification", "Segmentation", "Detection"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Model info panel
    model_info = {
        "Classification": {
            "arch": "EfficientNet-B4",
            "input": "380 × 380 px",
            "output": "4 Classes",
            "tags": ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
        },
        "Segmentation": {
            "arch": "U-Net",
            "input": "256 × 256 px",
            "output": "Binary Mask",
            "tags": ["Pixel-wise", "Tumor Region", "Overlay"]
        },
        "Detection": {
            "arch": "YOLOv8",
            "input": "640 × 640 px",
            "output": "Bounding Box",
            "tags": ["Real-time", "Localization", "Confidence"]
        }
    }

    info = model_info[model_choice]
    st.markdown(f"""
        <div class="glass-card" style="padding:16px;">
            <p class="sidebar-section-label" style="margin-bottom:8px;">Model Info</p>
            <p style="font-family:'Exo 2',sans-serif; font-size:0.82rem; color:#7aaccc;
                       margin:4px 0;">Architecture</p>
            <p style="font-family:'Orbitron',monospace; font-size:0.85rem;
                       color:#00e5ff; margin:0 0 10px;">{info['arch']}</p>
            <p style="font-family:'Exo 2',sans-serif; font-size:0.82rem; color:#7aaccc;
                       margin:4px 0;">Input Size</p>
            <p style="font-family:'Exo 2',sans-serif; font-size:0.88rem;
                       color:#e8f4ff; margin:0 0 10px;">{info['input']}</p>
            <p style="font-family:'Exo 2',sans-serif; font-size:0.82rem; color:#7aaccc;
                       margin:4px 0;">Output</p>
            <p style="font-family:'Exo 2',sans-serif; font-size:0.88rem;
                       color:#e8f4ff; margin:0 0 12px;">{info['output']}</p>
            {"".join(f'<span class="info-tag">{t}</span>' for t in info['tags'])}
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.success("✅  System Ready", icon=None)


# ===================== HEADER =====================
st.markdown('<h1 class="hero-title">BRAIN TUMOR AI SYSTEM</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Classification &nbsp;•&nbsp; Segmentation &nbsp;•&nbsp; Detection</p>',
    unsafe_allow_html=True
)
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)


# ===================== UPLOAD =====================
uploaded_file = st.file_uploader(
    "Upload MRI Image  (JPG / PNG / JPEG)",
    type=["jpg", "png", "jpeg"],
    help="Upload a brain MRI scan to analyse."
)


# ===================== MAIN LOGIC =====================
if uploaded_file is not None:
    image     = Image.open(uploaded_file)
    img_array = np.array(image)

    col_img, col_result = st.columns([1, 1], gap="large")

    # ── Left: original image ──────────────────────────────────
    with col_img:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-family:\'Orbitron\',monospace; font-size:0.78rem; '
            'letter-spacing:3px; color:#7aaccc; text-transform:uppercase; '
            'margin-bottom:12px;">Input MRI Scan</p>',
            unsafe_allow_html=True
        )
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)

        # File meta
        h, w = img_array.shape[:2]
        channels = 1 if img_array.ndim == 2 else img_array.shape[2]
        st.markdown(f"""
            <div style="display:flex; gap:10px; margin-top:14px; flex-wrap:wrap;">
                <div class="metric-box" style="flex:1; min-width:80px;">
                    <div class="metric-label">Width</div>
                    <div class="metric-value" style="font-size:1rem;">{w}px</div>
                </div>
                <div class="metric-box" style="flex:1; min-width:80px;">
                    <div class="metric-label">Height</div>
                    <div class="metric-value" style="font-size:1rem;">{h}px</div>
                </div>
                <div class="metric-box" style="flex:1; min-width:80px;">
                    <div class="metric-label">Channels</div>
                    <div class="metric-value" style="font-size:1rem;">{channels}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Right: results ────────────────────────────────────────
    with col_result:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-family:\'Orbitron\',monospace; font-size:0.78rem; '
            'letter-spacing:3px; color:#7aaccc; text-transform:uppercase; '
            'margin-bottom:12px;">Analysis Results</p>',
            unsafe_allow_html=True
        )

        if st.button("🚀  RUN AI ANALYSIS"):
            with st.spinner("Analysing MRI scan…"):

                try:
                    rgb_array = ensure_rgb(img_array)

                    # ── CLASSIFICATION ──────────────────────────
                    if model_choice == "Classification":
                        img_resized = cv2.resize(rgb_array, (380, 380))
                        img_norm    = img_resized.astype(np.float32) / 255.0
                        img_batch   = np.expand_dims(img_norm, axis=0)

                        pred        = classification_model.predict(img_batch, verbose=0)
                        class_idx   = int(np.argmax(pred))
                        confidence  = float(np.max(pred))
                        label       = CLASSES[class_idx]
                        badge_class = get_badge_class(label)

                        st.markdown(f"""
                            <div style="text-align:center; margin: 10px 0 20px;">
                                <p style="font-family:'Exo 2',sans-serif; font-size:0.78rem;
                                           color:#7aaccc; letter-spacing:2px;
                                           text-transform:uppercase; margin-bottom:8px;">
                                    Diagnosis
                                </p>
                                <div class="tumor-badge {badge_class}">{label}</div>
                            </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                            <p style="font-family:'Exo 2',sans-serif; font-size:0.8rem;
                                       color:#7aaccc; letter-spacing:1px;
                                       margin-bottom:6px;">
                                Confidence &nbsp;—&nbsp;
                                <span style="color:#00e5ff; font-weight:600;">
                                    {confidence * 100:.1f}%
                                </span>
                            </p>
                        """, unsafe_allow_html=True)
                        st.progress(confidence)

                        # Per-class breakdown
                        st.markdown("""
                            <p style="font-family:'Exo 2',sans-serif; font-size:0.78rem;
                                       color:#7aaccc; letter-spacing:2px;
                                       text-transform:uppercase; margin:18px 0 8px;">
                                Class Probabilities
                            </p>
                        """, unsafe_allow_html=True)

                        for i, cls in enumerate(CLASSES):
                            prob = float(pred[0][i])
                            col_a, col_b = st.columns([2, 3])
                            with col_a:
                                st.markdown(
                                    f'<p style="font-family:\'Exo 2\',sans-serif;'
                                    f'font-size:0.82rem; color:#e8f4ff; margin:6px 0;">'
                                    f'{cls}</p>',
                                    unsafe_allow_html=True
                                )
                            with col_b:
                                st.progress(prob)

                    # ── SEGMENTATION ────────────────────────────
                    elif model_choice == "Segmentation":
                        img_resized = cv2.resize(rgb_array, (256, 256))
                        img_norm    = img_resized.astype(np.float32) / 255.0
                        img_batch   = np.expand_dims(img_norm, axis=0)

                        mask_pred   = segmentation_model.predict(img_batch, verbose=0)[0]
                        mask_bin    = (mask_pred > 0.5).astype(np.uint8)

                        # Handle mask shape: (H, W, 1) or (H, W)
                        if mask_bin.ndim == 3:
                            mask_bin = mask_bin[:, :, 0]

                        mask_resized = cv2.resize(
                            mask_bin,
                            (rgb_array.shape[1], rgb_array.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )

                        # Overlay: tumor pixels → vivid red
                        overlay = rgb_array.copy()
                        overlay[mask_resized == 1] = [220, 30, 30]
                        blended = cv2.addWeighted(rgb_array, 0.55, overlay, 0.45, 0)

                        # Contour highlight
                        contours, _ = cv2.findContours(
                            mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(blended, contours, -1, (0, 220, 255), 2)

                        st.image(
                            blended,
                            caption="Tumour Segmentation — Red overlay + cyan contour",
                            use_container_width=True
                        )

                        tumor_pixels = int(np.sum(mask_resized))
                        total_pixels = mask_resized.size
                        coverage     = tumor_pixels / total_pixels * 100

                        st.markdown(f"""
                            <div style="display:flex; gap:10px; margin-top:14px;">
                                <div class="metric-box" style="flex:1;">
                                    <div class="metric-label">Tumour Pixels</div>
                                    <div class="metric-value">{tumor_pixels:,}</div>
                                </div>
                                <div class="metric-box" style="flex:1;">
                                    <div class="metric-label">Coverage</div>
                                    <div class="metric-value">{coverage:.2f}%</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    # ── DETECTION ───────────────────────────────
                    elif model_choice == "Detection":
                        results  = detection_model(rgb_array)
                        res_img  = results[0].plot()

                        # Convert BGR (OpenCV) → RGB for display
                        res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

                        st.image(
                            res_img_rgb,
                            caption="YOLOv8 Detection — Bounding boxes & confidence",
                            use_container_width=True
                        )

                        boxes      = results[0].boxes
                        n_detected = len(boxes) if boxes is not None else 0

                        if n_detected > 0:
                            confs = boxes.conf.cpu().numpy()
                            st.markdown(f"""
                                <div style="display:flex; gap:10px; margin-top:14px;">
                                    <div class="metric-box" style="flex:1;">
                                        <div class="metric-label">Objects Found</div>
                                        <div class="metric-value">{n_detected}</div>
                                    </div>
                                    <div class="metric-box" style="flex:1;">
                                        <div class="metric-label">Max Confidence</div>
                                        <div class="metric-value">{float(confs.max())*100:.1f}%</div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("No tumour region detected in this scan.", icon="ℹ️")

                except Exception as e:
                    st.error(f"Analysis failed: {e}", icon="🚨")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    # ── Empty-state prompt ────────────────────────────────────
    st.markdown("""
        <div class="glass-card" style="text-align:center; padding: 50px 30px;">
            <div style="font-size: 3.5rem; margin-bottom: 16px;">🧠</div>
            <p style="font-family:'Orbitron',monospace; font-size:1rem;
                       color:#00e5ff; letter-spacing:2px; margin-bottom:10px;">
                AWAITING MRI INPUT
            </p>
            <p style="font-family:'Exo 2',sans-serif; font-size:0.9rem;
                       color:#7aaccc; max-width:400px; margin:0 auto; line-height:1.7;">
                Upload a brain MRI scan using the uploader above, then select your
                preferred analysis mode from the sidebar and click <strong>Run AI Analysis</strong>.
            </p>
        </div>
    """, unsafe_allow_html=True)


# ===================== FOOTER =====================
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
st.markdown("""
    <div class="footer">
        🧠 Brain Tumor AI Suite &nbsp;•&nbsp;
        EfficientNet-B4 &nbsp;|&nbsp; U-Net &nbsp;|&nbsp; YOLOv8 &nbsp;•&nbsp;
        Built with TensorFlow · Ultralytics · Streamlit
    </div>
""", unsafe_allow_html=True)
