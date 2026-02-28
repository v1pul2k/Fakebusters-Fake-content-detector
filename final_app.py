import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# =========================================================
#                     DEVICE
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
#                     IMAGE MODEL DEFINITION
# =========================================================
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Use weights=None to avoid protected "fc"
        base = models.resnet18(weights=None)

        in_feat = base.fc.in_features

        # Build your custom classifier as a Sequential
        classifier = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 1)
        )

        # Assign the classifier by modifying state_dict structure
        # instead of assigning to attribute "fc"
        base._modules['fc'] = classifier

        self.base = base

    def forward(self, x):
        return self.base(x)


# =========================================================
#                     LOAD IMAGE MODEL
# =========================================================
@st.cache_resource
def load_image_model():
    ckpt = torch.load("cnn_resnet18_best.pth", map_location=DEVICE)

    model = CNNModel().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    threshold = ckpt.get("threshold", 0.5)
    class_to_idx = ckpt.get("class_to_idx", {"Fake": 0, "Real": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return model, threshold, idx_to_class

image_model, IMAGE_THRESHOLD, IMAGE_idx_to_class = load_image_model()

# =========================================================
#                     IMAGE TRANSFORMS
# =========================================================
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# =========================================================
#                     IMAGE PREDICTION
# =========================================================
def predict_image(image):
    with torch.no_grad():
        tensor = image_transform(image).unsqueeze(0).to(DEVICE)
        logits = image_model(tensor)
        prob_real = torch.sigmoid(logits).item()
        prob_fake = 1 - prob_real

        if prob_real > IMAGE_THRESHOLD:
            return "Real", prob_real
        else:
            return "Fake", prob_fake

# =========================================================
#                     UI HEADER
# =========================================================
st.set_page_config(page_title="FakeBusters", layout="centered")

st.title("FakeBusters")
st.subheader("Your AI-Powered Deepfake Detector")
st.write("Upload media to find out if it's **Real or Fake**.")

# =========================================================
#                     IMAGE UPLOAD UI
# =========================================================
st.markdown("## 🖼️ Image Deepfake Detection")

uploaded_file = st.file_uploader("Choose an image...",
                                 type=["jpg", "jpeg", "png"],
                                 key="imageUploader")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        label, conf = predict_image(image)

    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {conf:.2%}")

# ======================================================================
#                     VIDEO SUPPORT (XCEPTION MODEL)
# ======================================================================
import cv2
import math
from model_xception import XceptionAvgTemporal

# ---------------- VIDEO HELPERS ----------------
def sample_video_frames(file_bytes, n=8):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total // n, 1)

    frames = []
    pos = 0
    while len(frames) < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
        pos += interval

    cap.release()
    return frames

@st.cache_resource
def load_video_model():
    model = XceptionAvgTemporal(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("xception_best.pth", map_location=DEVICE))
    model.eval()
    return model

def predict_video(file_bytes):
    model = load_video_model()

    transform_video = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    frames = sample_video_frames(file_bytes, n=8)
    seq = torch.stack([transform_video(f) for f in frames]).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(seq).item()

    prob_fake = 1 / (1 + math.exp(-logit))
    prob_real = 1 - prob_fake

    if prob_real > 0.5:
        return "Real", prob_real
    else:
        return "Fake", prob_fake

# ---------------- VIDEO UI ----------------
st.markdown("---")
st.markdown("## 🎥 Deepfake Video Detection")

video_file = st.file_uploader("Choose a video...",
                              type=["mp4", "mov", "avi", "mkv"],
                              key="videoUploader")

if video_file:
    st.video(video_file)

    with st.spinner("Analyzing video..."):
        label, prob = predict_video(video_file.read())

    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {prob:.2%}")

    if label == "Fake":
        st.error("⚠️ This video may be AI-generated or manipulated.")
    else:
        st.balloons()
        st.success("✅ This video appears authentic.")
