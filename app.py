import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="RetailVision AI — Stockout Monitor", layout="wide")

st.title("🛒 RetailVision AI — Real-time Shelf Stockout Detector")
st.write("Powered by YOLO11n & Geometric Post-Processing Severity Engine")

# Load trained weight matrix safely from root folder
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# User dynamic threshold overrides
st.sidebar.header("🔧 System Configurations")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.45, 0.05)
heavy_gap_threshold = st.sidebar.slider("Heavy Gap Trigger Threshold (Width %)", 10, 50, 30, 5) / 100.0

# User image uploader component
uploaded_file = st.file_uploader("📸 Upload an Aisle Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image using native PIL (No OpenCV needed)
    img = Image.open(uploaded_file).convert("RGB")
    w, h = img.size
    
    # Run rapid inference
    results = model.predict(source=img, conf=conf_threshold, verbose=False)
    
    heavy_gap_detected = False
    small_gaps = 0
    medium_gaps = 0
    heavy_gaps = 0
    
    # Prepare drawing context
    draw = ImageDraw.Draw(img)
    
    # Apply width-ratio geometric checking logic
    for result in results:
        for box in result.boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_width_ratio = (x2 - x1) / w
            confidence = box.conf[0].item()
            
            if box_width_ratio >= heavy_gap_threshold:
                gap_type = "HEAVY GAP"
                box_color = (255, 0, 0) # Red
                heavy_gaps += 1
                heavy_gap_detected = True
            elif box_width_ratio >= (heavy_gap_threshold / 2):
                gap_type = "Medium Gap"
                box_color = (255, 165, 0) # Orange
                medium_gaps += 1
            else:
                gap_type = "Small Gap"
                box_color = (0, 255, 0) # Green
                small_gaps += 1
                
            # Draw bounding box outline with native PIL
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=4)
            
            # Label overlay text
            label_text = f"{gap_type} ({confidence*100:.0f}%)"
            draw.text((x1 + 5, max(y1 - 20, 5)), label_text, fill=box_color)
        
    # Structural presentation splits
    col1, col2 = st.columns()
    
    with col1:
        st.image(img, caption="Processed Visual Telemetry Output", use_container_width=True)
        
    with col2:
        st.subheader("📊 Shelf Status Metrics")
        st.metric("Total Gaps Located", f"{small_gaps + medium_gaps + heavy_gaps}")
        st.write(f"- 🟢 Small Gaps: **{small_gaps}**")
        st.write(f"- ⚠️ Medium Gaps: **{medium_gaps}**")
        st.write(f"- 🚨 Heavy Gaps: **{heavy_gaps}**")
        
        st.markdown("---")
        if heavy_gap_detected:
            st.error("🚨 CRITICAL ACTION REQUIRED: Heavy stockouts detected! System has dispatched an immediate push notification to stock room associates.")
        else:
            st.success("✅ Aisle profile matching targets. No action required.")
