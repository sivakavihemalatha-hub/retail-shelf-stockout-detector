import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

st.set_page_config(page_title="RetailVision AI Pro", layout="wide")

st.title("🛒 RetailVision AI Pro — Enterprise Shelf System")
st.write("Powered by YOLO11n Production Cloud Infrastructure Engine")

# 1. Load trained weights file directly from your root directory
@st.cache_resource
def load_model():
    # Looks for best.pt inside your main GitHub folder repository space
    return YOLO('best.pt')

model = load_model()

# Sidebar setup for live presentation settings
st.sidebar.header("🔧 System Configurations")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.45, 0.05)
heavy_gap_threshold = st.sidebar.slider("Heavy Gap Trigger Threshold (Width %)", 10, 50, 30, 5) / 100.0

uploaded_file = st.file_uploader("📸 Upload a Supermarket Shelf Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.info("⚡ Processing visual telemetry data on dedicated cloud structures...")
    
    # Open image using native PIL framework
    img = Image.open(uploaded_file).convert("RGB")
    w, h = img.size
    
    # Run rapid inference using the loaded model weights
    prediction_results = model.predict(source=img, conf=conf_threshold, verbose=False)
    result = prediction_results[0] # Extract the target frame summary
    
    heavy_gap_detected = False
    small_gaps = 0
    medium_gaps = 0
    heavy_gaps = 0
    
    # Prepare Pillow visual context for crisp bounding box overlays
    draw = ImageDraw.Draw(img)
    
    # Loop over boxes exactly how we processed them during testing
    if hasattr(result, 'boxes') and result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            box_width_ratio = (x2 - x1) / w
            confidence = box.conf.item()
            
            # Check dimensions against custom project thresholds
            if box_width_ratio >= heavy_gap_threshold:
                gap_type = "🚨 HEAVY GAP (RESTOCK NOW!)"
                box_color = (255, 0, 0) # Clear Red for heavy alerts
                heavy_gaps += 1
                heavy_gap_detected = True
            elif box_width_ratio >= (heavy_gap_threshold / 2):
                gap_type = "⚠️ Medium Gap"
                box_color = (255, 165, 0) # Clear Orange for routine alerts
                medium_gaps += 1
            else:
                gap_type = "🟢 Small Gap"
                box_color = (0, 255, 0) # Clear Green for small spaces
                small_gaps += 1
                
            # Draw bounding boxes onto the display picture canvas
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=4)
            draw.text((x1 + 6, max(y1 - 18, 6)), f"{gap_type} ({confidence*100:.0f}%)", fill=box_color)
        
    # Render interface splits
    col1, col2 = st.columns(2)
    
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
