import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

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
    # Convert file stream payload to OpenCV standard image array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w, _ = img.shape
    
    # Run rapid inference
    prediction_results = model.predict(source=img, conf=conf_threshold, verbose=False)
    
    # Extract the target tensor details out of the list framework
    result = prediction_results[0]
    
    heavy_gap_detected = False
    small_gaps = 0
    medium_gaps = 0
    heavy_gaps = 0
    
    # Apply width-ratio geometric checking logic exactly as tested in Colab
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
        box_width_ratio = (x2 - x1) / w
        confidence = box.conf.item()
        
        if box_width_ratio >= heavy_gap_threshold:
            gap_type = "🚨 HEAVY GAP"
            box_color = (0, 0, 255) # BGR Red for heavy alerts
            heavy_gaps += 1
            heavy_gap_detected = True
        elif box_width_ratio >= (heavy_gap_threshold / 2):
            gap_type = "⚠️ Medium Gap"
            box_color = (0, 165, 255) # BGR Orange
            medium_gaps += 1
        else:
            gap_type = "🟢 Small Gap"
            box_color = (0, 255, 0) # BGR Green
            small_gaps += 1
            
        # Inject colorful annotations onto image array matrix
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(img, f"{gap_type} ({confidence*100:.0f}%)", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
    # Re-align color spectrum matrix for web browser rendering
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Structural presentation splits fixed to explicit two-column configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img_rgb, caption="Processed Visual Telemetry Output", use_container_width=True)
        
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

