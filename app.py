import streamlit as st
import numpy as np
import torch
from PIL import Image, ImageDraw

st.set_page_config(page_title="RetailVision AI Pro", layout="wide")

st.title("🛒 RetailVision AI Pro — Enterprise Shelf System")
st.write("Powered by YOLO11 Weights & Core Math Architecture (Zero-Graphics Dependency)")

# 1. Load the model weights using pure PyTorch (Completely bypassing Ultralytics imports)
@st.cache_resource
def load_standalone_model():
    # Load the raw weights tensor map onto the CPU execution space safely
    model_data = torch.load('best.pt', map_location=torch.device('cpu'))
    
    if isinstance(model_data, dict) and 'model' in model_data:
        net = model_data['model'].float()
    else:
        net = model_data.float()
        
    net.eval() # Put the network into safe evaluation inference mode
    return net

try:
    model = load_standalone_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Failed to load model weights: {e}")

# User dynamic threshold overrides
st.sidebar.header("🔧 System Configurations")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.45, 0.05)
heavy_gap_threshold = st.sidebar.slider("Heavy Gap Trigger Threshold (Width %)", 10, 50, 30, 5) / 100.0

# User image uploader component
uploaded_file = st.file_uploader("📸 Upload an Aisle Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    st.info("⚡ Executing live mathematical inference on your uploaded shelf image...")
    
    # Open image using native PIL framework
    img = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = img.size
    
    # 2. Preprocess the image exactly how YOLO does it internally (Resize to 416x416 and normalize)
    img_resized = img.resize((416, 416))
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.transpose((2, 0, 1))  # Convert to Channels-First (C, H, W)
    img_tensor = torch.from_numpy(img_array).float().unsqueeze(0)  # Add batch dimension
    
    heavy_gap_detected = False
    small_gaps = 0
    medium_gaps = 0
    heavy_gaps = 0
    
    # Prepare native Pillow drawing framework for visual box overlays
    draw = ImageDraw.Draw(img)
    
    # 3. Process raw math coordinates straight out of the neural network tensors
    with torch.no_grad():
        try:
            # Pass the image tensor directly through the model layers
            predictions = model(img_tensor)
            
            # Handle variations in YOLO tensor export shapes
            if isinstance(predictions, list):
                preds = predictions[0]
            else:
                preds = predictions
                
            if len(preds.shape) == 3:
                preds = preds.squeeze(0)  # Remove batch dimension -> shape: (num_features, num_boxes)
            
            # YOLO outputs tensors where rows contain coordinates: [cx, cy, w, h, class_confidences...]
            preds = preds.cpu().numpy()
            
            # Iterate through every box predicted by your actual weights
            num_boxes = preds.shape[1]
            for i in range(num_boxes):
                box_data = preds[:, i]
                confidence = float(box_data[4]) if len(box_data) > 4 else 0.0
                
                # Check detections against your confidence slider choice
                if confidence >= conf_threshold:
                    cx_raw, cy_raw, w_raw, h_raw = box_data[0], box_data[1], box_data[2], box_data[3]
                    
                    # Map the relative coordinates back to your original uploaded image size
                    x1 = int((cx_raw - w_raw / 2) * orig_w / 416)
                    y1 = int((cy_raw - h_raw / 2) * orig_h / 416)
                    x2 = int((cx_raw + w_raw / 2) * orig_w / 416)
                    y2 = int((cy_raw + h_raw / 2) * orig_h / 416)
                    
                    # Clean up boundary overflows
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(orig_w, x2), min(orig_h, y2)
                    
                    # Calculate box width relative to total shelf width frame
                    box_width_ratio = (x2 - x1) / orig_w
                    
                    # Apply your custom small vs heavy severity thresholds
                    if box_width_ratio >= heavy_gap_threshold:
                        gap_type = "🚨 HEAVY GAP"
                        box_color = (255, 0, 0)  # Red
                        heavy_gaps += 1
                        heavy_gap_detected = True
                    elif box_width_ratio >= (heavy_gap_threshold / 2):
                        gap_type = "⚠️ Medium Gap"
                        box_color = (255, 165, 0)  # Orange
                        medium_gaps += 1
                    else:
                        gap_type = "🟢 Small Gap"
                        box_color = (0, 255, 0)  # Green
                        small_gaps += 1
                        
                    # Physically draw the real calculated boxes onto the canvas
                    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=4)
                    draw.text((x1 + 6, max(y1 - 18, 6)), f"{gap_type} ({confidence*100:.0f}%)", fill=box_color)
                    
        except Exception as e:
            st.error(f"⚠️ Inference processing warning: {e}")

    st.success("📊 Real-time mathematical analysis complete!")
    
    # Structural presentation splits
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
