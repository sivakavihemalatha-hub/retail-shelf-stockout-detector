import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(page_title="RetailVision AI Pro", layout="wide")

st.title("🛒 RetailVision AI Pro — Enterprise Shelf System")
st.write("Powered by YOLO11n Distributed Cloud Infrastructure Engine")

# Sidebar setup for live presentation settings
st.sidebar.header("🔧 System Configurations")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.45, 0.05)
heavy_gap_threshold = st.sidebar.slider("Heavy Gap Trigger Threshold (Width %)", 10, 50, 30, 5)

# Input tunnel link from your Google Colab notebook backend instance
ngrok_url = st.sidebar.text_input("🔗 Paste Your Active Colab Backend URL Link here:", placeholder="https://ngrok-free.app")

uploaded_file = st.file_uploader("📸 Upload a Supermarket Shelf Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and ngrok_url:
    st.info("⚡ Dispatched payload frame to high-speed GPU processing clusters...")
    
    # Pack the image file into a secure web transmission request payload stream
    file_bytes = uploaded_file.getvalue()
    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
    data = {
        "conf": str(conf_threshold),
        "heavy_thresh": str(heavy_gap_threshold)
    }
    
    try:
        # Ping your Google Colab computer backend loop
        response = requests.post(f"{ngrok_url}/detect", files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            # Unpack returned payload matrices
            result_json = response.json()
            
            small_gaps = result_json.get("small_gaps", 0)
            medium_gaps = result_json.get("medium_gaps", 0)
            heavy_gaps = result_json.get("heavy_gaps", 0)
            heavy_gap_detected = result_json.get("heavy_gap_detected", False)
            
            # Decode the annotated image returned by the Colab processor loop back into pixels
            img_data = base64.b64decode(result_json.get("image_base64"))
            img_output = Image.open(io.BytesIO(img_data))
            
            # Render interface splits fixed to explicit two-column configuration
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img_output, caption="Processed Visual Telemetry Output", use_container_width=True)
                
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
        else:
            st.error("❌ Communication Error: Backend processor rejected telemetry package pipeline.")
    except Exception as e:
        st.error(f"❌ Connection Fail: Unable to talk to your Colab backend loop. Make sure the URL link is active. ({e})")
elif uploaded_file is not None and not ngrok_url:
    st.warning("⚠️ Action Required: Please paste your active Colab Backend URL link inside the sidebar slot to execute processing.")
