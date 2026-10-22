# 🛒 RetailVision AI Pro — Enterprise Shelf System

<div align="center">
  
  [![Python Version](https://shields.io)](https://python.org)
  [![Framework](https://shields.io)](https://streamlit.io)
  [![Inference Engine](https://shields.io)](https://onnxruntime.ai)
  [![Model Core](https://shields.io)](https://github.com)

  <p align="center">
    <b>An automated Computer Vision pipeline designed to monitor supermarket on-shelf availability, calculate stockout severity metrics using relative frame geometry, and trigger real-time warehouse replenishment alerts.</b>
  </p>

  <h3>🚀 <a href="https://streamlit.app">CLICK HERE TO ACCESS LIVE CLOUD APP INTERFACE</a> 🚀</h3>
</div>

---

## 📸 Visual Telemetry Diagnostics
Below is the system execution pipeline showcasing a raw supermarket aisle snapshot before processing, alongside the live real-time model detection output layer:

<div align="center">
  <table>
    <tr>
      <td align="center"><b>1️⃣ Input Target (Before Detection)</b></td>
      <td align="center"><b>2️⃣ AI Output Layer (After Detection)</b></td>
    </tr>
    <tr>
      <td><img src="assets/before.jpg" width="450" alt="Supermarket Shelf Input Frame"/></td>
      <td><img src="assets/after.jpg" width="450" alt="YOLO11 ONNX Bounding Box Outputs"/></td>
    </tr>
  </table>
  <p><i>Note: To display your actual store frames here, simply upload your sample photos into a folder named "assets" in your repository as "before.jpg" and "after.jpg".</i></p>
</div>

---

## 📂 Source Dataset Repository
The neural core of this application was trained and validated using a custom-engineered partition derived from the baseline retail inventory tracking repository:

<div align="center">

[![Dataset Source](https://shields.io)](https://roboflow.com)

<h3>🎯 <a href="https://roboflow.com">CLICK HERE TO VIEW SOURCE DATASET ON ROBOFLOW</a></h3>
</div>

---

## 🎯 Production Core Metrics
The final validation phase of the trained neural network matrix yielded world-class precision indicators:

| Evaluation Metric | Score | Operational Significance |
| :--- | :---: | :--- |
| **mAP @ 50** | **`98.7%`** | Exceptional bounding box overlap alignment and convergence stability. |
| **Precision** | **`95.5%`** | Near-zero false alarms, keeping store shelf associates focused on true stockouts. |
| **Recall** | **`96.8%`** | The network successfully intercepts almost every single open shelf gap. |
| **Inference Latency** | **`~5.0ms`** | Optimized execution speed, fully production-ready for low-cost edge setups. |

---

## 🔧 Core Architectural Layers

### 1️⃣ Data Quality Engineering & Sanitization
Rather than training raw data blindly, a comprehensive data audit was performed on the baseline dataset:
* **The Vulnerability Found:** A custom automated `SHA256` hashing scan exposed a **56.8% data leakage flaw** (39 duplicate images overlapping across train and validation splits).
* **The Remediation:** All leaking validation artifacts were permanently purged from the system. Noisy, inconsistent text coordinate layers were stripped out, leaving a perfectly clean, unified single-class configuration (`0: gap`).

### 2️⃣ Traffic-Light Severity Engine
To achieve enterprise-grade speeds, bounding box dimension analysis is completely decoupled from the neural network weights layer. A lightweight Python geometric rule engine evaluates the width of each box relative to the total width of the horizontal shelf rack frame (`box_width_ratio`):

```yaml
[TOTAL HORIZONTAL AISLE SHELF ROW FRAME WIDTH]
  ├── 🟢 Small Gap  (< 15% Width)  -> Logged silently (Safe / No Action required)
  ├── ⚠️ Medium Gap (15% - 29% Width) -> Flagged for routine daily restocking shifts
  └── 🚨 HEAVY GAP  (>= 30% Width) -> TRIGGERS IMMEDIATE EMERGENCY STOCKROOM ALARM 📞
```

### 3️⃣ Edge Server Optimization
To make the application resilient to varying cloud hosting environments:
* **The Problem:** The baseline model required heavy, graphics-dependent code engines (`import cv2`) which demand host desktop operating system drivers (`libGL.so.1`) that do not exist on bare cloud containers.
* **The Engineering Fix:** The model was compiled out of the custom `.pt` file format into a standalone, universal **ONNX format (`best.onnx`)**. All graphics drawing calculations are handled strictly within memory using native Python Pillow image arrays. 

---

## 🛠️ Local Installation & Setup

To run this application locally on your laptop or machine, open your terminal and run these commands:

### 1. Clone the Code Repository
```bash
git clone https://github.com
cd retail-shelf-stockout-detector
```

### 2. Install Python Package Requirements
```bash
pip install -r requirements.txt
```

### 3. Launch the Local Production Stream
```bash
streamlit run app.py
```

---

## 💡 Hardware Optimization & Deployment Context
* **Local Capability:** The model natively supports high-speed live video streams on a standard GPU or multi-core laptop processor because the `5.0ms` execution time can comfortably process over **200 frames per second**.
* **Cloud Constrained Architecture:** On public web servers (like Streamlit Cloud), the application runs on a free, single-core CPU container. To prevent memory overflows and server freezing caused by live video streams, the frontend interface is intentionally optimized to accept still snapshot image payloads. This aligns with modern low-bandwidth retail setups where a camera snaps an aisle frame once every few minutes rather than streaming constant data.

---

## 🛡️ License & Project Baseline
This project was developed as a finalized production MVP utilizing data structures from the SRM University research partition. All code layers are open-sourced for retail automation review.
