
import streamlit as st
from ultralytics import YOLO
import joblib
import tempfile
import pandas as pd

st.set_page_config(
    page_title="AI Supermarket Vision Agent",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 AI Supermarket Vision Agent")
st.subheader("Stock Refilling Alert")

st.write(
    "Upload a supermarket image to detect empty shelf regions "
    "and determine whether restocking is required."
)

# Load trained models
model = YOLO("best.pt")
restock_model = joblib.load("restocking_decision_model.pkl")


# -----------------------------
# Region grouping functions
# -----------------------------

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area1 = box_area(box1)
    area2 = box_area(box2)

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def should_group(box1, box2):

    if iou(box1, box2) >= 0.25:
        return True

    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2

    horizontal_gap = max(
        0,
        max(x1a, x1b) - min(x2a, x2b)
    )

    vertical_overlap = max(
        0,
        min(y2a, y2b) - max(y1a, y1b)
    )

    height_a = y2a - y1a
    height_b = y2b - y1b

    smaller_height = min(height_a, height_b)

    vertical_alignment = (
        vertical_overlap / smaller_height
        if smaller_height > 0 else 0
    )

    if horizontal_gap <= 25 and vertical_alignment >= 0.50:
        return True

    return False


def combine_regions(boxes):

    groups = []

    for box in boxes:

        placed = False

        for group in groups:

            if any(
                should_group(box, other)
                for other in group
            ):
                group.append(box)
                placed = True
                break

        if not placed:
            groups.append([box])

    combined = []

    for group in groups:

        x1 = min(b[0] for b in group)
        y1 = min(b[1] for b in group)
        x2 = max(b[2] for b in group)
        y2 = max(b[3] for b in group)

        combined.append(
            [x1, y1, x2, y2]
        )

    return combined


# -----------------------------
# Image upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a supermarket image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded supermarket image",
        use_container_width=True
    )

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        image_path = temp_file.name


    # -----------------------------
    # YOLO detection
    # -----------------------------

    results = model.predict(
        source=image_path,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()

    # Combine nearby empty regions
    combined_regions = combine_regions(boxes)


    # -----------------------------
    # Calculate features
    # -----------------------------

    largest_area = max(
        [
            box_area(box)
            for box in combined_regions
        ],
        default=0
    )

    total_area = sum(
        box_area(box)
        for box in combined_regions
    )


    # -----------------------------
    # Restocking decision
    # -----------------------------

    features = pd.DataFrame(
        [[largest_area, total_area]],
        columns=[
            "largest_area",
            "total_area"
        ]
    )

    probabilities = restock_model.predict_proba(
        features
    )[0]

    classes = list(
        restock_model.classes_
    )

    # Get probability of the RESTOCK class
    restock_index = classes.index(1)

    restock_probability = probabilities[
        restock_index
    ]

    # Threshold selected during final evaluation
    RESTOCK_THRESHOLD = 0.45

    if restock_probability >= RESTOCK_THRESHOLD:
        prediction = 1
    else:
        prediction = 0

    probability = restock_probability


    # -----------------------------
    # Display YOLO result
    # -----------------------------

    annotated_image = result.plot()

    st.subheader(
        "🔍 Empty-Shelf Detection"
    )

    st.image(
        annotated_image,
        caption="Detected empty shelf regions",
        use_container_width=True
    )


    # -----------------------------
    # Display measurements
    # -----------------------------

    st.subheader(
        "📊 Analysis"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "YOLO detections",
            len(boxes)
        )

    with col2:
        st.metric(
            "Combined regions",
            len(combined_regions)
        )

    with col3:
        st.metric(
            "Total empty area",
            f"{total_area:.0f}"
        )


    # -----------------------------
    # Final decision
    # -----------------------------

    st.subheader(
        "🚨 Restocking Decision"
    )

    if prediction == 1:

        st.error(
            f"🔴 RESTOCK REQUIRED"
        )

    else:

        st.success(
            f"🟢 NO RESTOCK REQUIRED"
        )

    st.write(
        "Decision probability:",
        f"{probability:.1%}"
    )
