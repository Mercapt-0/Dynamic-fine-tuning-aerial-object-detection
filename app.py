import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the YOLO model
model = YOLO("custom.pt")

st.title("🖼️ Dynamic Fine-Tuning and Implementation of a High-Fidelity Geo-Spatial Object Recognition Framework for Real-Time and Static Surveillance")

# Upload an image
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Image dimensions
    img_height, img_width, _ = image_np.shape

    st.subheader("📏 Select Region of Interest (ROI)")
    
    # Sliders for ROI selection
    x_min = st.slider("X Min", 0, img_width, 50)
    y_min = st.slider("Y Min", 0, img_height, 50)
    x_max = st.slider("X Max", 0, img_width, 300)
    y_max = st.slider("Y Max", 0, img_height, 300)

    # Run YOLO prediction
    results = model.predict(source=image_np, conf=0.6)

    # Draw ROI box
    cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Process results
    for result in results:
        boxes = result.boxes.xyxy  # Bounding boxes
        class_ids = result.boxes.cls  # Class IDs
        names = result.names  # Class labels

        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = map(int, box[:4])

            # Check if the detection is inside ROI
            if x_min <= x1 and y_min <= y1 and x_max >= x2 and y_max >= y2:
                class_name = names[int(class_id)]  # Get class name

                # Draw bounding box & label
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image_np, class_name, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    # Show results
    st.image(image_np, caption="🎯 Detected Objects", channels="RGB")

    # Save processed image
    cv2.imwrite("roi_detected_output.jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    st.success("✅ Image saved as 'roi_detected_output.jpg'")

