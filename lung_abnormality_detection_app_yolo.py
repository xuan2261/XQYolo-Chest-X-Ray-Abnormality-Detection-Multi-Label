import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import time
import pydicom

# Load YOLO model
model = YOLO('XQYolo-best.pt')  # Replace with your actual YOLO model path

# Class mapping for YOLO detections
class_mapping = {
    0: 'Aortic Enlargement',
    1: 'Atelectasis',
    2: 'Calcification',
    3: 'Cardiomegaly',
    4: 'Consolidation',
    5: 'Interstitial Lung Disease (ILD)',
    6: 'Infiltration',
    7: 'Lung Opacity',
    8: 'Nodule/Mass',
    9: 'Other Lesion',
    10: 'Pleural Effusion',
    11: 'Pleural Thickening',
    12: 'Pneumothorax',
    13: 'Pulmonary Fibrosis',
    14: 'No Abnormalities Detected'
}

# Define colors for each label (excluding "No Abnormalities Detected")
label2color = [
    [59, 238, 119], [222, 21, 229], [94, 49, 164], [206, 221, 133], [117, 75, 3],
    [210, 224, 119], [211, 176, 166], [63, 7, 197], [102, 65, 77], [194, 134, 175],
    [209, 219, 50], [255, 44, 47], [89, 125, 149], [110, 27, 100]
]

# Convert list of RGB values to tuple for easier usage with PIL
label2color = [tuple(color) for color in label2color]

from pydicom.pixel_data_handlers.util import apply_voi_lut

def read_xray(path, voi_lut = True, fix_monochrome = True, downscale_factor = 3):
    # dicom = pydicom.read_file(path)
    dicom = pydicom.dcmread(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255.0).astype(np.uint8)
    new_shape = tuple([int(x / downscale_factor) for x in data.shape])
    data = cv2.resize(data, (new_shape[1], new_shape[0]))

    return data

def is_valid_dicom(uploaded_file):
    if not uploaded_file.name.lower().endswith(('.dcm', '.dicom')):
        return False  # Kiểm tra phần mở rộng tệp

    try:
        file_bytes = uploaded_file.getvalue()  # Đọc nội dung tệp dưới dạng bytes
        header = file_bytes[128:135]
        if header[:4] != b'DICM':
            return False  # Kiểm tra magic number

        pydicom.dcmread(uploaded_file)  # Thử đọc bằng pydicom
        return True
    except pydicom.errors.InvalidDicomError:
        print("Có ngoại lệ xảy ra")
        return False

# Function to convert DICOM to image using read_xray
def dicom_to_image(dicom_file):
    try:
        # Sao chép tệp vào thư mục làm việc hiện tại
        with open(dicom_file.name, "wb") as f:
            f.write(dicom_file.getbuffer())

        image_array = read_xray(dicom_file.name) 
        if image_array is None:
            raise Exception("Lỗi khi đọc tệp DICOM")
        final_image = Image.fromarray(image_array)

        # Xóa tệp sau khi xử lý xong
        os.remove(dicom_file.name)

        return final_image
    except (pydicom.errors.InvalidDicomError, OSError, Exception) as e:
        st.error(f"Lỗi khi đọc tệp DICOM: {e}")
        return None

# Preprocess image for YOLO (ensures proper format)
def preprocess_image(image):
    try:
        # Convert to NumPy array
        image_array = np.array(image)
        # Ensure the image has 3 channels (RGB)
        if image_array.ndim == 2:
            image_array = np.stack((image_array,)*3, axis=-1)
        return image_array
    except Exception as e:
        st.warning("Please upload a valid radiograph.")
        return None

# Draw bounding boxes on image based on YOLO detections with larger text
def draw_bounding_boxes(image, results):
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    draw = ImageDraw.Draw(image)
    info_list = []  # List to store detected abnormalities and their confidence
    color_list = []  # List to store the colors corresponding to each detected abnormality
    has_detections = False  # To track if any detections are made
    
    # Set the font size
    font_size = 20  # You can adjust the font size as needed
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Load a custom font
    except IOError:
        font = ImageFont.load_default()  # Use default font if custom font is not found

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0]  # Confidence score of the prediction
            
            if class_id < 14:  # Skip the "No Abnormalities Detected" class
                color = tuple(map(int, label2color[class_id]))  # Ensure color is a tuple of integers
                label = class_mapping.get(class_id, "Unknown")
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                
                # Draw the bounding box
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
                
                # Draw the label and confidence with larger text
                text = f"{label} ({confidence:.2f})"
                
                # Get the size of the text bounding box
                text_bbox = draw.textbbox((x_min, y_min), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Add a background rectangle for the text to make it more visible
                draw.rectangle(
                    [(x_min, y_min - text_height), (x_min + text_width, y_min)],
                    fill=color
                )
                
                # Draw the text on top of the background
                draw.text((x_min, y_min - text_height), text, font=font, fill=(255, 255, 255))
                
                info_list.append(f"{label}: {confidence:.2f}")
                color_list.append(color)  # Store the color for this detection
                has_detections = True
    
    if not has_detections:
        st.write("No abnormalities detected with sufficient confidence.")
    
    return image, info_list, color_list

# Main function for Streamlit app
def main():
    st.title("Lung Abnormality Detection with YOLO")

    st.write("This tool uses YOLO for detecting lung abnormalities in chest X-rays.")
    st.markdown("---")
    
    # Test some sample images
    st.subheader("Test It Out")
    image_paths = ["app_images/image1.png", "app_images/image2.png", "app_images/image3.png", "app_images/image4.png"]

    # Function to resize the image to a specified width and height
    def resize_image(image_path, width, height):
        image = Image.open(image_path)
        resized_image = image.resize((width, height))
        return resized_image
    
    # Specify the width and height for resizing
    image_width = 400
    image_height = 400
    
    # Use st.columns to display multiple images side by side
    columns = st.columns(len(image_paths))

    # Display each image in a column with a "Detect" button underneath
    for idx, (column, image_path) in enumerate(zip(columns, image_paths)):
        resized_image = resize_image(image_path, image_width, image_height)
        column.image(resized_image, caption=f"Chest X-Ray {idx + 1}", use_container_width=True)

        # Add a "Detect" button centered below each image
        if column.button(f"Detect {idx + 1}"):
            with st.spinner(f"Processing chest x-ray {idx + 1}..."):
                # Simulate a delay to represent processing time
                time.sleep(2)
                image_data = preprocess_image(resized_image)
                results = model(image_data)  # Make predictions with YOLO
                output_image, info_list, color_list = draw_bounding_boxes(resized_image.copy(), results)  # Draw bounding boxes
                column.image(output_image, caption=f"Results for Chest X-Ray {idx + 1}")
                # Display abnormalities and confidence scores with matching colors
                if info_list:
                    st.write("### Detected Abnormalities and Confidence:")
                    for info, color in zip(info_list, color_list):
                        st.markdown(f"<span style='color: rgb{color};'>{info}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader('Submit Your Own X-ray')
    uploaded_file = st.file_uploader("Choose an image...", type=["dicom", "png", "jpg", "jpeg"])
    # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Check if it's a DICOM file or regular image
        if is_valid_dicom(uploaded_file):
            image = dicom_to_image(uploaded_file)
            st.image(image, caption="Uploaded DICOM Image", use_container_width=True)
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the uploaded image
        input_data = preprocess_image(image)

        if st.button("Detect Abnormalities"):
            with st.spinner("Model is making predictions..."):
                # Make predictions with YOLO
                results = model(input_data)
                # Draw bounding boxes on the uploaded image
                output_image, info_list, color_list = draw_bounding_boxes(image.copy(), results)
                st.image(output_image, caption="Predictions with Bounding Boxes")
                # Display abnormalities and confidence scores with matching colors
                if info_list:
                    st.write("### Detected Abnormalities and Confidence:")
                    for info, color in zip(info_list, color_list):
                        st.markdown(f"<span style='color: rgb{color};'>{info}</span>", unsafe_allow_html=True)

            st.write("*Important: Consult a healthcare professional for further advice.*")

    st.write("   ")
    st.markdown("---")

    github_project_url = "https://github.com/xuan2261/XQYolo-Chest-X-Ray-Abnormality-Detection-Multi-Label"
    github_project_markdown = f'[GitHub]({github_project_url})'

    st.write("   ")
    st.write(f"This A.I. tool is based on Yolo detect model using Python. The model currently has an Box(P  R  mAP50  mAP50-95): (0.476  0.446  0.432  0.274) and can be found in {github_project_markdown}. Please feel free to connect with me on LinkedIn or via email. Feedback is welcome!") 
    
    # Sidebar information
    st.sidebar.title('About the Creator')
    st.sidebar.image("app_images/headshot.jpg", use_container_width=True)

    linkedin_url = "https://www.linkedin.com/in/b%C3%B9i-xu%C3%A2n-bb9120170/"
    github_url = "https://github.com/xuan2261/"
    medium_url = "https://medium.com/@buithanhxuan2261"

    st.sidebar.subheader('Xuan Bui Thanh')
    st.sidebar.markdown(f'[LinkedIn]({linkedin_url}) | [GitHub]({github_url}) | [Blog]({medium_url})', unsafe_allow_html=True)
    st.sidebar.write('Contact: buithanhxuan2261@gmail.com')

if __name__ == "__main__":
    main()
