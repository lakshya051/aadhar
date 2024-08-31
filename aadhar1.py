import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import re

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def plt_imshow(title, image):
    # Convert the image frame BGR to RGB color space and display it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')
    st.pyplot(fig)

def detect_text_blocks(img):
    detection_result = reader.detect(img,
                                     width_ths=0.7,
                                     mag_ratio=1.5)
    text_coordinates = detection_result[0][0]
    return text_coordinates

def draw_bounds(img, bbox):
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    for b in bbox:
        p0, p1, p2, p3 = [b[0], b[2]], [b[1], b[2]], \
                         [b[1], b[3]], [b[0], b[3]]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill='red', width=2)
    return np.array(image)

def extract_info(text):
    data = {}
    if "income" in text.lower() or "tax" in text.lower() or "department" in text.lower():
        # PAN card
        data['ID Type'] = "PAN"
        data['PAN'] = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', text)
        data['Name'] = re.search(r'(?<=Name\s)(.+)(?=\n)', text, re.IGNORECASE)
        data['Father Name'] = re.search(r'(?<=Father\'s Name\s)(.+)(?=\n)', text, re.IGNORECASE)
        data['Date of Birth'] = re.search(r'\d{2}/\d{2}/\d{4}', text)
    elif "male" in text.lower() or "female" in text.lower() or "government of india" in text.lower():
        # Aadhaar card
        data['ID Type'] = "Aadhaar"
        data['Aadhaar Number'] = re.search(r'\d{4}\s\d{4}\s\d{4}', text)
        
        # Improved name extraction for Aadhaar
        name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', text)
        if name_match:
            data['Name'] = name_match.group(1)
        else:
            data['Name'] = "Not found"
        
        data['Date of Birth'] = re.search(r'(?:DOB|Date of Birth)?\s*:?\s*(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
        data['Sex'] = "Female" if "female" in text.lower() else "Male"
    
    # Clean up the extracted data
    for key, value in data.items():
        if value and hasattr(value, 'group'):
            data[key] = value.group().strip()
        elif value:
            data[key] = value.strip()
        else:
            data[key] = "Not found"
    
    return data

def main():
    st.title("ID Card OCR App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Display the original image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image
        if st.button("Process Image"):
            # Detect text blocks
            text_coordinates = detect_text_blocks(img_array)
            text_blocks_in_image = draw_bounds(img_array, text_coordinates)

            # Display image with bounding boxes
            st.subheader("Detected Text Regions")
            plt_imshow("Image with Bounding Boxes", text_blocks_in_image)

            # Recognize text
            recognition_results = reader.recognize(img_array,
                                                   horizontal_list=text_coordinates,
                                                   free_list=[])
            detected_text = [txt[1] for txt in recognition_results]
            detected_text = ",".join(detected_text)

            # Extract information
            extracted_info = extract_info(detected_text)

            # Display extracted information
            st.subheader("Extracted Information")
            for key, value in extracted_info.items():
                st.write(f"{key}: {value}")

            # Display raw detected text
            st.subheader("Raw Detected Text")
            st.text(detected_text)

if __name__ == "__main__":
    main()