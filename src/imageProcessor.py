import cv2
import pytesseract
import pandas as pd
import os
import glob
import numpy as np

# --- CONFIGURATION ---
IMAGE_FOLDER = "./2-data"
OUTPUT_CSV = "./1-output/layloData_v2.csv"
custom_config = r'--oem 3 --psm 6'

# TUNING PARAMETERS (0.0 to 1.0)
# Adjust these if the rows are still cutting through text
HEADER_CUT_PCT = 0.13  # Cut top 13% of image
FOOTER_CUT_PCT = 0.10  # Cut bottom 10% of image

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    
    # 1. Crop Header and Footer FIRST
    h, w, _ = img.shape
    y_start = int(h * HEADER_CUT_PCT)
    y_end = int(h * (1.0 - FOOTER_CUT_PCT))
    cropped_img = img[y_start:y_end, :]
    
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    # 3. Invert (Dark Mode -> Light Mode)
    inverted = cv2.bitwise_not(gray)
    
    # 4. Binarize (Thresholding)
    # This forces everything to be either PURE BLACK or PURE WHITE.
    # Removes "gray" artifacts like faint lines or background noise.
    _, thresh = cv2.threshold(inverted, 180, 255, cv2.THRESH_BINARY)
    
    return thresh

def get_row_slices(img):
    """
    Splits the cropped data area into 5 even rows.
    """
    height, width = img.shape
    # We assume the visible area strictly contains 5 row slots
    row_height = height // 5
    
    rows = []
    for i in range(5):
        y1 = i * row_height
        y2 = (i + 1) * row_height
        
        # Safety check to ensure we don't go out of bounds
        if y2 > height: y2 = height
            
        rows.append(img[y1:y2, :])
        
    return rows

def extract_text(row_img, x1_pct, x2_pct):
    h, w = row_img.shape
    x1 = int(w * x1_pct)
    x2 = int(w * x2_pct)
    crop = row_img[:, x1:x2]
    
    # Add a little white border around the crop (helps OCR read edge text)
    crop = cv2.copyMakeBorder(crop, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
    
    return pytesseract.image_to_string(crop, config=custom_config).strip()

# --- MAIN LOOP ---

def main():
    all_data = []
    image_files = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.png")))
    
    print(f"Found {len(image_files)} images...")

    for idx, img_file in enumerate(image_files):
        # print(f"Processing {idx+1}...") 
        
        processed_img = preprocess_image(img_file)
        rows = get_row_slices(processed_img)
        
        for row_img in rows:
            # Extract Raw Data
            # I tweaked the percentages slightly based on your screenshot
            contact_raw = extract_text(row_img, 0.0, 0.28)
            
            # If contact is empty, the row is likely empty (end of list)
            if not contact_raw or len(contact_raw) < 3:
                continue

            record = {
                "Contact_Raw": contact_raw,
                # Location usually starts around 28%
                "Location": extract_text(row_img, 0.28, 0.48).replace('\n', ', '),
                # Joined starts around 48%
                "Joined_On": extract_text(row_img, 0.48, 0.68).replace('\n', ' '),
                # Channel starts around 68%
                "Acq_Channel": extract_text(row_img, 0.68, 0.82),
                # Engagements starts around 82%
                "Engagements": extract_text(row_img, 0.82, 1.0)
            }
            
            # Only add digits to engagements to clean up artifacts
            record["Engagements"] = ''.join(filter(str.isdigit, record["Engagements"]))
            
            all_data.append(record)

    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()