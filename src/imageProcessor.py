import cv2
import pytesseract
import pandas as pd
import os
import glob
import re
import numpy as np

# --- CONFIGURATION ---
IMAGE_FOLDER = "./2-data"
OUTPUT_CSV = "./1-output/layloData_v5.csv"

# TUNING PARAMETERS
# Header/Footer cuts remain similar, roughly 12% top / 6% bottom
HEADER_CUT_PCT = 0.12 
FOOTER_CUT_PCT = 0.06 

def load_and_crop_vertical(image_path):
    """
    Loads image and crops the static header and footer.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w, _ = img.shape
    y_start = int(h * HEADER_CUT_PCT)
    y_end = int(h * (1.0 - FOOTER_CUT_PCT))
    return img[y_start:y_end, :]

def get_row_slices(img, num_rows=5):
    """
    Divides the cropped image into N even rows.
    """
    h, w, _ = img.shape
    row_height = h // num_rows
    rows = []
    for i in range(num_rows):
        y1 = i * row_height
        y2 = (i + 1) * row_height
        if y2 > h: y2 = h
        rows.append(img[y1:y2, :])
    return rows

def process_text_cell(row_img, x1_pct, x2_pct):
    """
    Standard Text Processing (Name, Location, Channel)
    """
    h, w, _ = row_img.shape
    crop = row_img[:, int(w * x1_pct):int(w * x2_pct)]
    
    # 1. Gray & Resize
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 2. Invert (Dark Mode -> Light Mode)
    inverted = cv2.bitwise_not(scaled)
    
    # 3. Otsu Thresholding (High Contrast)
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. OCR
    config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(thresh, config=config).strip()

def process_number_cell(row_img, x1_pct, x2_pct):
    """
    Number Processing (Engagement Bubbles)
    """
    h, w, _ = row_img.shape
    crop = row_img[:, int(w * x1_pct):int(w * x2_pct)]
    
    # 1. Gray & Resize
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # 2. Invert
    inverted = cv2.bitwise_not(scaled)
    
    # 3. Adaptive Threshold (Handles Yellow vs Grey buttons)
    thresh = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    
    # 4. OCR with Digit Whitelist
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    return pytesseract.image_to_string(thresh, config=config).strip()

# --- CLEANING HELPERS ---

def clean_contact_string(text):
    """
    Removes newlines and leading artifacts like ')' or '.' 
    that might leak from the icon on the left.
    """
    text = text.replace('\n', ' ').replace('\r', '').strip()
    # Regex: Remove leading punctuation often caused by icon edges
    text = re.sub(r'^[\s\)\.\-\|]+', '', text)
    return text

def clean_newlines(text):
    return text.replace('\n', ' ').replace('\r', '').strip()

def clean_date(raw_text):
    match = re.search(r'[A-Z][a-z]{2} \d{1,2}, \d{4}', raw_text)
    return match.group(0) if match else raw_text

# --- MAIN ---

def main():
    all_data = []
    image_files = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.png")))
    
    print(f"Processing {len(image_files)} images...")

    for img_file in image_files:
        processed_img = load_and_crop_vertical(img_file)
        if processed_img is None: continue
        
        rows = get_row_slices(processed_img)
        
        for row_img in rows:
            # --- ADJUSTED COORDINATES FOR WIDER IMAGE ---
            
            # 1. Contact: 9% to 29%
            # Shifted left to 0.09 to catch 'A' in Ale/Annalysia
            contact_raw = process_text_cell(row_img, 0.09, 0.29)
            
            # Skip empty rows
            if len(contact_raw) < 2: continue

            # 2. Location: 29% to 46%
            loc_raw = process_text_cell(row_img, 0.29, 0.46)
            
            # 3. Joined: 46% to 63%
            joined_raw = process_text_cell(row_img, 0.46, 0.63)
            
            # 4. Channel: 63% to 79%
            # Shifted left to 0.63 to catch 'T' in Twitter and 'O' in Other
            channel_raw = process_text_cell(row_img, 0.63, 0.79)
            
            # 5. Engagements: 79% to 87%
            # Stopped at 0.87 to EXCLUDE the purple star icon on the far right
            engage_raw = process_number_cell(row_img, 0.79, 0.87)
            
            record = {
                "Contact_Raw": clean_contact_string(contact_raw),
                "Location": clean_newlines(loc_raw),
                "Joined_On": clean_date(joined_raw),
                "Acq_Channel": clean_newlines(channel_raw),
                "Engagements": engage_raw
            }
            
            all_data.append(record)

    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Success! Saved {len(df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()