import cv2
import pytesseract
import pandas as pd
import os
import glob
import re
import numpy as np

# --- CONFIGURATION ---
IMAGE_FOLDER = "./2-data"
OUTPUT_CSV = "./1-output/layloData_v6.csv"

# TUNING PARAMETERS
# Header/Footer cuts
HEADER_CUT_PCT = 0.12 
FOOTER_CUT_PCT = 0.06 

def load_and_crop_vertical(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    h, w, _ = img.shape
    y_start = int(h * HEADER_CUT_PCT)
    y_end = int(h * (1.0 - FOOTER_CUT_PCT))
    return img[y_start:y_end, :]

def get_row_slices(img, num_rows=5):
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
    h, w, _ = row_img.shape
    crop = row_img[:, int(w * x1_pct):int(w * x2_pct)]
    
    # 1. Gray & Resize
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 2. Invert & Threshold
    inverted = cv2.bitwise_not(scaled)
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. OCR (Keep newlines for formatting logic later)
    config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(thresh, config=config).strip()

def process_number_cell(row_img, x1_pct, x2_pct):
    h, w, _ = row_img.shape
    crop = row_img[:, int(w * x1_pct):int(w * x2_pct)]
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    inverted = cv2.bitwise_not(scaled)
    
    thresh = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    return pytesseract.image_to_string(thresh, config=config).strip()

# --- PARSING LOGIC ---

def clean_location(text):
    """
    Replaces newlines with commas. 
    "Chicago\nIllinois" -> "Chicago, Illinois"
    """
    if not text: return ""
    # Replace newlines with comma+space
    text = text.replace('\n', ', ')
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_contact_details(raw_text):
    """
    Extracts Email, Handle, Phone, and Profile Name from raw text.
    """
    # Initialize
    email = None
    handle = None
    phone = None
    
    # 1. Extract Email
    # Regex: Standard email pattern
    email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', raw_text)
    if email_match:
        email = email_match.group(0)
        raw_text = raw_text.replace(email, '') # Remove from string

    # 2. Extract Instagram Handle
    # Regex: Starts with @, followed by allowed username chars
    handle_match = re.search(r'@[a-zA-Z0-9._]+', raw_text)
    if handle_match:
        handle = handle_match.group(0)
        raw_text = raw_text.replace(handle, '')

    # 3. Extract Phone Number
    # Regex: Matches (xxx) xxx-xxxx, +xx xxx..., or xxx-xxx-xxxx
    # It looks for a sequence of digits and separators that is at least 7 chars long
    phone_match = re.search(r'(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', raw_text)
    if phone_match:
        phone = phone_match.group(0)
        raw_text = raw_text.replace(phone, '')

    # 4. Extract Profile Name (Whatever is left)
    # Clean up newlines, extra spaces, and edge artifacts (., |, ))
    name = raw_text.replace('\n', ' ')
    # Remove artifacts often left by the icon on the left side
    name = re.sub(r'^[\s\)\.\-\|]+', '', name) 
    # Remove artifacts that might remain on the right
    name = re.sub(r'[\s]+$', '', name)
    name = re.sub(r'\s+', ' ', name).strip() # Collapse spaces

    return name, email, phone, handle

def clean_date(raw_text):
    match = re.search(r'[A-Z][a-z]{2} \d{1,2}, \d{4}', raw_text)
    return match.group(0) if match else raw_text.replace('\n', ' ').strip()

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
            # 1. Contact (0.09 - 0.29)
            contact_raw = process_text_cell(row_img, 0.09, 0.29)
            
            if len(contact_raw) < 2: continue

            # Parse the specific contact fields
            profile_name, email, phone, handle = parse_contact_details(contact_raw)

            # 2. Location (0.29 - 0.46)
            loc_raw = process_text_cell(row_img, 0.29, 0.46)
            location_clean = clean_location(loc_raw)
            
            # 3. Joined (0.46 - 0.63)
            joined_raw = process_text_cell(row_img, 0.46, 0.63)
            
            # 4. Channel (0.63 - 0.79)
            channel_raw = process_text_cell(row_img, 0.63, 0.79)
            
            # 5. Engagements (0.79 - 0.87)
            engage_raw = process_number_cell(row_img, 0.79, 0.87)
            
            record = {
                "Profile_Name": profile_name,
                "Email": email,
                "Phone": phone,
                "IG_Handle": handle,
                "Location": location_clean,
                "Joined_On": clean_date(joined_raw),
                "Acq_Channel": channel_raw.replace('\n', ' ').strip(),
                "Engagements": engage_raw
            }
            
            all_data.append(record)

    df = pd.DataFrame(all_data)
    
    # Reorder columns for readability
    cols = ["Profile_Name", "IG_Handle", "Email", "Phone", "Location", "Joined_On", "Acq_Channel", "Engagements"]
    df = df[cols]
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Success! Saved {len(df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()