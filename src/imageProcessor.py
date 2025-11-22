import cv2
import pytesseract
import pandas as pd
import os
import glob
import re
import numpy as np
import concurrent.futures
import time
from datetime import datetime

# --- CONFIGURATION ---
IMAGE_FOLDER = "./2-data"
OUTPUT_CSV = f"./1-output/layloData_{datetime.now().strftime('%m-%d_%H-%M')}.csv"

# TUNING PARAMETERS
HEADER_CUT_PCT = 0.12 
FOOTER_CUT_PCT = 0.06 

# --- HELPER FUNCTIONS ---

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
    """
    Standard processing for text fields (Name, Email, etc)
    """
    h, w, _ = row_img.shape
    crop = row_img[:, int(w * x1_pct):int(w * x2_pct)]
    
    # 1. Gray & Resize
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 2. Invert & Threshold
    inverted = cv2.bitwise_not(scaled)
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. OCR
    config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(thresh, config=config).strip()

def process_number_cell(row_img, x1_pct, x2_pct):
    """
    Robust Number Processing for Colored Buttons (Yellow/Gold/Grey).
    Uses Blue Channel Extraction to maximize contrast between Yellow (Low Blue) 
    and White Text (High Blue).
    """
    h, w, _ = row_img.shape
    crop = row_img[:, int(w * x1_pct):int(w * x2_pct)]
    
    # 1. Extract Blue Channel (OpenCV is BGR, so index 0 is Blue)
    # This turns Yellow buttons DARK and White text BRIGHT.
    b_channel = crop[:, :, 0]
    
    # 2. Resize (3x for small numbers)
    scaled = cv2.resize(b_channel, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # 3. Invert
    # We want the Text to be Black and Background to be White for Tesseract.
    # In b_channel: Text is High (255), Button is Low (~0-50).
    # After Inversion: Text is Low (0), Button is High (255).
    inverted = cv2.bitwise_not(scaled)
    
    # 4. Otsu Thresholding
    # This automatically finds the separation point between the button color and text
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Add Padding
    # OCR often fails if the number touches the edge of the crop. Add a 5px white border.
    thresh = cv2.copyMakeBorder(thresh, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
    
    # 6. OCR
    # psm 7 = Treat the image as a single text line (better for "42" than psm 6)
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(thresh, config=config).strip()
    
    return text

# --- PARSING LOGIC ---

def clean_location(text):
    if not text: return ""
    text = text.replace('\n', ', ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_contact_details(raw_text):
    email = None
    handle = None
    phone = None
    
    email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', raw_text)
    if email_match:
        email = email_match.group(0)
        raw_text = raw_text.replace(email, '')

    handle_match = re.search(r'@[a-zA-Z0-9._]+', raw_text)
    if handle_match:
        handle = handle_match.group(0)
        raw_text = raw_text.replace(handle, '')

    phone_match = re.search(r'(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', raw_text)
    if phone_match:
        phone = phone_match.group(0)
        raw_text = raw_text.replace(phone, '')

    name = raw_text.replace('\n', ' ')
    name = re.sub(r'^[\s\)\.\-\|]+', '', name) 
    name = re.sub(r'[\s]+$', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    return name, email, phone, handle

def clean_date(raw_text):
    match = re.search(r'[A-Z][a-z]{2} \d{1,2}, \d{4}', raw_text)
    return match.group(0) if match else raw_text.replace('\n', ' ').strip()

# --- WORKER FUNCTION ---

def process_single_image(img_file):
    # Prevent OpenCV from multithreading internally to avoid deadlock/thrashing
    cv2.setNumThreads(0)
    
    image_data = []
    try:
        processed_img = load_and_crop_vertical(img_file)
        if processed_img is None: 
            return []
        
        rows = get_row_slices(processed_img)
        
        for row_img in rows:
            # 1. Contact
            contact_raw = process_text_cell(row_img, 0.09, 0.29)
            
            if len(contact_raw) < 2: continue

            # Parse specific contact fields
            profile_name, email, phone, handle = parse_contact_details(contact_raw)

            # 2. Location
            loc_raw = process_text_cell(row_img, 0.29, 0.46)
            location_clean = clean_location(loc_raw)
            
            # 3. Joined
            joined_raw = process_text_cell(row_img, 0.46, 0.63)
            
            # 4. Channel
            channel_raw = process_text_cell(row_img, 0.63, 0.79)
            
            # 5. Engagements (Updated Logic)
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
            
            image_data.append(record)
            
    except Exception as e:
        print(f"Error processing {img_file}: {e}")
        
    return image_data

# --- MAIN ---

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.png")))
    total_files = len(image_files)
    
    print(f"Starting parallel processing for {total_files} images...")
    start_time = time.time()

    all_data = []
    
    # Use CPU count to determine workers
    max_workers = os.cpu_count() or 4 

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_image, img): img for img in image_files}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_file):
            data_from_image = future.result()
            if data_from_image:
                all_data.extend(data_from_image)
            
            completed_count += 1
            if completed_count % 10 == 0:
                print(f"Progress: {completed_count}/{total_files} images processed...")

    df = pd.DataFrame(all_data)
    
    cols = ["Profile_Name", "IG_Handle", "Email", "Phone", "Location", "Joined_On", "Acq_Channel", "Engagements"]
    
    if not df.empty:
        df = df[cols]
    
    df.to_csv(OUTPUT_CSV, index=False)
    
    elapsed = time.time() - start_time
    print(f"Success! Saved {len(df)} rows to {OUTPUT_CSV}")
    print(f"Total time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()