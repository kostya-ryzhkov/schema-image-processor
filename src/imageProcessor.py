import cv2
import pytesseract
import pandas as pd
import os
import glob

# Point this to where your screenshots are
IMAGE_FOLDER = "./2-data"
OUTPUT_CSV = "./1-output/layloData.csv"

# Tesseract configuration (optimize for block of text)
custom_config = r'--oem 3 --psm 6'

def preprocess_image(image_path):
    """
    Loads image, converts to grayscale, and inverts (Dark mode -> Light mode)
    """

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert: Black background becomes white, text becomes black
    inverted = cv2.bitwise_not(gray)
    
    # Optional: Thresholding to make text crisp black and white
    # _, thresh = cv2.threshold(inverted, 150, 255, cv2.THRESH_BINARY)
    
    return inverted

def get_row_slices(img):
    """
    Splits the image into 5 distinct rows based on fixed height or line detection.
    Since macros are consistent, we can try fixed slicing first.
    Assumption: 5 rows of equal height.
    """

    height, width = img.shape
    row_height = height // 5
    
    rows = []
    for i in range(5):
        # y_start, y_end
        y1 = i * row_height
        y2 = (i + 1) * row_height
        
        # Crop the row
        row_img = img[y1:y2, 0:width]
        rows.append(row_img)
        
    return rows

def extract_text_from_area(row_img, x_start_pct, x_end_pct):
    """
    Crops a specific column from a row based on percentage width 
    and runs OCR.
    """

    h, w = row_img.shape
    x1 = int(w * x_start_pct)
    x2 = int(w * x_end_pct)
    
    # Crop column
    crop = row_img[0:h, x1:x2]
    
    # Run OCR
    text = pytesseract.image_to_string(crop, config=custom_config)
    return text.strip()

# --- SPECIFIC COLUMN EXTRACTORS ---

def parse_contact(row_img):
    """
    PRIMED: Currently just dumps raw text.
    We will add the complex regex/ruleset here later.
    """

    # Approx width: 0% to 30% of the screen
    raw_text = extract_text_from_area(row_img, 0.0, 0.30)
    return raw_text

def parse_location(row_img):
    # Approx width: 30% to 48%
    raw_text = extract_text_from_area(row_img, 0.30, 0.48)
    # Cleanup: Replace newlines with space or comma
    return raw_text.replace('\n', ', ')

def parse_joined(row_img):
    # Approx width: 48% to 65%
    raw_text = extract_text_from_area(row_img, 0.48, 0.65)
    return raw_text.replace('\n', ' ')

def parse_channel(row_img):
    # Approx width: 65% to 80%
    raw_text = extract_text_from_area(row_img, 0.65, 0.80)
    return raw_text

def parse_engagements(row_img):
    # Approx width: 80% to 100%
    raw_text = extract_text_from_area(row_img, 0.80, 1.0)
    # Extract only digits
    digits = ''.join(filter(str.isdigit, raw_text))
    return digits

# --- MAIN EXECUTION ---

def main():
    all_data = []
    image_files = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.png")))
    
    print(f"Found {len(image_files)} images to process...")

    for idx, img_file in enumerate(image_files):
        print(f"Processing {idx+1}/{len(image_files)}: {os.path.basename(img_file)}")
        
        processed_img = preprocess_image(img_file)
        rows = get_row_slices(processed_img)
        
        for row_img in rows:
            # Extract data
            record = {
                "Contact_Raw": parse_contact(row_img),
                "Location": parse_location(row_img),
                "Joined_On": parse_joined(row_img),
                "Acq_Channel": parse_channel(row_img),
                "Engagements": parse_engagements(row_img)
            }
            
            # Basic validation: ignore empty rows if any
            if record["Contact_Raw"]: 
                all_data.append(record)

    # Save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Saved {len(df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()