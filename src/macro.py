import pyautogui
import time
import os

# --- CONFIGURATION ---
TOTAL_SCREENSHOTS = 5  # num of screenshots
CLICK_DELAY = 1.0        # Wait 1s after clicking
SCREENSHOT_DELAY = 1.0   # Wait 1s for overlay to appear before hitting Enter
SAVE_DELAY = 0.5         # Wait 0.5s after Enter to ensure save completes

# Optional: specific coordinates for the "Next" button or row expansion
# If set to None, it will click wherever your mouse currently is.
CLICK_X = None 
CLICK_Y = None 
# ---------------------

print("⚠️  Make sure you have manually set your Cmd+Shift+5 selection area first!")
print("⚠️  Move your mouse to the clickable area (if not hardcoded).")
print("Starting in 5 seconds... switch to your window now.")
time.sleep(5)

for i in range(TOTAL_SCREENSHOTS):
    # 1. Perform the Click
    if CLICK_X and CLICK_Y:
        pyautogui.click(CLICK_X, CLICK_Y)
    else:
        pyautogui.click() # Clicks at current mouse position
    
    # 2. Wait for page to update
    time.sleep(CLICK_DELAY)
    
    # 3. Trigger Screenshot Mode (Cmd + Shift + 5)
    pyautogui.hotkey('command', 'shift', '5')
    time.sleep(SCREENSHOT_DELAY) # Wait for overlay
    
    # 4. Confirm Screenshot (Enter)
    pyautogui.press('enter')
    time.sleep(SAVE_DELAY) # Wait for file save
    
    print(f"Captured screenshot {i+1}/{TOTAL_SCREENSHOTS}")

print("Done!")