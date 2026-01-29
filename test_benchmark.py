import requests
import os
import time

# --- CONFIGURATION ---
BASE_URL = "http://localhost:8080"
# ⚠️ REPLACE THIS with your exact filename
TEST_IMAGE = r"F:\pihealth\supposed_OCR\lbmaske\GUR-0425-CL-0196324_INW_GUR-0425-CL-0196324_27042025165908.pdf_page_16.png"

def run_test():
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Error: Could not find file '{TEST_IMAGE}'")
        print("   Make sure the image is in the same folder as this script.")
        return

    print(f"🚀 Processing: {TEST_IMAGE}")
    print("-" * 50)

    # --- STEP 1: GET THE DEBUG IMAGE ---
    print("📸 Downloading 'Robot View' (Debug Image)...")
    try:
        with open(TEST_IMAGE, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BASE_URL}/debug-view", files=files)
            
            if response.status_code == 200:
                with open("debug_output.jpg", "wb") as out:
                    out.write(response.content)
                print("   ✅ Saved to 'debug_output.jpg'. OPEN THIS FILE NOW!")
            else:
                print(f"   ❌ Failed to get debug image. Status: {response.status_code}")
                print(response.text)
                
    except Exception as e:
        print(f"   ❌ Connection Error: {e}")

    print("-" * 50)

    # --- STEP 2: GET THE TEXT ---
    print("📝 Extracting Text...")
    start_time = time.time()
    try:
        with open(TEST_IMAGE, "rb") as f:
            files = {"file": f}
            # Note: We use the /ocr endpoint here
            response = requests.post(f"{BASE_URL}/ocr", files=files)
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                text = data.get("text", "")
                
                print(f"   ✅ Success! (Took {duration:.2f}s)")
                print("\n⬇️ --- EXTRACTED TEXT START --- ⬇️")
                print(text.strip())
                print("⬆️ --- EXTRACTED TEXT END --- ⬆️")
            else:
                print(f"   ❌ Failed. Status: {response.status_code}")
                print(response.text)

    except Exception as e:
        print(f"   ❌ Connection Error: {e}")

if __name__ == "__main__":
    run_test()