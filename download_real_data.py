import os
import requests
import shutil

DATA_DIR = "data"
BASE_URL = "https://raw.githubusercontent.com/MohamedAliHabib/Brain-Tumor-Detection/master/"

# Known filenames based on previous successful listings
NO_FILES = [
    "1 no.jpeg", "2 no.jpeg", "3 no.jpg", "4 no.jpg", "5 no.jpg", "6 no.jpg", "7 no.jpg", "8 no.jpg", "9 no.jpg",
    "10 no.jpg", "11 no.jpg", "12 no.jpg", "13 no.jpg", "14 no.jpg", "15 no.jpg", "17 no.jpg", "18 no.jpg", "19 no.jpg",
    "20 no.jpg", "21 no.jpg", "22 no.jpg", "23 no.jpg", "24 no.jpg", "25 no.jpg", "26 no.jpg", "27 no.jpg", "28 no.jpg",
    "29 no.jpg", "30 no.jpg", "31 no.jpg", "32 no.jpg", "33 no.jpg", "34 no.jpg", "35 no.jpg", "36 no.jpg", "37 no.jpg",
    "38 no.jpg", "39 no.jpg", "40 no.jpg", "41 no.jpg", "42 no.jpg", "43 no.jpg", "44 no.jpg", "45 no.jpg", "46 no.jpg",
    "N1.JPG", "N2.JPG", "N3.jpg", "N5.jpg", "N6.jpg", "N11.jpg", "N15.jpg", "N16.jpg", "N17.jpg", "N19.JPG",
    "N20.JPG", "N21.jpg", "N22.JPG", "N26.JPG",
    "No11.jpg", "No12.jpg", "No13.jpg", "No14.jpg", "No15.jpg", "No16.jpg", "No17.jpg", "No18.jpg", "No19.jpg", "No20.jpg"
]

YES_FILES = [
    "Y1.jpg", "Y2.jpg", "Y3.jpg", "Y4.jpg", "Y6.jpg", "Y7.jpg", "Y8.jpg", "Y9.jpg", "Y10.jpg",
    "Y11.jpg", "Y12.jpg", "Y13.jpg", "Y14.jpg", "Y15.jpg", "Y16.JPG", "Y17.jpg", "Y18.JPG", "Y19.JPG", "Y20.jpg",
    "Y21.jpg", "Y22.jpg", "Y23.JPG", "Y24.jpg", "Y25.jpg", "Y26.jpg", "Y27.jpg", "Y28.jpg", "Y29.jpg", "Y30.jpg",
    "Y31.jpg", "Y32.jpg", "Y33.jpg", "Y34.jpg", "Y35.jpg", "Y36.JPG", "Y37.jpg", "Y38.jpg", "Y39.jpg", "Y40.JPG",
    "Y41.jpg", "Y42.jpg", "Y44.JPG", "Y45.JPG", "Y46.jpg", "Y47.JPG", "Y49.JPG", "Y50.JPG", "Y51.jpg", "Y52.jpg",
    "Y53.jpg", "Y54.jpg", "Y55.jpg", "Y56.jpg", "Y58.JPG", "Y59.JPG", "Y60.jpg", "Y61.jpg", "Y62.jpg", "Y65.JPG",
    "Y66.JPG", "Y67.JPG", "Y69.jpg", "Y70.jpg", "Y71.JPG", "Y73.jpg", "Y74.jpg", "Y75.JPG", "Y76.jpg", "Y77.jpg",
    "Y78.jpg", "Y79.jpg", "Y81.jpg", "Y82.jpg", "Y85.JPG", "Y86.JPG", "Y89.JPG", "Y90.jpg", "Y91.jpg", "Y92.jpg",
    "Y95.jpg", "Y96.jpg", "Y97.JPG", "Y98.JPG", "Y99.JPG", "Y100.JPG"
]

def download_file(url, save_path):
    try:
        # Handle spaces
        url = url.replace(" ", "%20")
        r = requests.get(url, stream=True, timeout=5)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            print(f"[OK] Downloaded: {os.path.basename(save_path)}")
            return True
        else:
            print(f"[FAIL] Failed ({r.status_code}): {url}")
    except Exception as e:
        print(f"[ERR] Error: {e}")
    return False

def main():
    print("Starting REAL data download...")
    
    # Setup directories
    for category in ["yes", "no"]:
        path = os.path.join(DATA_DIR, "train", category)
        if not os.path.exists(path):
            os.makedirs(path)
            
        val_path = os.path.join(DATA_DIR, "val", category)
        if not os.path.exists(val_path):
            os.makedirs(val_path)

    # Download YES
    print("\n--- Downloading Tumor (YES) Images ---")
    count_yes = 0
    save_dir_yes = os.path.join(DATA_DIR, "train", "yes")
    for filename in YES_FILES:
        if download_file(BASE_URL + "yes/" + filename, os.path.join(save_dir_yes, filename)):
            count_yes += 1

    # Download NO
    print("\n--- Downloading Non-Tumor (NO) Images ---")
    count_no = 0
    save_dir_no = os.path.join(DATA_DIR, "train", "no")
    for filename in NO_FILES:
        if download_file(BASE_URL + "no/" + filename, os.path.join(save_dir_no, filename)):
            count_no += 1
            
    print(f"\nSummary: {count_yes} Tumor images, {count_no} Non-Tumor images downloaded.")
    
    # Move some to val
    for category in ["yes", "no"]:
        train_path = os.path.join(DATA_DIR, "train", category)
        val_path = os.path.join(DATA_DIR, "val", category)
        files = os.listdir(train_path)
        # Move 20% to val
        num_to_move = int(len(files) * 0.2)
        for i in range(num_to_move):
            shutil.move(os.path.join(train_path, files[i]), os.path.join(val_path, files[i]))
    
    print("Data split into Train and Val.")

if __name__ == "__main__":
    main()
