import os
import requests
import shutil

DATA_DIR = "data"
BASE_URL = "https://raw.githubusercontent.com/MohamedAliHabib/Brain-Tumor-Detection/master/"

def download_file(url, save_path):
    try:
        r = requests.get(url, stream=True, timeout=5)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            return True
    except Exception as e:
        pass
    return False

def download_dataset():
    print("Attempting to download real MRI data...")
    
    categories = {
        "yes": ["Y", "y"], 
        "no": ["N", "n", "No", "no", "No", "no ", "No "]
    }
    
    for category, prefixes in categories.items():
        count = 0
        save_dir = os.path.join(DATA_DIR, "train", category)
        # Don't wipe if we want to accumulate, but here we want a clean start for real data
        if os.path.exists(save_dir) and count == 0:
             # Only wipe if it looks like dummy data (check for dummy_0.jpg)
             if os.path.exists(os.path.join(save_dir, "dummy_0.jpg")):
                 shutil.rmtree(save_dir)
                 os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        
        # Try to download up to 300 images per class
        for i in range(1, 1000): # Iterate through possible indices
            if count >= 300: # Increased limit
                break
                
            # Patterns to try
            filenames = []
            for ext in ["jpg", "JPG", "jpeg", "png"]:
                filenames.append(f"Y{i}.{ext}") if category == 'yes' else None
                filenames.append(f"y{i}.{ext}") if category == 'yes' else None
                
                if category == 'no':
                    filenames.append(f"N{i}.{ext}")
                    filenames.append(f"n{i}.{ext}")
                    filenames.append(f"No{i}.{ext}")
                    filenames.append(f"no{i}.{ext}")
                    filenames.append(f"{i} no.{ext}")
                    filenames.append(f"{i} no.{ext}")
            
            for filename in filenames:
                # Handle spaces in URL
                url_filename = filename.replace(" ", "%20")
                url = BASE_URL + category + "/" + url_filename
                save_path = os.path.join(save_dir, filename)
                
                if not os.path.exists(save_path):
                    if download_file(url, save_path):
                        print(f"Downloaded {filename}")
                        count += 1
                        
                        # Save a sample for testing
                        if category == "no" and count == 1:
                            shutil.copy(save_path, "test_image_no.jpg")
                        if category == "yes" and count == 1:
                            shutil.copy(save_path, "test_image_yes.jpg")
                            
                        break # Found file for this number
                else:
                    count += 1
                    break
                    
        print(f"Downloaded {count} images for category '{category}'")
    
    # Copy to validation
    for category in ["yes", "no"]:
        val_dir = os.path.join(DATA_DIR, "val", category)
        if os.path.exists(val_dir):
            shutil.rmtree(val_dir)
        os.makedirs(val_dir, exist_ok=True)
        
        train_dir = os.path.join(DATA_DIR, "train", category)
        files = os.listdir(train_dir)
        for f in files[:5]: # Move 5 to val
            shutil.copy(os.path.join(train_dir, f), os.path.join(val_dir, f))

if __name__ == "__main__":
    download_dataset()
