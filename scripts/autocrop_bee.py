# install Pillow if not already installed:
# python3 -m pip install pillow
# doesnt entirely work the way we want it with some crops being strange. Maybe because of shadow.

# written by KC Seltmann with ChatGPT on July 23, 2025
# batch-crops all PNGs in a folder by removing transparent borders

from PIL import Image
import os

# === CONFIGURATION ===
input_folder = "/Volumes/IMAGES/bombus_synthetic_image_data/extracted_bees/B_vos"

# === PROCESS ALL PNG FILES IN FOLDER ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        file_path = os.path.join(input_folder, filename)

        try:
            image = Image.open(file_path).convert("RGBA")
            bbox = image.getbbox()

            if bbox:
                cropped = image.crop(bbox)
                cropped.save(file_path)
                print(f"✅ Cropped and saved: {filename}")
            else:
                print(f"⚠️ Skipped (fully transparent): {filename}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

