# pip install pillow if needed:
# python3 -m pip install pillow

from PIL import Image
import os
import random

# === CONFIGURATION ===
bee_folder = "/Volumes/IMAGES/bombus_synthetic_image_data/extracted_bees/B_vos"
background_root = "/Volumes/IMAGES/bombus_synthetic_image_data/extracted_frames/SMBB"
output_root = "/Volumes/IMAGES/bombus_synthetic_image_data/bee_on_background/B_vos"
scale_factor = 0.1  # Bee will be 10x smaller
max_outputs = 10    # Limit total number of outputs

os.makedirs(output_root, exist_ok=True)

# === Load all bee images ===
bee_images = [
    os.path.join(bee_folder, f)
    for f in os.listdir(bee_folder)
    if f.lower().endswith(".png")
]

output_count = 0

# === Process all backgrounds ===
for dirpath, _, filenames in os.walk(background_root):
    for fname in filenames:
        if output_count >= max_outputs:
            break
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        background_path = os.path.join(dirpath, fname)
        try:
            background = Image.open(background_path).convert("RGBA")
        except Exception as e:
            print(f"❌ Failed to load background {background_path}: {e}")
            continue

        bg_w, bg_h = background.size

        for bee_path in bee_images:
            if output_count >= max_outputs:
                break

            try:
                bee = Image.open(bee_path).convert("RGBA")
                original_w, original_h = bee.size

                # Resize bee
                new_size = (int(original_w * scale_factor), int(original_h * scale_factor))
                if new_size[0] < 1 or new_size[1] < 1:
                    print(f"⚠️ Skipping tiny bee: {bee_path}")
                    continue

                bee_resized = bee.resize(new_size, Image.LANCZOS)

                # Random rotation
                angle = random.uniform(0, 360)
                bee_rotated = bee_resized.rotate(angle, expand=True)

                bee_w, bee_h = bee_rotated.size
                if bee_w >= bg_w or bee_h >= bg_h:
                    print(f"⚠️ Rotated bee too large: {bee_path}")
                    continue

                # Random position
                x = random.randint(0, bg_w - bee_w)
                y = random.randint(0, bg_h - bee_h)

                composite = background.copy()
                composite.paste(bee_rotated, (x, y), bee_rotated)

                # Save output
                bee_name = os.path.splitext(os.path.basename(bee_path))[0]
                bg_name = os.path.splitext(fname)[0]
                out_name = f"{bg_name}__{bee_name}.png"
                out_path = os.path.join(output_root, out_name)
                composite.convert("RGB").save(out_path, "PNG")

                output_count += 1
                print(f"✅ [{output_count}/{max_outputs}] Saved: {out_path}")

            except Exception as e:
                print(f"❌ Error processing {bee_path} on {background_path}: {e}")

    if output_count >= max_outputs:
        break
