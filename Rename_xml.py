from PIL import Image
import os
import shutil

input_folder = "/home/nvidia/Package-Detection-Model/val_xml"
output_folder = "val-xml-renamed"

ext = '.xml'

counter = 1000

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(ext):
        try:
            input_path = os.path.join(input_folder, filename)
            new_filename = f"{counter:03d}{ext}"
            output_path = os.path.join(output_folder, new_filename)

            shutil.copy(input_path, output_path)


           

            print(f"Renamed: {filename} -> {new_filename}")

            counter += 1

        except Exception as e:
            print(f"Error with {filename}: {e}")

print("Done")