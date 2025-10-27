
import os
import tifffile
import numpy as np
from skimage.measure import marching_cubes
import glob

def save_obj(verts, faces, output_path):
    """
    Saves a mesh to an .obj file.
    """
    with open(output_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

# --- Configuration ---
input_dir = "/home/coding/_nuclear_morpho_data/Lamin/Preprocessing_3"
output_dir = "/home/coding/_nuclear_morpho_data/Lamin/Preprocessing_3_mesh"
# ---

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all .tif files in the input directory
tif_files = glob.glob(os.path.join(input_dir, '*.tif'))

if not tif_files:
    print(f"No .tif files found in {input_dir}")
else:
    print(f"Found {len(tif_files)} .tif files to process.")

for file_path in tif_files:
    print(f"Processing {os.path.basename(file_path)}...")
    try:
        # 1. Load the .tif image stack
        image_stack = tifffile.imread(file_path)

        # 2. Segmentation (simple thresholding)
        # This threshold may need to be adjusted for your specific images.
        threshold = 0.5
        binary_mask = image_stack > threshold

        # 3. Surface Extraction (Marching Cubes)
        # The `level` parameter is the iso-value. For a binary mask, 0.5 is good.
        verts, faces, _, _ = marching_cubes(binary_mask, level=0.5)

        # 4. Save the Mesh
        base_filename = os.path.basename(file_path)
        output_filename = os.path.splitext(base_filename)[0] + '.obj'
        output_path = os.path.join(output_dir, output_filename)
        
        save_obj(verts, faces, output_path)
        print(f"  -> Mesh saved to {output_path}")

    except Exception as e:
        print(f"  -> Error processing {os.path.basename(file_path)}: {e}")

print("\nProcessing complete.")
