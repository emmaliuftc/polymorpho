
import trimesh
import numpy as np
import os
import glob

def process_mesh_to_sdf_point_cloud(mesh_path, num_points=4096):
    """
    Loads a mesh, samples points, calculates SDF, and returns a point cloud
    with SDF values.
    """
    # 1. Load the mesh
    mesh = trimesh.load(mesh_path)

    # 2. Sample points on the surface
    points, _ = trimesh.sample.sample_surface(mesh, num_points)

    # 3. Calculate SDF
    # The trimesh.proximity.signed_distance function returns the signed distance
    # from the given points to the mesh surface.
    sdf = trimesh.proximity.signed_distance(mesh, points)

    # Combine points and SDF values
    # The result is a (num_points, 4) array where the last column is the SDF
    sdf_point_cloud = np.hstack((points, sdf[:, np.newaxis]))

    return sdf_point_cloud

# --- Configuration ---
input_dir = "/home/coding/_nuclear_morpho_data/Lamin/Preprocessing_3_mesh"
output_dir = "/home/coding/_nuclear_morpho_data/Lamin/Preprocessing_3_sdf"
num_points_to_sample = 4096
# ---

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all .obj files in the input directory
obj_files = glob.glob(os.path.join(input_dir, '*.obj'))

if not obj_files:
    print(f"No .obj files found in {input_dir}")
else:
    print(f"Found {len(obj_files)} .obj files to process.")

for file_path in obj_files:
    print(f"Processing {os.path.basename(file_path)}...")
    try:
        # Process the mesh
        sdf_point_cloud = process_mesh_to_sdf_point_cloud(
            file_path,
            num_points=num_points_to_sample
        )

        # Save the SDF point cloud
        base_filename = os.path.basename(file_path)
        output_filename = os.path.splitext(base_filename)[0] + '.npy'
        output_path = os.path.join(output_dir, output_filename)
        
        np.save(output_path, sdf_point_cloud)
        print(f"  -> SDF point cloud saved to {output_path}")

    except Exception as e:
        print(f"  -> Error processing {os.path.basename(file_path)}: {e}")

print("\nProcessing complete.")
