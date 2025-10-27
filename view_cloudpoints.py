import napari
import numpy as np

# --- Replace this with the path to your file ---

file_path = '/home/coding/_nuclear_morpho_data/Lamin/Preprocessing_3_sdf/001.npy'

# 1. Load the .npy file into a NumPy array
# This assumes the file contains an array of shape (N, 3) for N points.
points = np.load(file_path)
print(f"Loaded point cloud with shape: {points.shape}")

# 2. Create a napari viewer instance
# ndisplay=3 is crucial to start in 3D viewing mode.
viewer = napari.Viewer(ndisplay=3)

# 3. Add the points as a Points layer
# We can customize the size and color of the points.
viewer.add_points(
    points,
    name='My Point Cloud',
    size=3,  # Adjust point size for better visibility
    face_color='cyan'
)

# 4. Run the napari GUI
napari.run()