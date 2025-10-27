import napari
import trimesh

# --- Replace this with the path to your file ---
file_path = '~/_nuclear_morpho_data/Lamin/Preprocessing_3_mesh/018.obj'

# 1. Load the mesh file using trimesh
# This reads the vertices and faces from the .obj file.
mesh = trimesh.load(file_path)

# 2. Create a napari viewer instance
viewer = napari.Viewer(ndisplay=3) # ndisplay=3 ensures it opens in 3D mode

# 3. Add the mesh as a Surface layer
# A napari surface layer takes a tuple: (vertices, faces)
viewer.add_surface((mesh.vertices, mesh.faces), name='My Mesh')

# 4. Run the napari viewer
napari.run()