import napari
from scipy import stats, signal
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                     morphology, restoration, segmentation, transform,
                     util)
from pathlib import Path
import math
import pywt

def fill(x):
    newfile = np.zeros_like(x)
    for i in range(len(x)):
        roi = x[i,:,:]
        new = morphology.remove_small_holes(roi, area_threshold=4000)
        newfile[i,:,:] = new
    newfile = morphology.remove_small_holes(newfile,area_threshold=6000)
    return newfile

file = io.imread("/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Preprocessing_3/033.tif")

viewer = napari.view_image(file,visible=True)

hole_fill = fill(file)

labels = measure.label(hole_fill)
data = measure.regionprops_table(labels,properties=("area",))
volume = np.count_nonzero(hole_fill)

print(f"volume: {volume}")

print(data)

verts,faces,normals,values = measure.marching_cubes(hole_fill,step_size=3)

surface_area = measure.mesh_surface_area(verts,faces)

print(f"surface area: {surface_area}")

sphericity = pow(math.pi, 0.3333)*pow(6*volume,0.666666)/surface_area

print(sphericity)

for i in range(10):
    gauss = filters.gaussian(hole_fill,sigma = i)
    viewer.add_image(gauss,name = f"sigma = {i}")
    boundary = segmentation.find_boundaries(gauss)
    viewer.add_image(boundary)
    

napari.run()

# wp = pywt.WaveletPacketND(hole_fill,wavelet='db2')
# # x=np.array([1,2,3,4,5,6,7,8])
# # wp = pywt.WaveletPacketND(data=x,wavelet='db1')
# # print(wp)
# # print(np.count_nonzero(wp))

# # print(wp.maxlevel)
# # print(wp['aaa'].data)


# cA, cD = pywt.dwt(hole_fill,'db2')
# print(cA)
# print(cD)
# viewer.add_image(cA)
# viewer.add_image(cD)