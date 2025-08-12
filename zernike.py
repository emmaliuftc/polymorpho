import os
import napari
# from scipy import stats, signal
# from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib import pyplot as plt, colors, cm
# from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                     morphology, restoration, segmentation, transform,
                     util)
from pathlib import Path
import math
# import pywt
from sklearn import svm,preprocessing,cluster,decomposition,metrics,manifold,model_selection,feature_selection
import mahotas
# import seaborn as sns
# import umap



def fill(x):
    newfile = np.zeros_like(x)
    for i in range(len(x)):
        roi = x[i,:,:]
        new = morphology.remove_small_holes(roi, area_threshold=4000)
        newfile[i,:,:] = new
    newfile = morphology.remove_small_holes(newfile,area_threshold=100000)
    return newfile

file = io.imread("/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Preprocessing_3/033.tif")

viewer = napari.view_image(file,visible=False)


directory_path = Path('/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Preprocessing_3')
dir = [f.path for f in os.scandir(directory_path) if f.is_file()]
all_files = []
for path in dir:
    all_files = np.append(all_files,str(path))

all_files = sorted(all_files)

data_size = len(all_files)


for i in range(len(all_files)):
    ogfile = io.imread(all_files[i])
    print(ogfile.shape)
    hole_fill = fill(ogfile)

    # viewer.add_image(hole_fill[20,:,:])
    com = mahotas.center_of_mass(hole_fill)
    zernike = []
    zernike.append(mahotas.features.zernike_moments(hole_fill[round(com[0]),:,:],200,8)) # 1D array of length 25 
    zernike.append(mahotas.features.zernike_moments(hole_fill[round(com[0]-10),:,:],200,8)) # 1D array of length 25 
    zernike.append(mahotas.features.zernike_moments(hole_fill[round(com[0]+10),:,:],200,8)) # 1D array of length 25 
    print(zernike)
    scaler = preprocessing.StandardScaler()
    zernike_scaled = scaler.fit_transform(zernike)

    pca = decomposition.PCA(n_components=1,whiten=True)
    zernike_pca = pca.fit_transform(zernike_scaled)










napari.run()
