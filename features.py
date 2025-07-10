import os
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
from sklearn import svm,preprocessing,cluster,decomposition

def fill(x):
    newfile = np.zeros_like(x)
    for i in range(len(x)):
        roi = x[i,:,:]
        new = morphology.remove_small_holes(roi, area_threshold=4000)
        newfile[i,:,:] = new
    newfile = morphology.remove_small_holes(newfile,area_threshold=6000)
    return newfile

file = io.imread("/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Preprocessing_3/033.tif")

viewer = napari.view_image(file,visible=False)


directory_path = Path('/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Preprocessing_3')
dir = [f.path for f in os.scandir(directory_path) if f.is_file()]
all_files = []
for path in dir:
    all_files = np.append(all_files,str(path))

all_files = sorted(all_files)
# print(all_files)

# shape = (number of images,number of features)
# (20, 3)
# volume, surface area, sphericity

data_size = len(all_files)

# x_train = np.empty(shape=(10,4))
x_train = np.empty(shape=(data_size,4))

# images = np.array((1,1))
volumes = []
surfaces = []
sphericities = []

for i in range(len(all_files)):
    ogfile = io.imread(all_files[i])
    print(all_files[i])
    hole_fill = fill(ogfile)
    viewer.add_image(hole_fill,name=f"{all_files[i]}",visible=False)
    # images = np.append(images,hole_fill)

    labels = measure.label(hole_fill)
    properties = measure.regionprops_table(labels,properties=("area",))
    volume = np.count_nonzero(hole_fill)

    print(f"volume: {volume}")
    volumes.append(volume)

    print(properties)

    verts,faces,normals,values = measure.marching_cubes(hole_fill,step_size=3)

    surface_area = measure.mesh_surface_area(verts,faces)

    print(f"surface area: {surface_area}")
    surfaces.append(surface_area)

    sphericity = pow(math.pi, 0.3333)*pow(6*volume,0.666666)/surface_area

    print(sphericity)
    sphericities.append(sphericity)
'''
    for i in range(10):
        gauss = filters.gaussian(hole_fill,sigma = i)
        viewer.add_image(gauss,name = f"sigma = {i}")
        boundary = segmentation.find_boundaries(gauss)
        viewer.add_image(boundary)
'''


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

# real_counts = np.array([2,3,4,2,4,2,1,3,3,3])
real_counts = [2,3,4,2,4,3,1,3,2,3,3,2,1,2,3,2,1,3,3,3]


npvolumes = np.array(volumes)
npsurfaces = np.array(surfaces)
npsphericities = np.array(sphericities)
print(volumes)

# x_train[:,0] = np.reshape(volumes,newshape=(10,1))[:,0]
# x_train[:,1] = np.reshape(surfaces,newshape=(10,1))[:,0]
# x_train[:,2] = np.reshape(sphericities,newshape=(10,1))[:,0]
# x_train[:,3] = np.reshape(real_counts,newshape=(10,1))[:,0]
x_train[:,0] = np.reshape(volumes,newshape=(data_size,1))[:,0]
x_train[:,1] = np.reshape(surfaces,newshape=(data_size,1))[:,0]
x_train[:,2] = np.reshape(sphericities,newshape=(data_size,1))[:,0]
x_train[:,3] = np.reshape(real_counts,newshape=(data_size,1))[:,0]


# print(x_train)

x_train_norm = preprocessing.normalize(x_train)

scaler = preprocessing.StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
# need x_test eventually
# scaler.transform(x_test)

pca = decomposition.PCA(n_components=2)
x_pca = pca.fit_transform(x_train_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative explained variance:", pca.explained_variance_ratio_.sum())

kmeans = cluster.KMeans(n_clusters=5,random_state=0,n_init="auto")

kmeans.fit(x_pca)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], color='red', marker='x')
plt.xlabel('pca1')
plt.ylabel('pca2')
for i, (x, y_val) in enumerate(x_pca):
    plt.annotate(str(i+32), (x, y_val), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
plt.show()

# napari.run()