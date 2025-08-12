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
        new = morphology.remove_small_holes(roi, area_threshold=1000)
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

# directory2_path = Path('/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Preprocessing')
# dir2 = [f.path for f in os.scandir(directory2_path) if f.is_file()]
# all_files2 = []
# for path in dir2:
#     all_files2 = np.append(all_files2,str(path))

# all_files2 = sorted(all_files2)

# print(all_files)

# shape = (number of images,number of features)
data_size = len(all_files)
feature_num = 41
x_data = np.empty(shape=(data_size,feature_num))

# images = np.array((1,1))
volumes = []
surfaces = []
sphericities = []
haralicks = []
zernikes = []
filenames = []
skeletons = []

for i in range(len(all_files)):
    fileindex = str(all_files[i][-6]+all_files[i][-5])
    filenames.append(fileindex)
    ogfile = io.imread(all_files[i])
    print(ogfile.shape)
    print(all_files[i])
    hole_fill = fill(ogfile)
    viewer.add_image(hole_fill,name=f"{all_files[i]}",visible=False)
    # images = np.append(images,hole_fill)

    labels = measure.label(hole_fill)
    # properties = measure.regionprops_table(labels,properties=("area","axis_major_length","axis_minor_length"))
    volume = np.count_nonzero(hole_fill)
    # maj = properties["axis_major_length"]
    # mino = properties["axis_minor_length"]
    # print(f"volume: {volume}")
    volumes.append(volume)

    slice = hole_fill[10,:,:]


    haralick = mahotas.features.haralick(hole_fill,return_mean=True)
    haralicks.append(haralick)
    print(haralick.shape)
    # print(f"binary iamge: {features.haralick(hole_fill,return_mean=True)}")
    
    # viewer.add_image(features.haralick(hole_fill),name=f"haralick {all_files[i]}",visible=False)
    # file = io.imread(all_files2[i])
    # print(all_files2[i])
    # haralick = features.haralick(file,return_mean=True)
    # haralicks.append(haralick)
    # print(f"binary iamge: {haralicks[i]}")

    verts,faces,normals,values = measure.marching_cubes(hole_fill,step_size=3)

    surface_area = measure.mesh_surface_area(verts,faces)

    # print(f"surface area: {surface_area}")
    surfaces.append(surface_area)

    sphericity = pow(math.pi, 0.3333)*pow(6*volume,0.666666)/surface_area

    # print(sphericity)
    sphericities.append(sphericity)

    com = mahotas.center_of_mass(hole_fill)
    zernike = np.empty(shape=(1,25))
    zernike[0,:] = mahotas.features.zernike_moments(hole_fill[round(com[0]),:,:],200,8).flatten() # 1D array of length 25 
    # zernike[1,:] = mahotas.features.zernike_moments(hole_fill[round(com[0]-10),:,:],200,8).flatten() # 1D array of length 25 
    # zernike[2,:] = mahotas.features.zernike_moments(hole_fill[round(com[0]+10),:,:],200,8).flatten() # 1D array of length 25 
    zernike = zernike.flatten()
    print(zernike.shape)
    zernikes.append(zernike)
    # print(zernike)
    # scaler = preprocessing.StandardScaler()
    # zernike_scaled = scaler.fit_transform(zernike)

    # pca = decomposition.PCA(n_components=1,whiten=True)
    # zernike_pca = pca.fit_transform(zernike_scaled)
    # zernike1.append(zernike_pca[0])
    # zernike2.append(zernike_pca[1])
    # zernike3.append(zernike_pca[2])


# real_counts = np.array([2,3,4,2,4,2,1,3,3,3])
real_counts = [2,2,4,4,4,4,2,1,3,4,2,1,4,1,3,4,3,3,3,3,3,3,2,3,2,2,3,4,2,4,3,1,3,2,3,3,2,1,2,3,2,1,3,3,3]
real_hole_counts = [0,0,0,1,1,0,0,0,1,0,1,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0,1,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,1,1]
hole_counts =      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]

print(sorted(real_counts))
npvolumes = np.array(volumes)
npsurfaces = np.array(surfaces)
npsphericities = np.array(sphericities)
npharalicks = np.array(haralicks)
npzernikes = np.array(zernikes)

print(npharalicks.shape)
# print(npvolumes.shape)
print(npzernikes.shape)

# # x_train[:,0] = np.reshape(volumes,newshape=(data_size,1))[:,0]
# # x_train[:,0] = np.reshape(surfaces,newshape=(data_size,1))[:,0]
# x_train[:,0] = np.reshape(sphericities,newshape=(data_size,1))[:,0]
# x_train[:,1] = np.reshape(real_counts,newshape=(data_size,1))[:,0]
# x_train[:,2:15] = npharalicks

# ALL WITHOUT COUNTS

# x_data[:,0] = np.reshape(volumes,newshape=(data_size,1))[:,0]
# x_data[:,1] = np.reshape(surfaces,newshape=(data_size,1))[:,0]
# x_data[:,2] = np.reshape(sphericities,newshape=(data_size,1))[:,0]
# x_data[:,3:17] = npharalicks

# VOLUME + SPHERICITY

x_data[:,0] = np.reshape(volumes,newshape=(data_size,1))[:,0]
x_data[:,1] = np.reshape(sphericities,newshape=(data_size,1))[:,0]
x_data[:,2:15] = npharalicks
x_data[:,16:] = npzernikes

# SURFACE AREA + SPHERICITY

# x_data[:,0] = np.reshape(surfaces,newshape=(data_size,1))[:,0]
# x_data[:,1] = np.reshape(sphericities,newshape=(data_size,1))[:,0]
# x_data[:,2:] = npharalicks

scaler = preprocessing.StandardScaler()
x_data_scaled = scaler.fit_transform(x_data)

# x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data_scaled,real_counts,test_size=0.3,random_state=1)

clf = svm.SVC(kernel="linear")
scores = model_selection.cross_val_score(clf,x_data_scaled,real_counts,cv=5)
print(scores)

print("lobe counts %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

clf = svm.SVC(kernel="linear")
scores = model_selection.cross_val_score(clf,x_data_scaled,hole_counts,cv=5)
print(scores)

print("hole counts %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

clf = svm.SVC(kernel="linear")
scores = model_selection.cross_val_score(clf,x_data_scaled,real_counts,cv=6)
print(scores)

print("real hole counts %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# clf.fit(x_train,y_train)

# y_pred = clf.predict(x_test)

# print(metrics.accuracy_score(y_test,y_pred))

# selector = feature_selection.RFE(clf,n_features_to_select=1,step=1)
# selector.fit(x_data_scaled,real_counts)
# print(selector.ranking_)


napari.run()


