import os
import napari
from scipy import stats, signal
# from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib import pyplot as plt, colors, cm
# from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                     morphology, restoration, segmentation, transform,
                     util)
from pathlib import Path
import math
# import pywt
from sklearn import svm,preprocessing,cluster,decomposition,metrics,manifold,model_selection
import mahotas
# import seaborn as sns
# import umap

def fill(x):
    newfile = np.zeros_like(x)
    for i in range(len(x)):
        roi = x[i,:,:]
        new = morphology.remove_small_holes(roi, area_threshold=4000)
        newfile[i,:,:] = new
    newfile = morphology.remove_small_holes(newfile,area_threshold=6000)
    return newfile

file = io.imread("/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Test/Test_001_4_Out.tif")

viewer = napari.view_image(file,visible=False)

# print(file.shape)
shape = file.shape

counted_lobes = []


directory_path = Path('/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Preprocessing_3')
dir = [f.path for f in os.scandir(directory_path) if f.is_file()]
all_files = []
for path in dir:
    all_files = np.append(all_files,str(path))

all_files = sorted(all_files)
filenames= []
signals = []

for i in range(len(all_files)):
    fileindex = str(all_files[i][-6]+all_files[i][-5])
    filenames.append(fileindex)
    ogfile = io.imread(all_files[i])
    print(all_files[i])
    hole_fill = fill(ogfile)

    step = 3
    verts,faces,normals,values = measure.marching_cubes(hole_fill,step_size=step)

    # print(verts)

    zsum = 0
    xsum = 0
    ysum = 0
    for i in range(len(verts)-1):
        zsum += verts[i][0]
    for i in range(len(verts)-1):
        xsum += verts[i][1]
    for i in range(len(verts)-1):
        ysum += verts[i][2]
    # print(zsum,xsum,ysum)
    centroid = [zsum/len(verts),xsum/len(verts),ysum/len(verts)]
    # print(centroid)
    # centroid = [z,x,y]
    distance = np.zeros((3,len(verts)))
    # 0: distance
    # 1: phi
    # 2: theta
    def c2s(x,y,z,dist):
        phi = np.arctan2(y,x)
        theta = np.arccos(z/dist)
        return theta,phi

    for i in range(len(verts)-1):
        distance[0][i] = np.linalg.norm(verts[i]-centroid)
        theta, phi = c2s(verts[i][1]-centroid[1],verts[i][2]-centroid[2],verts[i][0]-centroid[0],distance[0][i])
        distance[1][i] = phi
        distance[2][i] = theta

    # print(f"a distance:{distance}")

    # bp = plt.boxplot(distance, showfliers=True)
    '''
    for i in range(len(distance)-1):
        y = distance[i]
        # Add some random "jitter" to the x-axis
        x = 1 + np.random.randn()
        plt.plot(x, y, 'r.', alpha=0.2)
    '''

    # plt.hist(distance[0], bins=40)


    mask = np.argsort(distance[1])
    distance = distance[:,mask]

    distance[1,:] = np.round(distance[1,:],decimals=2)
    # print(f"rounded distance: {distance}")

    max_dist = {}
    max_theta = max(distance[1])
    min_theta = min(distance[1])

    for i in range(distance.shape[1]):
        row1_value = distance[0, i]
        row2_value = distance[1, i]

        if row2_value in max_dist:
            max_dist[row2_value] = max(max_dist[row2_value], row1_value)
        else:
            max_dist[row2_value] = row1_value

    result = list(max_dist.values()) # y-values
    xvalues = range(len(result))

    x=np.empty(shape=650)
    for i in range(650):
        x[i] = i*(len(result)/650)

    final = np.interp(x=x,xp=xvalues,fp=result)
    # print(final)
    signals.append(final)
    print(len(final))

    uniquetheta = np.unique(distance[1])[-100:]
    uniquetheta = np.append(uniquetheta,np.unique(distance[1]))
    uniquetheta = np.append(uniquetheta,np.unique(distance[1])[:100])

    newresult = result[-100:]
    newresult = np.append(newresult,result)
    newresult = np.append(newresult,result[:100])

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.scatter(result,range(len(result)),alpha=0.2)
    ax1.scatter(final,x,alpha=0.2,color="red")
    plt.xlabel("Distance to centroid")
    plt.ylabel("theta in radians")
    # plt.show()
    plt.close()


    widthmin = 25
    prominencemin = 15
    heightmin = 100

    peaks,properties = signal.find_peaks(result,width=widthmin,prominence=prominencemin,height=heightmin)
    # print(peaks)
    # print(len(result))

    newpeaks,newproperties = signal.find_peaks(newresult,width=widthmin,prominence=prominencemin,height=heightmin)
    # print(newpeaks)
    newpeaks = newpeaks-100
    newpeaks = newpeaks%len(result)
    if len(np.unique(newpeaks))==0:
        print(f"lobe count: {len(np.unique(newpeaks))+1}")
        counted_lobes.append(len(np.unique(newpeaks))+1)
    else: 
        print(f"lobe count: {len(np.unique(newpeaks))}")
        counted_lobes.append(len(np.unique(newpeaks)))

    # print(properties)
    # print(newproperties)
    # print(properties["prominences"].max())
    # plt.plot(uniquetheta[newpeaks],newresult[newpeaks],'x')

    # plt.scatter(distance[0],distance[2],alpha=0.2)
    # print(f"v:{verts},f:{faces},n:{normals},v:{values}")

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).


    '''
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces],alpha=0.2)
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.plot(centroid[0],centroid[1],centroid[2],'o',color='red')

    ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    ax.set_ylabel("y-axis: b = 10")
    ax.set_zlabel("z-axis: c = 16")

    mins = np.min(verts,axis=0)
    maxs = np.max(verts,axis=0)
    print(mins)
    print(maxs)

    ax.set_xlim(mins[0]-20, maxs[0]+20)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(mins[1]-40, maxs[1]+40)  # b = 10
    ax.set_zlim(mins[2]-40, maxs[2]+40)  # c = 16
    '''




    '''hull = ConvexHull(verts)

    ax = fig.add_subplot(111, projection='3d')
    # Plot defining corner points
    for i in range(hull.vertices.size):
        print(verts[hull.vertices[i]])
        ax.plot(verts[hull.vertices[i]][0],verts[hull.vertices[i]][1],verts[hull.vertices[i]][2],'o')
    '''

    # DISPLAY MESH
    # import napari
    # from skimage import io
    # from skimage import measure 
    # import os

    # file = io.imread("/Users/coding/Downloads/_nuclear_morpho_data/Lamin/_Out/Test_cropped3_1_substack__Out.tif")

    # verts,faces,normals,values = measure.marching_cubes(file,step_size=3)

    # surface = (verts, faces, values)

    # viewer = napari.view_surface(surface)

real_counts = [2,2,4,4,4,4,2,1,3,4,2,1,4,1,3,4,3,3,3,3,3,3,2,3,2,2,3,4,2,4,3,1,3,2,3,3,2,1,2,3,2,1,3,3,3]

x_train = np.array(signals)
print(x_train.shape)

norm = preprocessing.Normalizer().fit(x_train)
x_train_norm = norm.transform(x_train)

clf = svm.SVC(kernel="linear")
scores = model_selection.cross_val_score(clf,x_train_norm,real_counts,cv=5)
print(scores)

print("lobe counts %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

clf = svm.SVC(kernel="linear")

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_train_norm,real_counts,test_size=0.3,random_state=1)

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
print(y_pred)

print(metrics.accuracy_score(y_test,y_pred))

print(metrics.accuracy_score(real_counts,counted_lobes))