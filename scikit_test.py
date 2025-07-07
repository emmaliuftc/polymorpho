
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


file = io.imread("/Users/coding/Downloads/_nuclear_morpho_data/Lamin/Preprocessing_2/002.tif")

viewer = napari.view_image(file, visible=False)

# hole_fill = np.zeros_like(file)
# viewer.add_image(file[5,:,:])

print(file.shape)

shape = file.shape

def counting(file,final_count):
    labels = measure.label(file,connectivity=1)
    # viewer.add_image(file)
    viewer.add_labels(labels)
    lobes = measure.regionprops_table(labels,properties=('area','area_filled',))
    lobe_areas = lobes['area']
    filtered_lobes = lobe_areas[lobe_areas>80000]
    final_count = max(final_count,filtered_lobes.size)
    print(f"number of labels{np.unique(labels).size},number of filtered lobes:{filtered_lobes.size}")
    erosion = morphology.binary_erosion(file,footprint=np.ones(shape=(3,3,3)))
    return erosion,final_count
def fill(x):
    newfile = np.zeros_like(x)
    for i in range(len(x)):
        roi = x[i,:,:]
        new = morphology.remove_small_holes(roi, area_threshold=4000)
        newfile[i,:,:] = new
    newfile = morphology.remove_small_holes(newfile,area_threshold=6000)
    return newfile
def erode(n, filled):
    result = np.zeros_like(filled)
    result = morphology.binary_erosion(filled)
    while(n!=1):
        result = morphology.binary_erosion(result)
        n-=1
    return result
def close(x):
    newfile = np.zeros_like(x)
    for i in range(len(x)):
        roi = x[i,:,:]
        new = morphology.binary_closing(roi,footprint=np.ones(shape=(7,7))) # optional adjust kernel shape, footprint=np.ones(shape=(5,5,5)) doesn't work
        newfile[i,:,:] = new
    return newfile

def count_lobes(file, count): # doesn't take into account anything w/ an area limit
    eroded_file = morphology.binary_erosion(file,footprint=np.ones(shape=(3,12,12))) # change footprint to correspond to resolution?
    labels,labelcount = measure.label(eroded_file,connectivity=1,return_num=True)
    # print("labels {}, labelcount {}".format(labels, labelcount))
    if labelcount>0:
        current_lobe_count=0
        for i in range(1,labelcount+1):
            new = np.where(labels==i,labels,0)
            current_lobe_count+= count_lobes(new,0)
        return count + current_lobe_count
    elif labelcount==0:
        print("count f adding??????")
        viewer.add_image(file,name="blob before final erosion",visible=False)
        return count + 1
    else:
        count_lobes(file,count)
    return count_lobes(eroded_file)
        

# closing = morphology.binary_dilation(file)
# closing = morphology.binary_dilation(closing)

# viewer.add_image(closing)

# closing_labels = measure.label(closing,connectivity=1)
# print(f"original # holes maybe??????: {np.unique(closing_labels).size-1}")
# viewer.add_labels(closing_labels,name="closing labels??")
# closing_table = measure.regionprops_table(closing_labels,properties=('euler_number',))
# closing_data = closing_table['euler_number']
# print(f"euler number????????:{closing_data}")
# filtered_hole_areas = hole_areas[hole_areas>200]
# print(f"final hole count: {filtered_hole_areas.size}")


hole_fill = fill(file)
viewer.add_image(hole_fill)
arr = morphology.remove_small_objects(hole_fill,min_size=500,connectivity=1)
count = -1
total_lobe_count = count_lobes(arr,0)
print(total_lobe_count)

napari.run()

# beginning of image processing section
'''

viewer.add_image(hole_fill,name="hole fill after dilation before erosion")
viewer.add_image(file,opacity=0.5)



skeletonize = morphology.skeletonize(hole_fill, method="lee")
viewer.add_image(skeletonize)

skeleton = np.zeros((shape[1],shape[2]))
for i in range(shape[0]-1):
    skeleton = np.logical_or(skeleton,skeletonize[i,:,:])

viewer.add_image(skeleton)
binary_skeleton = morphology.binary_dilation(skeleton)
viewer.add_image(binary_skeleton)
holes_please = measure.label(binary_skeleton,connectivity=1)
holes_maybe = np.unique(holes_please).size-1
print(f"original # holes maybe??????: {holes_maybe}")
viewer.add_labels(holes_please,name="hole labels??")
holes = measure.regionprops_table(holes_please,properties=('area','euler_number','area_filled',))
hole_areas = holes['area']
hole_euler = holes['euler_number']
hole_area_filled = holes['area_filled']
print(hole_areas)
print(f"original hole area filled:{hole_area_filled}")
print(f"euler characteristic?????: {hole_euler}")
diff = np.subtract(hole_area_filled,hole_areas)
print(diff)
for element in diff:
    if element>8000:
        print("hole found??")

# filtered_hole_area_filled = hole_area_filled[hole_area_filled>5000]
# print(f"filtered hole area filled:{filtered_hole_area_filled}")
# # euler number doesn't make sense
# print(f"final hole count: {filtered_hole_area_filled.size}")

closed = morphology.binary_closing(hole_fill,np.ones(shape=(7,7,7)))
viewer.add_image(closed,name="CLOSED IMAGE")
closed = fill(closed)
viewer.add_image(closed,name="FILLED CLOSED IMAGE")
closed_slice = close(hole_fill)
viewer.add_image(closed_slice,name="closed by slice")
closed_slice = fill(closed_slice)
viewer.add_image(closed_slice,name="filled closed by slice")

count = -1
arr = closed
for i in range(10):
    arr,count = counting(arr,count)
print(f"FINAL LOBE COUNT?!?!?!??!!?:{count}")

gaussianblur = filters.median(closed)
viewer.add_image(gaussianblur,name="median blur for closed")
'''


'''
erosion = erode(6,hole_fill)
viewer.add_image(erosion,colormap="bop blue")


labels = measure.label(erosion)
viewer.add_labels(labels,name="labels")

regions = measure.regionprops_table(labels,properties=('area',))
# print(regions)
# for region in regions:
#     print(f"area: {region.area},")
print(f"original # labels:{np.unique(labels).size-1}")

areas_array = regions['area']
print(areas_array)
filtered_area = areas_array[areas_array>10000]
print(filtered_area)
print(f"final lobe count????: {filtered_area.size}")
'''


# napari.run()


#end of image processing section










# step = 3
# verts,faces,normals,values = measure.marching_cubes(hole_fill,step_size=step)

# print(verts)

# zsum = 0
# xsum = 0
# ysum = 0
# for i in range(len(verts)-1):
#     zsum += verts[i][0]
# for i in range(len(verts)-1):
#     xsum += verts[i][1]
# for i in range(len(verts)-1):
#     ysum += verts[i][2]
# print(zsum,xsum,ysum)
# centroid = [zsum/len(verts),xsum/len(verts),ysum/len(verts)]
# print(centroid)
# # centroid = [z,x,y]
# distance = np.zeros((3,len(verts)))
# # 0: distance
# # 1: phi
# # 2: theta
# def c2s(x,y,z,dist):
#     phi = np.arctan2(y,x)
#     theta = np.arccos(z/dist)
#     return theta,phi

# for i in range(len(verts)-1):
#     distance[0][i] = np.linalg.norm(verts[i]-centroid)
#     theta, phi = c2s(verts[i][1]-centroid[1],verts[i][2]-centroid[2],verts[i][0]-centroid[0],distance[0][i])
#     distance[1][i] = phi
#     distance[2][i] = theta

# print(f"a distance:{distance}")

# # bp = plt.boxplot(distance, showfliers=True)
# '''
# for i in range(len(distance)-1):
#     y = distance[i]
#     # Add some random "jitter" to the x-axis
#     x = 1 + np.random.randn()
#     plt.plot(x, y, 'r.', alpha=0.2)
# '''

# # plt.hist(distance[0], bins=40)


# mask = np.argsort(distance[1])
# distance = distance[:,mask]

# distance[1,:] = np.round(distance[1,:],decimals=2)
# print(f"rounded distance: {distance}")

# max_dist = {}
# max_theta = max(distance[1])
# min_theta = min(distance[1])

# for i in range(distance.shape[1]):
#     row1_value = distance[0, i]
#     row2_value = distance[1, i]

#     if row2_value in max_dist:
#         max_dist[row2_value] = max(max_dist[row2_value], row1_value)
#     else:
#         max_dist[row2_value] = row1_value

# result = list(max_dist.values())

# # print(result)

# # for i in range(-314,314,1):
# #     temp = []
# #     for j in range(len(distance)-1):
# #         print(f"distance 1j:{distance[1][j]}")
# #         if distance[1][j]==float(i/100):
# #             temp.append(distance[0][j])
# #     print(f"temp:{temp}")
# #     max_dist.append(max(temp)) 

# # print(max_dist)

# uniquetheta = np.unique(distance[1])[-100:]
# uniquetheta = np.append(uniquetheta,np.unique(distance[1]))
# uniquetheta = np.append(uniquetheta,np.unique(distance[1])[:100])

# newresult = result[-100:]
# newresult = np.append(newresult,result)
# newresult = np.append(newresult,result[:100])


# #plt.scatter(result,np.unique(distance[1]),alpha=0.2)
# plt.scatter(newresult,uniquetheta,alpha=0.2,color="red")
# plt.xlabel("Distance to centroid")
# plt.ylabel("theta in radians")

# peaks,properties = signal.find_peaks(result,width=10)
# print(peaks)
# print(len(result))

# newpeaks,newproperties = signal.find_peaks(newresult,width=10)
# print(newpeaks)
# newpeaks = newpeaks-100
# newpeaks = newpeaks%len(result)
# print(f"lobe count: {len(np.unique(newpeaks))}")

# print(properties)
# print(newproperties)











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


# plt.tight_layout()
# plt.show()



# DISPLAY MESH
# import napari
# from skimage import io
# from skimage import measure 
# import os

# file = io.imread("/Users/coding/Downloads/_nuclear_morpho_data/Lamin/_Out/Test_cropped3_1_substack__Out.tif")

# verts,faces,normals,values = measure.marching_cubes(file,step_size=3)

# surface = (verts, faces, values)

# viewer = napari.view_surface(surface)





