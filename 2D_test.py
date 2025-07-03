import napari
from scipy import stats
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
import os
from pathlib import Path

file = io.imread("/Users/coding/Downloads/_nuclear_morpho_data/Geimsa/test_2d/test3_4-1.tif")

viewer = napari.view_image(file,contrast_limits=(0,1))

def count_lobes(file, count): # doesn't take into account anything w/ an area limit
    eroded_file = morphology.binary_erosion(file,footprint=np.ones(shape=(17,17)))
    labels,labelcount = measure.label(eroded_file,connectivity=1,return_num=True)
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
        




def counting(file,final_count):
    labels,labelcount = measure.label(file,connectivity=1,return_num=True)
    # viewer.add_image(file)
    # print(f"number of labels: {labelcount}")
    #viewer.add_labels(labels)
    lobes = measure.regionprops_table(labels,properties=('area','area_filled',))
    lobe_areas = lobes['area']
    filtered_lobes = lobe_areas[lobe_areas>500]
    final_count = max(final_count,filtered_lobes.size)
    #print(f"number of labels: {np.unique(labels).size}, number of filtered lobes: {filtered_lobes.size}")
    erosion = morphology.binary_erosion(file,footprint=np.ones(shape=(3,3)))
    if filtered_lobes.size==0:
        flag=True
        return erosion, final_count,flag
    else: flag = False
    for i in range(labelcount):
        new = np.where(labels==i,labels,0)
        bylabel = morphology.binary_erosion(new,footprint=np.ones(shape=(10,10)))
        #viewer.add_image(bylabel)
    return erosion,final_count,flag

directory_path = Path('/Users/coding/Downloads/_nuclear_morpho_data/Geimsa/test_2d')
all_files = list(directory_path.glob('*'))
print(len(all_files))
for path in all_files:
    ogfile = io.imread(path)
    print(path) 
    viewer.add_image(ogfile,contrast_limits=(0,1),visible=False)
    labeled_ogfile = measure.label(ogfile)
    file = morphology.remove_small_objects(labeled_ogfile,min_size=1500,connectivity=1)
    viewer.add_image(file,contrast_limits=(0,1),visible=False)

    print(np.count_nonzero(file))
    count = -1
    arr = file
    while(True):
        arr,count,flag = counting(arr,count)
        if flag: break
    # print(f"FINAL LOBE COUNT of {path}:{count}")
    arr = file
    total_lobe_count = count_lobes(arr,0)
    # print(f"fraud total lobe count: {total_lobe_count}")
    print(f"manual lobe count: XXX, pass entire: {count}, recursive: {total_lobe_count}")



# arr2 = morphology.binary_opening(arr1)
# viewer.add_image(arr2)
# arr3 = morphology.binary_opening(arr2)
# viewer.add_image(arr3)
# arr4 = morphology.binary_opening(arr3)
# viewer.add_image(arr4)


napari.run()

# 