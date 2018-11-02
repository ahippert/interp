#from mpl_toolkits.basemap import Basemap, cm
import numpy as np
import matplotlib.pyplot as plt
import string
#import sys


# def main():
#     #filename = 'All_GlaciersXYZ_RasterThick_Shifted_LatLong_data.grd'
#     filename = 'Thickness_Zbed_Simu_Argentiere_Lat-long_data.txt'
#     cl_start = 1
#     cl_end = 3
#     basemap = True
#     all_tab = True
#     col = 1122
#     lig = 1211
#     latcorners = [45.630282, 46.0312433]
#     loncorners = [6.7409295, 7.11242326]
#     data = read_grid(filename, cl_start, cl_end, all_tab)
#     plot_grid(data, lig, col, basemap)
    
# ***********************************************
# Read a text file and fills a list with its data 
# 
def read_grid(filename, cl_start, cl_end, all_tab, typ):
    #start = 0
    x = []
    with open(filename) as f_in:
        print('Read file: OK')
        for line in f_in:
            if all_tab:
                for num_str in line.split()[cl_start:cl_end+1]:
                    x.append(float(string.replace(num_str, ',', '.')))
            else:
                for num_str in line.split()[cl_start:cl_end+1]:
                    if typ:
                        x.append(float(num_str))
                    else:
                        x.append(int(num_str))
    print('Fill list with data: OK')
    return x

# ***********************************************


# ***********************************************
# Plot grid read by read_grid()
#
# def plot_grid(data, ligne, colonne, basemap):

#     plt.figure()

#     if basemap:
#         lons = data[0::3]
#         lats = data[1::3]
#         data = data[2::3]
#         m = Basemap(projection='stere',lon_0=0.,lat_0=90.,\
#                     llcrnrlat=np.min(lats)-0.001,urcrnrlat=np.max(lats)+0.001,\
#                     llcrnrlon=np.max(lons)-0.095,urcrnrlon=np.max(lons)+0.015,\
#                     rsphere=6371200.,resolution='l',area_thresh=10000)
#         x, y = m(lons, lats)
#         m.scatter(x, y, marker='o', c=data, cmap='viridis')

#     else:
#         data_map = np.reshape(data, (ligne, colonne))
#         plt.imshow(data_map)
#         plt.colorbar()
        
#     plt.show()


#
# Python defines some variables before executing the code
#
# if __name__ == "__main__" : 
#     main()

