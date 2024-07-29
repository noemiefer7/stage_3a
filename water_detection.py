#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import netCDF4 as nc
from functions import *
import os
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy import ndimage
import rasterio
from rasterio.enums import Resampling
from osgeo import gdal, osr
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

#%%
plt.close('all')
nc_folder = 'processed_data/092922/extracts/'
nc_filenames = os.listdir(nc_folder)

lat_list = []
lon_list = []
lat_gps = []
lon_gps = []
sigma0_vv_list = []
sigma0_vh_list = []
ratio_list = []

for filename in nc_filenames:
    sigma0_vh, lat, lon = read_nc_file(nc_folder+filename, "Sigma0_VH_db")
    sigma0_vv,_,_ = read_nc_file(nc_folder+filename, "Sigma0_VV_db")
    
    [R,C] = np.shape(sigma0_vh)
    
    sigma0_vv_list.append(sigma0_vv)
    sigma0_vh_list.append(sigma0_vh)
    
    lat_list.append(lat)
    lon_list.append(lon)
    
    lat_gps.append([decimal_to_gps(lat[i]) for i in range(R)])
    lon_gps.append([decimal_to_gps(lon[i]) for i in range(R)])
    
   
    ratio = np.zeros((R,C))
    for i in range(R):
        for j in range(C):
            ratio[i,j] = sigma0_vv[i,j]/sigma0_vh[i,j]
    
    ratio_list.append(ratio)
    
#%%
# plt.close('all')
folder = r'D:\nc_files\etiage'
filenames = os.listdir(folder)
sum_mask = 0
c = 0



for filename in filenames:
    print(c)
    sigma0_vv, lat, lon = read_nc_file(folder+'/'+filename, 'Sigma0_VV_db')
    
    [R,C] = np.shape(sigma0_vv)
    lat_gps = [decimal_to_gps(lat[i]) for i in range(R)]
    lon_gps = [decimal_to_gps(lon[i]) for i in range(C)]
    
    
    
    
    # ech = 5
    # grid, ech_size = data_extraction(sigma0_vv[:,:4281], ech)
    
    # mask = np.zeros((ech_size*ech,4281))
    
    mask_extract = []
    c_ech = 0
    # for extract in grid:
    #     # gs = gridspec.GridSpec(2, 2)
    #     # plt.figure()
    #     # ax = plt.subplot(gs[0,0]) 
    #     # ax.imshow(extract, cmap = 'gray')
    #     # ax.set_title('Extract')
    #     # ax.set_xticklabels(lon_gps, rotation = 45)
    #     # ax.set_yticklabels([lat_gps[i] for i in range(c,c+ech_size)])
        
    #     # ax = plt.subplot(gs[1,0])
    #     mask_extract,_, threshold = (water_mask(extract))
    #     # ax.matshow(mask_extract, cmap = 'binary')
    #     # ax.set_title('Water detection, threshold: ' + str(threshold))
    #     # ax.set_xticklabels(lon_gps, rotation = 45)
    #     # ax.set_yticklabels([lat_gps[i] for i in range(c,c+ech_size)])
        
    #     # ax = plt.subplot(gs[:,1])
    #     # hist, pixels = np.histogram(extract, bins = 'auto')
    #     # L = len(hist)
    #     # ax.stem(pixels[0:L],hist, )
    #     # ax.stem(threshold, np.max(hist), markerfmt = 'ro')
    #     # ax.set_title('Extract histogram')
    #     # ax.set_xlabel('Pixel values')
    #     # ax.set_ylabel('Number of pixels')
        
    #     mask[c_ech:c_ech+ech_size,:] = mask_extract
     
        
    #     c_ech += ech_size
    mask,th = water_mask(sigma0_vv[:8232,:4281])
    sum_mask += mask
        
    
    # fig, ax = plt.subplots(1,2, figsize=(10,6))
    # sigma_mask, th = water_mask(sigma0_vv)
      
    # ax[0].matshow(sigma_mask, cmap='viridis')
    # ax[0].set_title('Water detection without grid')
    # ax[0].set_xticklabels([lon_gps[i] for i in range(0,C,int(C/9))], rotation = 45)
    # ax[0].set_yticklabels([lat_gps[i] for i in range(0,R,int(R/9))])
        
    # ax[1].matshow(mask, cmap='viridis')
    # ax[1].set_title('Water detection with grid')
    # ax[1].set_xticklabels([lon_gps[i] for i in range(0,C,int(C/9))], rotation = 45)
    # ax[1].set_yticklabels([lat_gps[i] for i in range(0,R,int(R/9))])
    c += 1

  
water_prob_mat = sum_mask/c

fig,ax = plt.subplots(1,1,figsize=(10,6))
color_rainbow = ['indigo','blue','cyan', 'lime', 'green', 'yellow', 'orange', 'orangered', 'red', 'firebrick']
prob_show = ax.matshow(water_prob_mat, cmap='rainbow')
fig.colorbar(prob_show, ax=ax)
ax.set_title('Etiage. Nb image :'+str(c))
ax.set_xticklabels([lon_gps[i] for i in range(int(C/10),C,int(C/10))], rotation = 45)
ax.set_yticklabels([lat_gps[i] for i in range(int(R/10),R,int(R/10))])


etiage = np.ones((8232,4281))
for i in range(8232):
    for j in range(4281):
        if water_prob_mat[i,j]>=0.9:
            etiage[i,j] = 0
    
fig,ax = plt.subplots()
ax.matshow(etiage, cmap='gray')
ax.set_title('Etiage')
ax.set_xticklabels([lon_gps[i] for i in range(int(C/10),C,int(C/10))], rotation = 45)
ax.set_yticklabels([lat_gps[i] for i in range(int(R/10),R,int(R/10))])


#%%
# Superposition DEM
plt.close('all')
axis_font = {'fontname':'Arial', 'size':'13'}
title_font = {'fontname':'Arial', 'size':'22', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}

file_path = "DEM/dem1.nc"
file_dem,_,_ = read_nc_file(file_path, "band_1_S")

file_dem = np.flipud(file_dem)

lat = np.load('processed_data/lat.npy')
lon = np.load('processed_data/lon.npy')

R,C = dem.shape

lat_gps  = [decimal_to_gps(lat[i]) for i in range(R)]
lon_gps  = [decimal_to_gps(lon[i]) for i in range(C)]

fig, ax= plt.subplots(1,2, figsize=(10,6))
contour_plot = ax[1].contour(file_dem, cmap = "viridis",
            levels = list(range(0, 15, 1)))
ax[1].set_title("Elevation Contours of Nokou Lake", **title_font)
ax[1].set_xticklabels([lon_gps[i] for i in range(0,C, int(C/6))], rotation=45, **axis_font)
ax[1].set_yticklabels([lat_gps[i] for i in range(0,R, int(R/6))], **axis_font)
contour_plot.set_linestyle('solid')
cbar = fig.colorbar(contour_plot, ax=ax)
ax[1].set_aspect('equal', adjustable='box')
plt.show()

c = 0 
for image in sigma0_vv_list:
    mask = water_mask(image)
    mask = mask[16:985, 16:985]
    
    dem_extract, lat_extract, lon_extract = dem_extraction(dem, lon_dem, lat_dem, lat_list[c], lon_list[c])
    
    dem_extract_reshape = np.repeat(dem_extract, 3, axis = 0)
    dem_extract_reshape = np.repeat(dem_extract_reshape, 3, axis = 1)
    
    dem_mask = dem_extract_reshape*mask
    
    
    
    
    min_dem = np.min(dem_extract)
    max_dem = np.max(dem_extract)
    
    fig, ax = plt.subplots(1, 2, figsize=(10,6))
    ax[0].imshow(mask)
    ax[0].set_xticklabels([lon_gps[c][i] for i in range(0,R, int(R/6))], rotation=45, **axis_font)
    ax[0].set_yticklabels([lat_gps[c][i] for i in range(0,R, int(R/6))], **axis_font)
    ax[0].set_title('Water mask', **title_font)
    

    contour_plot = ax[1].contour(dem_extract_reshape, cmap = "viridis",
                levels = list(range(0, 15, 1)))
    ax[1].set_title("Elevation Contours of Nokou Lake", **title_font)
    ax[1].set_xticklabels([lon_gps[c][i] for i in range(0,R, int(R/6))], rotation=45, **axis_font)
    ax[1].set_yticklabels([lat_gps[c][i] for i in range(0,R, int(R/6))], **axis_font)
    contour_plot.set_linestyle('solid')
    cbar = fig.colorbar(contour_plot, ax=ax)
    ax[1].set_aspect('equal', adjustable='box')
    plt.show()
    
    c += 1
    
    
#%%
# Classification eau-terre-ville
plt.close('all')

sigma_110819 = np.load('processed_data/etiages_npy/20191226.npy')
permanent_water = np.load('processed_data/permanent_water.npy')

mask_classif, th = water_mask(sigma_110819)


label = [0,1,2]
class_ = ['Land', 'Water', 'Urban areas']
colors = ['green', 'blue', 'red']

plt.figure()
plt.matshow(mask, cmap=matplotlib.colors.ListedColormap(colors))
cb = plt.colorbar()
loc = np.arange(0,max(label),max(label)/float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(class_)

#%%
# seuillage par fenêtre glissante
plt.close('all')

window_size = 50
step = int(window_size/2)

sigma = np.load('processed_data/crues_npy/VH/20220929.npy')

extract = sigma[500:1500, :]
R,C = np.shape(extract)

sum_detection = np.zeros((R,C))

bmax_list = []
region_index = []
for i in range(0, R, step):
    for j in range(0, C, step):
        if i+window_size<=R and j+window_size<=C:
            window = extract[i:i+window_size, j:j+window_size]
            region_index.append(str(i)+":"+str(i+window_size)+";"+str(j)+':'+str(j+window_size))
            bmax = Bmax_calc(window)
            bmax_list.append(bmax)
            mask, th = water_mask(window)
            sum_detection[i:i+window_size, j:j+window_size] += mask
            
df_target = pd.DataFrame({'Index region':region_index, 'Bmax': bmax_list})
            

# mask = np.ones((R,C))
# for i in range(R):
#     for j in range(C):
#         if sum_detection[i,j] == 4:
#             mask[i,j] = 0
            
mask_simple,th = water_mask(extract)

file_dem = "DEM/dem_mosaik.tif" 
dem, lat_dem, lon_dem = tif_reading(file_dem)
dem_extract, lat_extract, lon_extract = extraction(dem, lon_dem, lat_dem, lat, lon)

# dem_extract_reshape = np.repeat(dem_extract, 3, axis = 0)
# dem_extract_reshape = np.repeat(dem_extract_reshape, 3, axis = 1)

dem_extract_reshape = np.flipud(dem_extract)

hist, pixels = np.histogram(extract, bins = 'auto')
plt.close('all')
gs = gridspec.GridSpec(3, 1)

ax = plt.subplot(gs[0,0])
colors = ['blue', 'green', 'yellow','orange', 'red']
ax.imshow(extract, cmap='gray')
ax.set_title('extract')

# ax = plt.subplot(gs[0,1])
# ax.contour(dem_extract_reshape, cmap = "viridis",
#             levels = list(range(0, 15, 1)))
# ax.set_title('DEM')

ax = plt.subplot(gs[1,0])
ax.matshow(mask_simple, cmap = 'gray')
ax.set_title('Seuillage sans fenêtre')

ax = plt.subplot(gs[2,0])
ax.matshow(sum_detection, cmap='gray')
ax.set_title('Seuillage avec fenêtre')

#%%
lat = np.load('processed_data/lat.npy')
lon = np.load('processed_data/lon.npy')

tif_file = 'google_earth_engin/classification_nokoue.tif'
classif, lat_classif, lon_classif = tif_reading(tif_file)
_
classif = np.delete(classif, (0), axis = 0)

classif_reshape, lat_class, lon_class = extraction(classif, lon_classif, lat_classif, lat, lon)
R,C = np.shape(classif_reshape)
permanent_water = (classif_reshape==80) ^ (classif_reshape==90)

plt.close('all')
plt.matshow(permanent_water)

label = [0,1,2,3,4,5,6,7,8,9]
palette= ['darkgreen', 'orange', 'yellow', 'fuchsia', 'tomato', 'grey','indigo','aqua', 'yellowgreen','beige']
classif_object = ['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Bare vegetation', 'Permanent water bodies', 'Herbaceaous Wetland', 'Mangroves'  'Moss and lichen']
plt.matshow(classif_reshape, cmap=matplotlib.colors.ListedColormap(palette))
cb = plt.colorbar()
loc = np.arange(0,max(label),max(label)/float(len(palette)))
cb.set_ticks(loc)
cb.set_ticklabels(classif_object)

# %%
#checking bmax
plt.close('all')

bmax1 = Bmax_calc(extract[0:150,0:150])
bmax2 = Bmax_calc(extract[0:150,2475:2625])
bmax3 = Bmax_calc(extract[0:150,1650:1800])

nb_bin = 200*200

hist1, pixels1 = np.histogram(sigma_110819[0:200,0:200], bins = nb_bin)
hist2, pixels2 = np.histogram(sigma_110819[6800:7000,900:1100],  bins = nb_bin)
hist3, pixels3 = np.histogram(sigma_110819[200:400,1500:1700],  bins = nb_bin)

fig, ax = plt.subplots(2,3, figsize=(10,6))
ax[0,0].imshow(extract[0:150,0:150], cmap = 'gray')
ax[0,0].set_title('Bmax :'+str(bmax1))
ax[1,0].hist(extract[0:150,0:150].flatten(), bins='auto')

ax[0,1].imshow(extract[0:150,2475:2625], cmap='gray')
ax[0,1].set_title('Bmax :'+str(bmax2))   
ax[1,1].hist(extract[0:150,2475:2625].flatten(), bins='auto')

ax[0,2].imshow(extract[0:150,1650:1800], cmap='gray')
ax[0,2].set_title('Bmax :'+str(bmax3))  
ax[1,2].hist(extract[0:150,1650:1800].flatten(), bins='auto')

#%%  
#Variable size window
lat = np.load('processed_data/lat.npy')
lon = np.load('processed_data/lon.npy')
lat_gps = [decimal_to_gps(lat[i]) for i in range(8201)]
lon_gps = [decimal_to_gps(lon[i]) for i in range(4248)]


fx_min = [15, 1612, 2971, 4231, 5329, 6374, 7564]
fy_min = [300, 927, 1165, 1183, 1105, 845, 0]
fx_max = [1612, 2971, 4231, 5329, 6374, 7564, 8201]
fy_max = [2612, 2516, 2531, 2926, 3252, 4248, 4248]

# np_folder = 'processed_data/crues_npy/'
# filenames = os.listdir(np_folder)
c = 1
sum_prob = np.zeros((8201, 4248))
sigma = np.load('processed_data/crues_npy/VH/20220929.npy')
# for filename in filenames:
#     sigma = np.load(np_folder+filename)
#     mask_sum = np.zeros((8201, 4248))
#     th_list = []
    
for i in range(7):
    mask, th = water_mask(sigma[fx_min[i]:fx_max[i],fy_min[i]:fy_max[i]])
    mask_sum[fx_min[i]:fx_max[i],fy_min[i]:fy_max[i]] = mask
    th_list.append(th)
        
    plt.figure()
    ax = plt.subplot()
    ax.matshow(mask_sum, cmap = 'gray')
    ax.set_title('Masque avec fenêtres inhomogènes')
    ax.set_yticklabels([lat_gps[i] for i in range(0,8201,int(8201/9))])
    ax.set_xticklabels([lon_gps[i] for i in range(0,4248,int(4248/9))], rotation=45)
    # sum_prob += mask_sum
    # c+=1

water_prob_mat = sum_prob/c    
fig,ax = plt.subplots()
color_rainbow = ['indigo','blue','cyan', 'lime', 'green', 'yellow', 'orange', 'orangered', 'red', 'firebrick']
prob_show = ax.matshow(water_prob_mat, cmap=matplotlib.colors.ListedColormap(color_rainbow))
fig.colorbar(prob_show, ax=ax)
ax.set_title('Water probabilities')
ax.set_xticklabels([lon_gps[i] for i in range(0,C,int(C/9))], rotation = 45)
ax.set_yticklabels([lat_gps[i] for i in range(0,R,int(R/9))])       


plt.figure()
ax = plt.subplot()
ax.imshow(sigma_110819, cmap='gray')
for i in range(7):
    rect = patches.Rectangle((fy_min[i], fx_min[i]),fy_max[i]-fy_min[i] , fx_max[i]-fx_min[i], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
ax.set_title('Découpage en fenêtre')
ax.set_yticklabels([lat_gps[i] for i in range(0,8201,int(8201/9))])
ax.set_xticklabels([lon_gps[i] for i in range(0,4248,int(4248/9))], rotation=45)


#%%
# looking for permanent water bodies
sigma_etiage = np.load('processed_data/etiages_npy/20191226.npy')
R,C = np.shape(sigma_etiage)

# mask_sum = np.ones((R, C))
# th_list = []

# for i in range(7):
#     mask, th = water_mask(sigma_etiage[fx_min[i]:fx_max[i],fy_min[i]:fy_max[i]])
#     mask_sum[fx_min[i]:fx_max[i],fy_min[i]:fy_max[i]] = mask
#     th_list.append(th)
    
mask, th = water_mask(sigma_etiage)
mask_filtered = ndimage.median_filter(mask, size=10)

np.save('processed_data/permanent_water', mask_filtered)


fig, ax = plt.subplots(1,2, figsize=(10,6))
ax[0].matshow(mask, cmap = 'gray')
ax[0].set_title('Masque')
ax[0].set_yticklabels([lat_gps[i] for i in range(0,R,int(R/9))])
ax[0].set_xticklabels([lon_gps[i] for i in range(0,C,int(C/9))], rotation=45)    

ax[1].matshow(mask_filtered, cmap = 'gray')
ax[1].set_title('Masque filtré par filtre médian')
ax[1].set_yticklabels([lat_gps[i] for i in range(0,R,int(R/9))])
ax[1].set_xticklabels([lon_gps[i] for i in range(0,C,int(C/9))], rotation=45)   

#%%
filename = 'processed_data/crues_npy/20220929.npy'


sigma = np.load(filename)
R,C = np.shape(sigma)

R_reduced = int(R/10)
sigma = sigma[0:R_reduced, :]
binary_threshold = -13
binary_image = np.zeros((R,C))

for i in range(R_reduced):
    for j in range(C):
        if sigma[i,j]>-10:
            sigma[i,j] = np.nan
            

            
tile_size = 200
tile_index = []
bmax_list = []
for i in range(0,R_reduced,tile_size):
    for j in range(0,C,tile_size):
        if i+tile_size<R_reduced and j+tile_size<C :
            tile = sigma[i:i+tile_size, j:j+tile_size]
            tile_index.append(str(i)+":"+str(i+tile_size)+";"+str(j)+':'+str(j+tile_size))
            bmax = Bmax_calc(tile)
            bmax_list.append(bmax)
           
df = pd.DataFrame({'Region': tile_index, 'Bmax': bmax_list})

#%%
filename = 'processed_data/crues_npy/VH/20220929.npy'


sigma = np.load(filename)

sigma = sigma[500:1500,:]

from skimage.filters import threshold_otsu, threshold_local


global_thresh = threshold_otsu(sigma)
binary_global = sigma > global_thresh

block_size = 35
local_thresh = threshold_local(sigma, block_size, offset=10)
binary_local = sigma > local_thresh

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax = axes.ravel()
plt.gray()

ax[0].imshow(sigma)
ax[0].set_title('Original')

ax[1].imshow(binary_global)
ax[1].set_title('Global thresholding')

ax[2].imshow(binary_local)
ax[2].set_title('Local thresholding')

for a in ax:
    a.set_axis_off()

plt.show()

#%%
plt.close('all')
sigma_20220929 = np.load('processed_data/crues_npy/VV/20220929.npy')
lat = np.load('processed_data/lat.npy')
lon = np.load('processed_data/lon.npy')

R,C = sigma_20220929.shape
lat_gps = [decimal_to_gps(lat[i]) for i in range(R)]
lon_gps = [decimal_to_gps(lon[i]) for i in range(C)]


ax = plt.subplot()
ax.imshow(sigma_20220929, cmap='gray')
#so : lon=1722, lat=4526
rect = patches.Rectangle((1720, 4540),4060-4020 , 1740-1700, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
#oueme : lon = 1855, lat = 23
rect = patches.Rectangle((1850, 20),40 , 40, linewidth=1, edgecolor='r', facecolor='none')
ax.set_title('Altimetry localisation')
ax.set_yticklabels(lat_gps[::int(R/9)])
ax.set_xticklabels(lon_gps[::int(C/9)], rotation=45)  

#%%
path_folder = r'D:\nc_files\2023'
filenames = os.listdir(path_folder)

for filename in filenames:
    sigma, lat, lon = read_nc_file(path_folder+'/'+filename, 'Sigma0_VV_db')
    R,C = sigma.shape
    lat_gps = [decimal_to_gps(lat[i]) for i in range(R)]
    lon_gps = [decimal_to_gps(lon[i]) for i in range(C)]
    
    mask,  th= water_mask(sigma)
    
    fig,ax = plt.subplots()
    ax.matshow(mask, cmap='gray')
    ax.set_title(filename[17:25]+', seuil:'+str(th))
    ax.set_yticklabels(lat_gps[::int(R/9)])
    ax.set_xticklabels(lon_gps[::int(C/9)], rotation=45)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    fig.show()
    
#%%
from scipy.signal import argrelextrema

path_folder = r'D:\nc_files\2023'
sigma_20220917, lat,lon= read_nc_file(path_folder+'/S1A_IW_GRDH_1SDV_20230831T180213_20230831T180238_050123_06082D_B87A_40.nc', 'Sigma0_VV_db')    
R,C = sigma_20220917.shape

lat_gps = [decimal_to_gps(lat[i]) for i in range(R)]
lon_gps = [decimal_to_gps(lon[i]) for i in range(C)]
    
# [rows, cols] = np.shape(sigma_20220917)
# mask = np.ones((rows, cols))
vals = []
# th = -13.31
# for i in range (rows):
#     for j in range (cols):
#         if sigma_20220917[i,j]<th:
#             mask[i,j]  = 0    

mask, th = water_mask(sigma_20220917)

fig,ax = plt.subplots()
ax.matshow(mask, cmap='gray')
ax.set_title('20230912'+', seuil:'+str(th))
ax.set_yticklabels(lat_gps[::int(R/9)])
ax.set_xticklabels(lon_gps[::int(C/9)], rotation=45)
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
fig.show()


ind = np.where(sigma_20220917<-0.5)
sigma_bis = []
for i in range(len(ind[0])):
    sigma_bis.append(sigma_20220917[ind[0][i]][ind[1][i]])
    
hist, pixel_values = np.histogram(sigma_bis, bins='auto')

plt.figure()
plt.stem(pixel_values[:len(hist)],hist)
plt.stem(th, 120000, 'r')
plt.title('Histogramme 20230831, seuil:'+str(th))

#%%
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema

path_folder = r'D:\nc_files\2022'
sigma_20220929, lat,lon = read_nc_file(path_folder+'/S1A_IW_GRDH_1SDV_20220929T180209_20220929T180234_045223_0567C0_92CA_40.nc', 'Sigma0_VV_db')    

R,C = sigma_20220929.shape
lat_gps = [decimal_to_gps(lat[i]) for i in range(R)]
lon_gps = [decimal_to_gps(lon[i]) for i in range(C)]

ind = np.where(sigma_20220929<-0.5)
sigma_bis = []
for i in range(len(ind[0])):
    sigma_bis.append(sigma_20220929[ind[0][i]][ind[1][i]])
    
hist, pixel_values = np.histogram(sigma_bis, bins='auto')

histhat = savgol_filter(hist,51,3)
ind_max = argrelextrema(histhat, np.greater, order=250)[0]
int_max = histhat[ind_max[0]:ind_max[1]]
ind_min_int_max = argrelextrema(int_max, np.less, order=300)[0]

ind_min = np.where(histhat == int_max[ind_min_int_max])

_,th_otsu = water_mask(sigma_20220929)
th_min = pixel_values[ind_min]
th = (th_otsu+th_min)/2

milieu = int((ind_min+ind_max[1])/2)

int_eau = pixel_values[ind_min[0][0]:milieu]
int_terre = pixel_values[milieu:ind_max[1]]

th_otsu2 = threshold_otsu_impl(sigma_20220929)

mask = np.ones((R, C))

for i in range (R):
    for j in range (C):
        if sigma_20220929[i,j]<th_otsu2:
            mask[i,j]  = 0 

plt.figure()
plt.stem(pixel_values[:len(hist)],hist)
plt.plot(pixel_values[:len(hist)],histhat, 'r')
plt.stem(th_min, 120000, 'g')
plt.stem(th_otsu, 120000, 'r')
plt.stem(th, 120000, 'b')


fig, ax = plt.subplots()
ax.matshow(mask, cmap='gray')
ax.set_title('20220929, seuil: '+str(th))
ax.set_yticklabels(lat_gps[::int(R/9)])
ax.set_xticklabels(lon_gps[::int(C/9)], rotation=45)
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
