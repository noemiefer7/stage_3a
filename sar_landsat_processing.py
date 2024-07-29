
#%%
#### BEGIN IMPORT ####
import os 

from functions import *
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import earthpy.spatial as es
import earthpy.plot as ep
import netCDF4 as nc

## END IMPORT ####

#%%
#### DEM #####
dem_path = r'D:\FABDEM'
dem = read_nc_file(dem_path+'/dem_fabdem.nc', 'band_1_S')
R = 8000
C = 3000

dem = dem[:R,:C]

#%%
#### SAR Processing ####
# Import SAR datas
sar_path = r'D:\DONNES_SAT_TRAITEES\landsat_sentinel1\S1_nc_files'
sar_filenames = os.listdir(sar_path)
mask_sar = []
th_sar = []

# Create watermasks from sar images
for filename in sar_filenames:
    sigma = read_nc_file(sar_path+'/'+filename, 'Sigma0_VV_db')
    mask = sar_processing(sigma, dem, 8233, 4281)
    mask_sar.append(mask)
    
del mask
del sigma
    
#%%
#### Landsat processing ####
landsat_path = r'D:\DONNES_SAT_TRAITEES\landsat_sentinel1\L8_tif\tif'
landsat_filenames = os.listdir(landsat_path)
andwi_list = []
th_landsat = []
ndvi_list = []
ndmi_list = []

# Create watermasks from landsat images
for filename in landsat_filenames:
    tif_file = rxr.open_rasterio(landsat_path+'/'+filename, masked=True)
    
    andwi = landsat_processing(tif_file[3]+tif_file[4]+tif_file[5], tif_file[0]+tif_file[1]+tif_file[2], 
                                dem, 8233, 4281)
    
  
    andwi_list.append(andwi)
    
    
    
    # ndvi = landsat_processing(tif_file[2], tif_file[3], 
    #                            dem, 8233, 4281)
    # ndvi_list.append(ndvi)
    
    # ndmi = landsat_processing(tif_file[1], tif_file[2], 
    #                            dem, 8233, 4281)
    # ndmi_list.append(ndmi)
    
    
del tif_file  

del andwi  
# del ndvi
# del ndmi



#%%

#### Comparison between optical and radar images

water_detection = []
for i in range(8):
    water_pixel = np.where((andwi_list[i]==0) & (mask_sar[i]==0),
                           0, 1)
    water_detection.append(water_pixel)
    
    
#%%


for i in range(11):
    fig, ax = plt.subplots(1,3,figsize=(10,6))
    ax[0].matshow(mask_sar[i], cmap='grey')
    ax[0].set_title('SAR, threshold: '+sar_filenames[i][17:25])
    
    ax[1].matshow(andwi_list[i], cmap='grey')
    ax[1].set_title('Landsat, threshold: '+landsat_filenames[i][:-4])
    
    ax[2].matshow(water_detection[i], cmap='grey')
    ax[2].set_title('SAR and Landsat')
    
    

#%%
# Accurancy calculation
OA = overall_accuracy(mask_sar, andwi_list)
    
#%%
### Crues 2021 #### 

landsat_20210926 = np.load(r'D:/DONNES_SAT_TRAITEES/landsat_sentinel1/L8_mask/L8_20191226.npy')
sar_20210922 = np.load(r'D:/DONNEES-NUMPY/water_mask_sar/20210922.npy')
sar_20211004 = np.load(r'D:/DONNEES-NUMPY/water_mask_sar/20211004.npy')

dem = dem[:R,:C]
sar_20210922 =sar_20210922[:R,:C]
sar_20211004 = sar_20211004[:R,:C]
landsat_20210926 = landsat_20210926[:R,:C]

sar_20210922_dem = np.where(dem<=20, sar_20210922, 1)
sar_20211004_dem = np.where(dem<=20, sar_20211004, 1)
landsat_20210926_dem = np.where(dem<=20, landsat_20210926,1)

water_detection = np.where((landsat_20210926_dem==0)&((sar_20210922_dem==0)|
                           (sar_20211004_dem==0)), 0, 1)

fig, ax = plt.subplots(1,4, figsize=(10,6))
ax[0].matshow(landsat_20210926_dem, cmap='gray')
ax[0].set_title('Landsat mask, 2021-09-26')

ax[1].matshow(sar_20210922_dem, cmap='gray')
ax[1].set_title('Sar mask, 2021-09-22')

ax[2].matshow(sar_20211004_dem, cmap='gray')
ax[2].set_title('Sar mask, 2021-10-04')

ax[3].matshow(water_detection, cmap='grey')
ax[3].set_title('Sar et Landsat')







    
