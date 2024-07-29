#%%
#### BEGIN IMPORT ####
import numpy as np
import matplotlib.pyplot as plt
from functions import *
import matplotlib.colors as clr
import matplotlib.patches as patches
import os
import rioxarray as rxr
import earthpy.spatial as es
#### EN IMPORT ####

#%%

#### Loading datas ####

mask_sar = np.load(r'D:/DONNEES-NUMPY/sar_mask.npy')
mask_landsat = np.load(r'D:/DONNEES-NUMPY/landsat_mask.npy')
mask_sar_landsat = np.load(r'D:/DONNEES-NUMPY/sar_landsat_mask.npy')

permanent_water_bodies = np.load(r'D:/DONNEES-NUMPY/permanent_water_bodies_dem.npy')
wetlands = np.load(r'D:/DONNEES-NUMPY/wetland.npy')

dem_path = r'D:\fabdem'
dem = read_nc_file(dem_path+'/dem_fabdem.nc', 'band_1_S')
dem = dem[:8000,:3000]

ndmi = np.load(r'D:/DONNEES-NUMPY/ndmi.npy')
ndvi = np.load(r'D:/DONNEES-NUMPY/ndvi.npy')

lat = np.load(r'D:/DONNEES-NUMPY/lat_gps.npy')
lon = np.load(r'D:/DONNEES-NUMPY/lon_gps.npy')



#%%
fig, ax = plt.subplots()
ax.matshow(permanent_water_bodies, cmap='grey')
ax.set_title('Permanent water bodies')


#%%

ndmi_mean = np.mean(ndmi,0)

fig, ax =plt.subplots()
ndmi_plot = ax.matshow(ndmi_mean, cmap='cividis')
ax.set_title('NDMI')
fig.colorbar(ndmi_plot, ax=ax)

#%%
wetlands = np.where(ndmi_mean>=0.8, 2, 0)
del ndmi_mean

#%%

#### Classification wetlands-permanent water bodies ####

carte = wetlands+permanent_water_bodies
carte_new = np.where(carte==3, 1, carte)

colors_wetland = ['white','red','lightgray']
colors_water_bodie = ['white','red']

axis_font = {'fontname':'Arial', 'size':'14'}

R = 8000
C = 3000

fig, ax = plt.subplots()
plot = ax.matshow(carte_new, cmap = clr.ListedColormap(colors_wetland))
ax.set_xticklabels([lon[j] for j in range(0,C,int(C/10))], rotation = 45,**axis_font)
ax.set_yticklabels([lat[j] for j in range(0,R,int(R/10))], **axis_font) 
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
ax.set_title('Classification')
# cbar = fig.colorbar(plot, ax=ax)
# cbar.set_ticks(ticks=[0,1,2],labels = ['',"Corps d'eau permanents", 'Zones humides'])


#%%

### Elevation des anomalies ###

fig,ax = plt.subplots()
# ax.matshow(mask, cmap='grey', zorder=1)
contour_plot = ax.contour(np.flipud(wetlands), cmap='viridis')
# contour_plot.set_linestyle('solid')
cbar = fig.colorbar(contour_plot, ax=ax)
ax.set_aspect('equal', adjustable='box')
ax.set_title('Contours au Nord du bassin versant, zone de fausses detection')


#%%

### Elevation des zones humides ###

dem_wetlands = np.where(wetlands==2, dem, np.nan)
fig, ax = plt.subplots()
plot=ax.matshow(dem_wetlands, cmap='viridis')
ax.set_title('Élévation des zones humides')
fig.colorbar(plot, ax=ax)

#%%

#### Localisation des mesures altimetriques ####

dem20 = np.where(dem<=20, dem, np.nan)
fig,ax = plt.subplots()
plot = ax.matshow(dem20, cmap='viridis')
#so : lon=1722, lat=4526
rect = patches.Rectangle((1720, 4540),4060-4020 , 1740-1700, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
#oueme : lon = 1855, lat = 23
rect = patches.Rectangle((1850, 20),40 , 40, linewidth=1, edgecolor='r', facecolor='none')
ax.set_title('Altimetry localisation')
fig.colorbar(plot, ax=ax)



#%%

#### Delete anamalies ####

new_mask = np.where((permanent_water_bodies==1)&(ndvi[5]<0.2), 1, mask_sar[7])
fig, ax = plt.subplots(1,2)
ax[0].matshow(new_mask, cmap='grey')
ax[1].matshow(mask_sar[5], cmap='grey')



