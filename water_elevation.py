#%%

#### BEGIN IMPORT

import numpy as np
import matplotlib.pyplot as plt
import os
from functions import*
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

#### END IMPORT

#%%

#### Loading datas ####
permanent_water_bodies = np.load(r'D:/DONNEES-NUMPY/permanent_water_bodies_dem.npy')


dem_path = r'D:\FABDEM'
dem = read_nc_file(dem_path+'/dem_fabdem.nc', 'band_1_S')

R = 8000
C = 3000

dem = dem[:R,:C]


water_prob = np.load(r'D:/DONNEES-NUMPY/water_prob.npy')


lat = np.load(r'D:/DONNEES-NUMPY/lat_gps.npy')
lon = np.load(r'D:/DONNEES-NUMPY/lon_gps.npy') 

#%%

####  visualization of contour in the north part of the watershed ####
dem_nord = dem[0:2500,0:2500]
sigma_path = r'D:\sentinel1\nc_files\anomalies'
sigma_20220201 = read_nc_file(sigma_path+'/S1A_IW_GRDH_1SDV_20220201T180159_20220201T180224_041723_04F6EF_51A4_40.nc', 'Sigma0_VV_db')

mask, th = water_mask(sigma_20220201)
mask = mask[:2500,:2500]

mask = np.where(dem_nord<=20, mask, np.nan)
water_level = [i for i in range(0,20,2)]

fig,ax = plt.subplots()
# ax.matshow(mask, cmap='grey', zorder=1)
contour_plot = ax.contour(np.flipud(dem_nord), cmap='viridis', levels=water_level)
contour_plot.set_linestyle('solid')
cbar = fig.colorbar(contour_plot, ax=ax)
ax.set_aspect('equal', adjustable='box')
ax.set_title('Contours au Nord du bassin versant, zone de fausses detection')

#%%

#### Water elevation so ####
dem_path = r'D:\fabdem'
dem = read_nc_file(dem_path+'/dem_fabdem.nc', 'band_1_S')

water_level_so = np.load(r'D:/DONNEES-NUMPY/water_level_so.npy')
water_level_crue_so = water_level_so[8:10]

shapes =  [4325, 4725, 1497, 1897]

water_elevation(water_level_crue_so, dem, permanent_water_bodies,
                lon, lat, shapes)


#%%

#### Water elevations Oueme ####

dem_path = r'D:\fabdem'
dem = read_nc_file(dem_path+'/dem_fabdem.nc', 'band_1_S')


water_level_oueme = np.load(r'D:/DONNEES-NUMPY/water_level_oueme.npy')
water_level_crue_oueme = water_level_oueme[8:10]

shapes = [50, 1500, 1400, 2000]

water_elevation(water_level_crue_oueme, dem, permanent_water_bodies,
                lon, lat, shapes)



#%%
hist = []
pixels = []

c = 0
for elev in diff:
    histo = np.histogram(elev[~np.isnan(elev)], bins= 20)
    plt.figure()
    plt.plot(histo[1][1:len(histo[0])], histo[0][1:])
    plt.title("Histogramme de l'hauteur d'eau en "+mois[c], ** axis_font)
    plt.show()
    c += 1





    
