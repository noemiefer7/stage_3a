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

sar_path = r'D:\DONNES_SAT_TRAITEES\Sentinel-1\nc_files\annees'
folders = os.listdir(sar_path)

ndvi = np.load(r'D:/DONNEES-NUMPY/ndvi_ref.npy')
permanent_water_bodies = np.load(r'D:/DONNEES-NUMPY/permanent_water_bodies_dem.npy')


dem_path = r'D:\FABDEM'
dem = read_nc_file(dem_path+'/dem_fabdem.nc', 'band_1_S')

R = 8000
C = 3000

dem = dem[:R,:C]


water_prob = np.load(r'D:/DONNEES-NUMPY/water_prob.npy')
wetland = np.load(r'D:/DONNEES-NUMPY/wetland.npy')


lat = np.load(r'D:/DONNEES-NUMPY/lat_gps.npy')
lon = np.load(r'D:/DONNEES-NUMPY/lon_gps.npy')      
        

#%%
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
numpy_path = r'D:\numpy_mask_ok'
filenames = os.listdir(numpy_path)

R = 8000
C = 3000
s = np.zeros((12, R, C))
c = np.zeros(12)
pixels = []
dates = []
for filename in filenames:
    print(filename[:8])
    mask = np.load(numpy_path+'/'+filename)
    
    mask = np.where(mask==0,1,0)
    
    month = int(filename[4:6])-1
    s[month,:,:] += mask
    c[month] += 1

    pixels.append(np.count_nonzero(mask==1))
    dates.append(filename[:8])
    
    
months = ['Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet',
          'Aout', 'Septembre', 'Octobre', 'Novembre', 'Decembre']



water_prob = []
for i in range(12):
    water_prob.append(s[i,:,:]/c[i])
    

# fig, ax = plt.subplots(2,6, figsize=(12,6))
# ax = ax.flatten()
# for i in range(12):
#     im = ax[i].matshow(water_prob[i], cmap='rainbow')
#     ax[i].set_title(months[i])
#     ax[i].set_axis_off()
         
# fig.tight_layout()
# fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.02, pad=0.04)


#%%

#### Probabilites maps ####
cartes = [water_prob[i]+permanent_water_bodies for i in range(12)]
cartes = [np.where(cartes[i] >=1, 1, cartes[i]) for i in range(12)]

carte_wetland = [cartes[i]+wetland for i in range(12)]
cartes = [np.where(carte_wetland[i]==2, 2, cartes[i] ) for i in range(12)]


val = np.unique(cartes[0])
colormap = ListedColormap(['white','darkblue','blue','royalblue','cyan','seagreen','green','lime','greenyellow',
            'yellow','orange','darkorange','orangered','red','lightgray'])
# Définir les bornes pour chaque couleur
bounds = np.append(val, 3.5)
norm = BoundaryNorm(bounds, colormap.N)

 
    
months = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet',
          'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
x1, x2, y1, y2 = 50, 1500, 1400, 2000
R1 = x2-x1
C1 = y2-y1

axis_font = {'fontname':'Arial', 'size':'14'}

for i in range(12):
    fig, ax = plt.subplots()
    # ax = ax.flatten()
    im = ax.matshow(cartes[i][x1:x2,y1:y2], cmap=colormap, norm=norm)
    ax.set_title(months[i], **axis_font)
    ax.set_xticklabels([lon[j] for j in range(y1,y2,int(C1/10))], rotation = 45,**axis_font)
    ax.set_yticklabels([lat[j] for j in range(x1,x2,int(R1/10))], **axis_font) 
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    fig.tight_layout()
    # Ajout de la colorbar avec les bornes définies
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, ticks=val)
    




    





  









    
