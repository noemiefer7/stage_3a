import rioxarray as rxr
import xarray as xr
import numpy as np
import netCDF4 as nc
import earthpy.spatial as es
import matplotlib.pyplot as plt



# Opens a single tif file and returns an xarray object
def open_clean_band(band_path, crop_layer=None):
    """A function that opens a Landsat band as an (rio)xarray object

    Parameters
    ----------
    band_path : list
        A list of paths to the tif files that you wish to combine.

    crop_layer : geopandas geodataframe
        A geodataframe containing the clip extent of interest. NOTE: this will 
        fail if the clip extent is in a different CRS than the raster data.

    Returns
    -------
    An single xarray object with the Landsat band data.

    """
    
    if crop_layer is not None:
        try:
            clip_bound = crop_layer.geometry
            cleaned_band = rxr.open_rasterio(band_path,
                                             masked=True).rio.clip(clip_bound,
                                                                   from_disk=True).squeeze()
        except Exception as err:
            print("Oops, I need a geodataframe object for this to work.")
            print(err)
    else:
        cleaned_band = rxr.open_rasterio(band_path,
                                         masked=True).squeeze()

    return cleaned_band


def process_bands(paths, crop_layer=None, stack=False):
    """
    Open, clean and crop a list of raster files using rioxarray
    
    Parameters
    ----------
    paths : list
        A list of paths to raster files that could be stack (of the
        same resolution, crs and spatial extent)
    
    crop_layer: geodataframe
        A geodataframe containing the crop geometry that you wish to crop
        your data to.
        
    stack : boolean
        If True, return a stacked xarray object. If false will return a list
        of xarray object
        
    Returns
    -------
        Either a list of xarray objects or a stacked xarray object
    
    """
    
    all_bands = []
    for i, aband in enumerate(paths):
        cleaned = open_clean_band(aband, crop_layer)
        cleaned["band"] = i+1
        all_bands.append(cleaned)
 
    if stack:
        print("I'm stacking your data now.")
        return xr.concat(all_bands, dim="band")
    else:
        print("Returning a list of xarray objects.")
        return all_bands
    
    

def otsu_thresholding(image):
    """
    Provide a threshold for the input image based on the otsu method.
    
    Parameters
    ----------
    image : Numpy array
        Numpy array containing the dB values of a SAR image.
    

    Returns
    -------
    threshold : float
        The otsu threshold

    """
    
    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins='auto')
     
    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
     
    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    sigmaB = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(sigmaB)
    
    threshold = bin_mids[:-1][index_of_max_val]

    return threshold



def water_mask(image):
    '''
    Calculate the water mask over a sar or andzi image, return the otsu
    threshold

    Parameters
    ----------
    image : array of float32
      

    Returns
    -------
    mask : array of 0 or 1
        water mask
    threshold : float
    '''
    threshold = otsu_thresholding(image)  
    
    # verification du seuil
    if threshold>-6:
        ind = np.where(image<-0.5)
        image_bis = []
        for i in range(len(ind[0])):
            image_bis.append(image[ind[0][i]][ind[1][i]])
        threshold = otsu_thresholding(image_bis)
 
    mask = image>threshold
                
    return mask,threshold



def read_nc_file(nc_file, variable):
    '''
    Read nc file and translate it into numpy array

    Parameters
    ----------
    nc_file : netCDF file
        
    variable : Name of the band

    Returns
    -------
    sigma0 : numpy array

    '''
    data = nc.Dataset(nc_file)
    sigma0 = np.array(data.variables[variable][:])
    return sigma0



def cloud_removal(im, types):
    '''
    Cloud correction over optical images

    Parameters
    ----------
    im : array of float from landsat images, must contain at list 8 bands :
        The sixth firts : Blue, Green, Red, NIR, SWIR1, SWIR2. The last one
        is the Quality Assurance pixel
        
    types : string
    Can be Sentinel or Landsat

    Returns
    -------
    im_pre_cl_free : array of float
        Cloud removed
    mask : array of boolean
    

    '''
    bands = im[0:6,:,:]
    qa_pixel = im[6,:,:]
    
    match types:
        case 'Landsat':
            all_masked_values = [21890, 22018, 22280, 23826, 23888, 24032]
        case 'Sentinel':
            all_masked_values = [66, 130, 194, 198, 200, 202]

    # Mask the data using the pixel QA layer
    mask = np.isin(qa_pixel, all_masked_values)
    
    # Créer un masque pour les valeurs à masquer
    mask = np.isin(qa_pixel, all_masked_values)
    
    # Iterate over each band and mask the cloud pixels
    im_pre_cl_free = np.array([np.where(mask, np.nan, band) for band in bands])
    
    return im_pre_cl_free, mask





def sar_processing(im, dem, R1, C1):
    '''
    Apply water mask on sar images

    Parameters
    ----------
    im : array of float from sar image
    
    dem : array of float from dem
        
    R1 : row size
        
    C1 : column size
        

    Returns
    -------
    mask : array of float - water mask

    '''
    R = 8000
    C = 3000
    im = im[:R1,:C1]

    mask, th = water_mask(im)
    mask = np.where(dem<=20, mask, np.nan)
    mask = mask[:R,:C]
    
    return mask



def landsat_processing(band1, band2, dem, R1, C1):
    '''
    Apply water mask over the landsat images

    Parameters
    ----------
    band1 : array of float
        first band in the differenced normalisation
    band2 : array of float
        second band in the differenced normalisation
    dem : array of float from dem
        
    R1 : row size
        
    C1 : column size
        

    Returns
    -------
    mask : array of float - water mask

    '''
    index = es.normalized_diff(band1, band2)
    R = 8000
    C = 3000
    
    index = index[:R1, :C1]
    th = otsu_thresholding(index[~np.isnan(index)])
    mask = index<th
    mask = np.where(dem<=20, mask, np.nan)
    mask = mask[:R,:C]
  
    return mask


def decimal_to_gps(coord):
    '''
    Convert latitude and longitude from decimal form to gps form

    Parameters
    ----------
    coord : list of float
       Latitude or longitude in decimal 

    Returns
    -------
    gps_format : list of string
        latitude or longitude in gps format

    '''
    degrees = int(coord)
    minutes_float = (coord - degrees) * 60
    minutes = int(minutes_float)
    seconds = round((minutes_float - minutes) * 60)
    
    # Formater la chaîne de caractères pour obtenir le format GPS
    gps_format = "{0}° {1}' {2}\"".format(degrees, minutes, seconds)
    
    return gps_format 
 
    
def Bmax_calc(image):
    '''
    Calcul of the bmax indice

    Parameters
    ----------
    image : array of float
        

    Returns
    -------
    Bmax : float
    

    '''
    R, C = image.shape
    nb_pixel = R * C
    
    flat_image = image[~np.isnan(image)]
    
    hist, pixel_values = np.histogram(flat_image, bins='auto' , density=True)  # Use density=True to directly get normalized histogram
    hist_size = hist.shape[0]
    
    pixel_values = pixel_values[0:hist_size]
    
    BCV = []
    p1 = 0
    p2 = 0
    m1 = 0
    m2 = 0
    
    DPD = hist / nb_pixel
    index = np.arange(hist_size)
    for t in range(1, hist_size - 1):
        
        p1 = np.sum(DPD[:t])  # probability of class 1
        p2 = np.sum(DPD[t:])  # probability of class 2 
        
        m1 = np.sum(index[:t] * DPD[:t]) / p1  # mean of class 1
        m2 = np.sum(index[t:] * DPD[t:]) / p2  # mean of class 2
        
        sigma1 = np.sum((index[:t] - m1) ** 2 * DPD[:t]) / p1   # class variance 1
        sigma2 = np.sum((index[t:] - m2) ** 2 * DPD[t:]) / p2  # class variance 2
        
        sigmaW = p1 * sigma1 + p2 * sigma2  # within-class variance
        sigmaB = p1 * p2 * (m1 - m2) ** 2  # between-class variance
        sigmaT = sigmaW + sigmaB  # total variance
        
        BCV.append(sigmaB / sigmaT)  # normalized between-class variance

    Bmax = np.max(BCV)
    
    return Bmax



def overall_accuracy(mask_sar, mask_landsat):
    '''
    Calculate the overall accuracy between sar mask and landsat mask

    Parameters
    ----------
    mask_sar : array of float
        water masks from sar images
    mask_landsat : array of float
        water masks from landsat images

    Returns
    -------
    OA : list of float
         overall accuracies for each date

    '''
    
    total = np.count_nonzero((mask_sar==0)|(mask_sar==1))
    
    OA = []

    for i in range (8):
        LW_SW = (np.count_nonzero((mask_landsat[i]==0)&(mask_sar[i]==0))/total*100)
        LNW_SNW = (np.count_nonzero((mask_landsat[i]==1)&(mask_sar[i]==1))/total*100)
        OA.append(LW_SW+LNW_SNW)
        
    return OA




def water_elevation(water_level, dem, water_bodies, lon, lat, shapes):
    '''
    Calculate the elevation of the water by using DEM and water altimetry

    Parameters
    ----------
    water_level : list of float
        water level from altimetry measure
    dem : array of float
        DEM
    water_bodies : array of float
        map of permanent water bodies
    lon : list of string
        Longitude
    lat : list of string
        Latitude
    shapes : list of integer
        x, y coordinates of the extract

    Returns
    -------
    None.

    '''
    x1, x2, y1, y2 =  shapes[0], shapes[1], shapes[2],shapes[3]
    R1 = x2-x1
    C1 = y2-y1

    dem1 = dem[x1:x2, y1:y2]

    water_bodies1 = water_bodies[x1:x2, y1:y2]
    

   

    diff = np.zeros((2,R1,C1))

    for i in range(2):
      
        diff[i] =  np.where((water_bodies1==0)&(dem1<=water_level[i])
                              ,water_level[i] - dem1, np.nan)
        
        diff[i] = np.where(water_bodies1==0, diff[i], 0)
        

        
    axis_font = {'fontname':'Arial', 'size':'14'}

    mois = ['Septembre','Octobre']
    for i in range(2):
        fig, ax = plt.subplots()
        im = ax.matshow(diff[i], cmap = 'viridis', vmin=0, vmax=6)
        ax.set_xticklabels([lon[j] for j in range(y1,y2,int(C1/10))], rotation = 45,**axis_font)
        ax.set_yticklabels([lat[j] for j in range(x1,x2,int(R1/10))], **axis_font) 
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        ax.set_title(mois[i],**axis_font)
        fig.colorbar(im)
    