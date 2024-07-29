# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:41:34 2024

@author: nferdinand
"""

import datetime
import time
from snappy import ProductIO
from snappy import HashMap
import os, gc
from snappy import GPF
import shutil
from glob import glob
import rioxarray as rxr


def do_terrain_correction(source, proj, downsample):
    print('\tTerrain correction...')
    parameters = HashMap()
    parameters.put('demName', 'SRTM 3Sec')
    parameters.put('demResamplingMethod', 'BILINEAR_INTERPOLATION') 
    parameters.put('imgResamplingMethod', 'NEAREST_NEIGHBOUR')
    # parameters.put('mapProjection', proj)       # comment this line if no need to convert to UTM/WGS84, default is WGS84
    parameters.put('saveProjectedLocalIncidenceAngle', True)
    parameters.put('saveSelectedSourceBand', True)
    parameters.put('pixelSpacingInMeter', 10.0) 
    parameters.put('nodataValueAtSea', True)
    output = GPF.createProduct('Terrain-Correction', parameters, source)
    return output

def do_subset(source, wkt):
    print('\tSubsetting...')
    parameters = HashMap()
    parameters.put('geoRegion', wkt)
    output = GPF.createProduct('Subset', parameters, source)
    return output

def do_collocation(file_master, file_slave):
    sourceProducts = HashMap()
    sourceProducts.put("master", file_master)
    sourceProducts.put("slave", file_slave)
    parameters = HashMap()
    parameters.put("targetProductName", 'collocated')
    parameters.put("targetProductType", 'COLLOCATED')
    parameters.put('renameMasterComponents', True)
    parameters.put('renameSlaveComponents', True)
    parameters.put('masterComponentPattern', '${ORIGINAL_NAME}_M')
    parameters.put('slaveComponentPattern', '${ORIGINAL_NAME}_S')
    parameters.put('resamplingType', 'NEAREST_NEIGHBOUR')
    output = GPF.createProduct('CreateSTack', parameters, sourceProducts)
    return output


path_sar = r'D:\donnees_S1_SAR\0m4-0m8\S1A_IW_GRDH_1SDV_20201102T180157_20201102T180222_035073_04179F_CDCA.SAFE'


gc.enable()
gc.collect()

wkt_before_TC = "POLYGON((2.06559 7.04151, 2.66430 7.04151, \
    2.66430 6.25240, 2.06559 6.25240, 2.06559 7.04151))"
wkt = "POLYGON((2.64906 7.05798, 2.26453 7.05798, \
    2.26453 6.31849, 2.64906 6.31849, 2.64906 7.05798))"
    
proj = '''PROJCS["UTM Zone / World Geodetic System 1984",GEOGCS["World Geodetic System 1984",DATUM["World Geodetic System 1984",SPHEROID["WGS 84", 6378137.0, 298.257223563, AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich", 0.0, AUTHORITY["EPSG","8901"]],UNIT["degree", 0.017453292519943295],AXIS["Geodetic longitude", EAST],AXIS["Geodetic latitude", NORTH]],PROJECTION["Transverse_Mercator"],PARAMETER["central_meridian", -159.0],PARAMETER["latitude_of_origin", 0.0],PARAMETER["scale_factor", 0.9996],PARAMETER["false_easting", 500000.0],PARAMETER["false_northing", 0.0],UNIT["m", 1.0],AXIS["Easting", EAST],AXIS["Northing", NORTH]]'''    
sentinel_1 = ProductIO.readProduct(path_sar +  "\\manifest.safe")

subset_before_TC = do_subset(sentinel_1, wkt_before_TC)
down_tercorrected = do_terrain_correction(subset_before_TC, proj, 0)
down_subset = do_subset(down_tercorrected, wkt)


# sar_path = os.path.join("sentinel1",
#                         "nc_files",
#                         "annees",
#                         "2018")

#%%
path = r'D:\landsat'
os.chdir(path)


landsat_path = os.path.join("LC08_L2SP_192055_20191226_20200824_02_T1")

landsat_tif_path = glob(os.path.join(landsat_path,
                            "*B[2-3]*.tif"))


# sar_nc_path = glob(os.path.join(sar_path,
#                            "*20180222*.nc"))


# sar_nc = ProductIO.readProduct(sar_nc_path[0])

landsat_collocate = []
for band_path in landsat_tif_path:
    band = rxr.open_rasterio(band_path, masked=True)
    collocate = do_collocation(down_subset, band)
    landsat_collocate.append(collocate)
    