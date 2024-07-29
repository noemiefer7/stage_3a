# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:27:18 2024

@author: nferdinand
"""

# Need to configure Python to use the SNAP-Python (snappy) interface(https://senbox.atlassian.net/wiki/spaces/SNAP/pages/50855941/Configure+Python+to+use+the+SNAP-Python+snappy+interface)
# Read in unzipped Sentinel-1 GRD products (EW and IW modes)

import datetime
import time
from snappy import ProductIO
from snappy import HashMap
import os, gc
from snappy import GPF
import shutil

def do_apply_orbit_file(source):
    print('\tApply orbit file...')
    parameters = HashMap()
    parameters.put('Apply-Orbit-File', True)
    output = GPF.createProduct('Apply-Orbit-File', parameters, source)
    return output

def do_thermal_noise_removal(source):
    print('\tThermal noise removal...')
    parameters = HashMap()
    parameters.put('removeThermalNoise', True)
    output = GPF.createProduct('ThermalNoiseRemoval', parameters, source)
    return output

def do_calibration(source, polarization, pols):
    print('\tCalibration...')
    parameters = HashMap()
    parameters.put('outputSigmaBand', True)
    if polarization == 'DH':
        parameters.put('sourceBands', 'Intensity_HH,Intensity_HV')
    elif polarization == 'DV':
        parameters.put('sourceBands', 'Intensity_VH,Intensity_VV')
    elif polarization == 'SH' or polarization == 'HH':
        parameters.put('sourceBands', 'Intensity_HH')
    elif polarization == 'SV':
        parameters.put('sourceBands', 'Intensity_VV')
    else:
        print("different polarization!")
    parameters.put('selectedPolarisations', pols)
    parameters.put('outputImageScaleInDb', False)
    output = GPF.createProduct("Calibration", parameters, source)
    return output

def do_speckle_filtering(source):
    print('\tSpeckle filtering...')
    parameters = HashMap()
    parameters.put('filter', 'Lee Sigma')
    parameters.put('windowSize', '7x7')
    
    output = GPF.createProduct('Speckle-Filter', parameters, source)
    return output

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

def do_convert_to_db(source):
    parameters = HashMap()
    parameters.put('outputImageScaleInDb', True)
    output = GPF.createProduct('LinearToFromdB', parameters, source)
    return output

# def main():
## All Sentinel-1 data sub folders are located within a super folder (make sure the data is already unzipped and each sub folder name ends with '.SAFE'):
path = r'F:\donnees_S1_SAR\sar'

    
outpath = r'F:\nc_files'
if not os.path.exists(outpath):
    os.makedirs(outpath)
## well-known-text (WKT) file for subsetting (can be obtained from SNAP by drawing a polygon)
wkt_before_TC = "POLYGON((2.06559 7.04151, 2.66430 7.04151, \
    2.66430 6.25240, 2.06559 6.25240, 2.06559 7.04151))"
wkt = "POLYGON((2.64906 7.05798, 2.26453 7.05798, \
    2.26453 6.31849, 2.64906 6.31849, 2.64906 7.05798))"
## UTM projection parameters
proj = '''PROJCS["UTM Zone / World Geodetic System 1984",GEOGCS["World Geodetic System 1984",DATUM["World Geodetic System 1984",SPHEROID["WGS 84", 6378137.0, 298.257223563, AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich", 0.0, AUTHORITY["EPSG","8901"]],UNIT["degree", 0.017453292519943295],AXIS["Geodetic longitude", EAST],AXIS["Geodetic latitude", NORTH]],PROJECTION["Transverse_Mercator"],PARAMETER["central_meridian", -159.0],PARAMETER["latitude_of_origin", 0.0],PARAMETER["scale_factor", 0.9996],PARAMETER["false_easting", 500000.0],PARAMETER["false_northing", 0.0],UNIT["m", 1.0],AXIS["Easting", EAST],AXIS["Northing", NORTH]]'''

for folder in os.listdir(path):
    gc.enable()
    gc.collect()
    sentinel_1 = ProductIO.readProduct(path + "\\" + folder + "\\manifest.safe")
    print(sentinel_1)

    loopstarttime=str(datetime.datetime.now())
    print('Start time:', loopstarttime)
    start_time = time.time()

    ## Extract mode, product type, and polarizations from filename
    modestamp = folder.split("_")[1]
    productstamp = folder.split("_")[2]
    polstamp = folder.split("_")[3]

    polarization = polstamp[2:4]
    if polarization == 'DV':
        pols = 'VH,VV'
    elif polarization == 'DH':
        pols = 'HH,HV'
    elif polarization == 'SH' or polarization == 'HH':
        pols = 'HH'
    elif polarization == 'SV':
        pols = 'VV'
    else:
        print("Polarization error!")

    ## Start preprocessing:
    applyorbit = do_apply_orbit_file(sentinel_1)
    thermaremoved = do_thermal_noise_removal(applyorbit)
    calibrated = do_calibration(thermaremoved, polarization, pols)
    down_filtered = do_speckle_filtering(calibrated)
    del applyorbit
    del thermaremoved
    del calibrated
    ## IW images are downsampled from 10m to 40m (the same resolution as EW images).
    if (modestamp == 'IW' and productstamp == 'GRDH') or (modestamp == 'EW' and productstamp == 'GRDH'):
        subset_before_TC = do_subset(down_filtered, wkt_before_TC)
        down_tercorrected = do_terrain_correction(subset_before_TC, proj, 0)
        down_subset = do_subset(down_tercorrected, wkt)
        del subset_before_TC
        del down_filtered
        del down_tercorrected
    elif modestamp == 'EW' and productstamp == 'GRDM':
        tercorrected = do_terrain_correction(down_filtered, proj, 0)
        subset = do_subset(tercorrected, wkt)
        del down_filtered
        del tercorrected
    else:
        print("Different spatial resolution is found.")
        
    

    down = 1
    try: down_subset
    except NameError:
        down = None
    if down is None:
        print("Writing...")
        db_scale = do_convert_to_db(subset)
        del subset
        ProductIO.writeProduct(db_scale, outpath + '\\' + folder[:-5]+'.nc', 'NetCDF4-CF')
        del db_scale
    elif down == 1:
        print("Writing undersampled image...")
        db_scale = do_convert_to_db(down_subset)
        del down_subset
        ProductIO.writeProduct(db_scale, outpath + '\\' + folder[:-5] + '_40.nc', 'NetCDF4-CF')
        del db_scale
    else:
        print("Error.")

    print('Done.')
    sentinel_1.dispose()
    sentinel_1.closeIO()
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # shutil.rmtree(path + "\\" + folder)

# if __name__== "__main__":
#     main()