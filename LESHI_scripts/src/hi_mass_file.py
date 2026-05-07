import numpy as np
import pandas as pd
import cv2
import copy
import pickle
import emcee
import glob
import copy
import os

from scipy import special
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from astropy.io import fits  # We use fits to open the actual data file
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.stats import sigma_clipped_stats, sigma_clip
from spectral_cube import SpectralCube
from p_tqdm import p_map
import multiprocessing
import matplotlib.pyplot as plt


import warnings

# Convert RuntimeWarning into an exception
warnings.filterwarnings("error", category=RuntimeWarning)

def sky_coord_in_deg_from_pix(x_pix_coord,y_pix_coord,wcs):
    skycoord = SkyCoord.from_pixel(xp=x_pix_coord,yp=y_pix_coord, wcs=wcs)
    RA = skycoord.ra.degree
    DEC = skycoord.dec.degree
    return RA, DEC

def contour_sky_coord_in_deg_from_pix(contour,wcs):
    contour_sky = contour.copy()
    for row in range(contour.shape[0]):
        ra,dec = sky_coord_in_deg_from_pix(contour[row][0],contour[row][1],wcs)
        contour_sky[row][0],contour_sky[row][1] = float(ra),float(dec)
    return contour_sky

def contour_coord_in_pix_from_deg(contour,wcs):
    contour_pix = contour.copy()
    for row in range(contour.shape[0]):
        x_pix, y_pix = skycoord_to_pixel(SkyCoord(contour[row][0],contour[row][1], frame="fk5", unit="deg"),wcs)
        contour_pix[row][0],contour_pix[row][1] = float(x_pix),float(y_pix)
    return contour_pix

def contour_to_mask(contour_points,shape):
    contour = np.array([contour_points], dtype=np.int32)
    mask = np.zeros(shape, dtype=bool)
    temp_mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(temp_mask, contour, contourIdx=-1, color=1, thickness=cv2.FILLED)
    cv2.drawContours(temp_mask, contour, contourIdx=-1, color=1, thickness=1)
    mask[temp_mask == 1] = True
    return mask

def get_contour_spectrum(contour_sky,cube,wavelength_range_array):
    
    ra_max,dec_max = np.nanmax(contour_sky.T[0]),np.nanmax(contour_sky.T[1])
    ra_min,dec_min = np.nanmin(contour_sky.T[0]),np.nanmin(contour_sky.T[1])
    x_pix_max, y_pix_max = skycoord_to_pixel(SkyCoord(ra_max,dec_max, frame="fk5", unit="deg"),cube.wcs)
    x_pix_min, y_pix_min = skycoord_to_pixel(SkyCoord(ra_min,dec_min, frame="fk5", unit="deg"),cube.wcs)
    x_pix_max, y_pix_max = x_pix_max+10, y_pix_max+10
    x_pix_min, y_pix_min = x_pix_min-10, y_pix_min-10
    
    x_pix_left, x_pix_right = int(x_pix_min), int(x_pix_max)
    y_pix_left, y_pix_right = int(y_pix_min), int(y_pix_max)

    if x_pix_left<0: x_pix_left=0
    if y_pix_left<0: y_pix_left=0
    if x_pix_right>cube.shape[2]: x_pix_right=cube.shape[2]-1
    if y_pix_right>cube.shape[1]: y_pix_right=cube.shape[1]-1

    z_channel_left, z_channel_right = wavelength_range_array[0], wavelength_range_array[-1]

    if zchannel_left<0:zchannel_left=0
    if zchannel_right>cube.shape[0]: zchannel_right = -1
    zchannel_left, zchannel_right = int(zchannel_left), int(zchannel_right)

    cubelet=cube[zchannel_left:zchannel_right,ypix_left:ypix_right,xpix_left:xpix_right]
    contour_pix = contour_coord_in_pix_from_deg(contour_sky,cubelet.wcs)
    
    mask = contour_to_mask(contour_pix,cubelet[0,:,:].shape)
    masked_cubelet = cubelet.with_mmask(mask)
    flux=np.zeros(len(wavelength_range_array))
    for i, wavelength in enumerate(wavelength_range_array):     
        
        image_slice = cubelet.unmasked_data[wavelength,:,:]

        # apply mask and sum up the flux in channel image
        flux[i] = np.nansum(image_slice[mask])
    return flux
    
def read_in_radio_file(radio_file):
    cube=SpectralCube.read(radio_file)
    cube_data = cube.unmasked_data[:,:,:].value
    wcs_cube = cube.wcs
    cube_channel_length = cube_data.shape[0]
    dpix=np.abs(wcs_cube.wcs.cdelt[0])*3600
    try:
        try:beam_diameter_arcsec_array = cube.beams.major.value
        except: beam_diameter_arcsec_array = np.array([cube.beam.major.value])
    except: beam_diameter_arcsec_array = np.array([15])
        
    return cube, cube_data, wcs_cube, cube_channel_length,dpix, beam_diameter_arcsec_array

  
def measure_hi_mass(source_data_table_input):
    global source_data_table
    source_data_table = source_data_table_input

    source_data_table['log_MHI'], source_data_table['log_MHI_error'] = np.ones(len(source_data_table))*(-99), np.ones(len(source_data_table))*(-99)
    source_data_table['flux_Jy_Hz'], source_data_table['flux_Jy_Hz_error'] = np.ones(len(source_data_table))*(-99), np.ones(len(source_data_table))*(-99)
    source_data_table['3D_SNR'] = np.ones(len(source_data_table))*(-99)
    source_data_table['D_lumin'] = np.ones(len(source_data_table))*(-99)
    
    for source in range(len(source_data_table)):
        
        contour_file = path_to_contours+'/contours/%s_contours_sky.data'%(source_data_table['ID'].values[source])
        with open(contour_file, 'rb') as f:
            contours_sky = pickle.load(f)

        z_channel = source_data_table['z_channel_center'].values
        wavelength_range_left, wavelength_range_right = source_data_table['z_channel_min'].values, source_data_table['z_channel_max'].values+1
        if wavelength_range_left<0: wavelength_range_left=2
        if wavelength_range_right>= cube_data.shape[0] : wavelength_range_right = cube_data.shape[0] -2
        wavelength_range_array = np.arange(wavelength_range_left,wavelength_range_right,1,dtype=int)
        
        flux = get_contour_spectrum(contour_sky,cube,wavelength_range_array)

        total_flux = np.nansum(flux)
        b_diameter_pix = np.median(beam_diameter_arcsec_array)/dpix
        total_flux_Jy_Hz = total_flux*wcs_cube.wcs.cdelt[0]*1/(np.pi*b_diameter_pix**2/(4*np.log(2)))

        # calculate error
        total_flux_Jy_Hz = 

        
        D_lumin = (source_data_table['redshift'].values[source])
        MHI_mass = np.log10(49.7*total_flux_Jy_Hz*D_lumin**2)
        MHI_mass_error = total_flux_error/total_flux/np.log(10)

        source_data_table['log_MHI'].values[source], source_data_table['log_MHI_error'].values[source] = MHI_mass, MHI_mass_error
        source_data_table['flux_Jy_Hz'].values[source], source_data_table['flux_Jy_Hz_error'].values[source] = total_flux_Jy_Hz, total_flux_Jy_Hz_error
        source_data_table['3D_SNR'].values[source] = SNR
        source_data_table['D_lumin'].values[source] = D_lumin
        
    return source_data_table

def hi_mass_script(data_table, path_to_radio_file, path_to_contours_input, path_to_results_input,core_no_input):
    global cube, cube_data, wcs_cube, cube_channel_length, dpix, path_to_results, path_to_contours, beam_diameter_arcsec_array
 
    path_to_results = path_to_results_input
    path_to_contours = path_to_contours_input
    cube, cube_data, wcs_cube, cube_channel_length, dpix, beam_diameter_arcsec_array =read_in_radio_file(path_to_radio_file)


    if core_no_input == None: core_number = multiprocessing.cpu_count()
    else: core_number = core_no_input
        
    hi_mass_results = p_map(measure_hi_mass,
                                      [data_table[i:(i+1)] for i in range(len(data_table))], num_cpus=core_number)
    all_sources_dict = pd.concat(hi_mass_results)
    
    all_sources_dict.to_csv(path_to_results+'/found_sources_extent.csv',index=False)
    
    return all_sources_dict

    
