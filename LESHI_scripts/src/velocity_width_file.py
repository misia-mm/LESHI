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

def frequency_to_channel(frequency,wcs):
    wave0,wavedelta,channel0 = wcs.wcs.crval[2],wcs.wcs.cdelt[2],wcs.wcs.crpix[2]
    channel = (frequency-wave0)/wavedelta+channel0
    return channel
    
def channel_to_frequency(channel,wcs):
    wave0,wavedelta,channel0 = wcs.wcs.crval[2], wcs.wcs.cdelt[2],wcs.wcs.crpix[2]
    frequency = wave0+wavedelta*(channel-channel0)
    return frequency

def contour_sky_coord_in_deg_from_pix(contour,wcs):
    contour_sky = np.ones(contour.shape)
    for row in range(contour.shape[0]):
        ra,dec = sky_coord_in_deg_from_pix(contour[row][0],contour[row][1],wcs)
        contour_sky[row][0],contour_sky[row][1] = ra,dec
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
    if x_pix_min>x_pix_max:
        x_pix_min, x_pix_max = x_pix_max, x_pix_min
    if y_pix_min>y_pix_max:
        y_pix_min, y_pix_max = y_pix_max, y_pix_min
    
    x_pix_max, y_pix_max = x_pix_max+10, y_pix_max+10
    x_pix_min, y_pix_min = x_pix_min-10, y_pix_min-10
    
    x_pix_left, x_pix_right = int(x_pix_min), int(x_pix_max)
    y_pix_left, y_pix_right = int(y_pix_min), int(y_pix_max)

    if x_pix_left<0: x_pix_left=0
    if y_pix_left<0: y_pix_left=0
    if x_pix_right>cube.shape[2]: x_pix_right=cube.shape[2]-1
    if y_pix_right>cube.shape[1]: y_pix_right=cube.shape[1]-1

    z_channel_left, z_channel_right = wavelength_range_array[0], wavelength_range_array[-1]
    z_channel_left, z_channel_right = int(z_channel_left), int(z_channel_right)

    cubelet=cube[z_channel_left:z_channel_right,y_pix_left:y_pix_right,x_pix_left:x_pix_right]
    contour_pix = contour_coord_in_pix_from_deg(contour_sky,cubelet.wcs)
    
    mask = contour_to_mask(contour_pix,cubelet[0,:,:].shape)

    flux=np.zeros(len(wavelength_range_array))
    for i in range(cubelet.shape[0]):     
        
        image_slice = cubelet.unmasked_data[i,:,:]

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



def calculate_velocity_widths(x,xp, xe, a, w, b, c, C, yerr):
    x_cont = np.arange(np.nanmin(x),np.nanmax(x),0.1)
    model_cont = busy(x_cont, xp, xe, a, w, b, c, C)
    
    first_maximum, second_maximum, average_maximum = find_peaks_of_busy_function(x, xp, xe, a, w, b, c, C, yerr)
    
    half_signal_channels = x_cont[model_cont>=( (average_maximum-C)*0.5 + C )]
    W50_channels = (half_signal_channels[-1]-half_signal_channels[0])+0.1
    x0 = int((np.nanmin(half_signal_channels) + np.nanmax(half_signal_channels))/2)

    if first_maximum>second_maximum:
        signal_channels = x_cont[model_cont>( (first_maximum-C)*0.02 + C )]
        W100_channels = (signal_channels[-1]-signal_channels[0])+0.1
    else:
        signal_channels = x_cont[model_cont>( (second_maximum-C)*0.02 + C )]
        W100_channels = (signal_channels[-1]-signal_channels[0])+0.1

    zmin = signal_channels[0]
    zmax = signal_channels[-1]
            
    return W50_channels, W100_channels, x0, zmin, zmax


def find_peaks_of_busy_function(x, xp, xe, a, w, b, c, C, yerr):
    xp, xe, a, w, b, c, C = round(xp,13), round(xe,13), round(a,13), round(w,13), round(b,13), round(c,13), round(C,13)
    x_cont = np.arange(np.nanmin(x),np.nanmax(x),0.05)
    
    y_cont = busy(x, xp, xe, a, w, b, c, C)
    first_half = copy.deepcopy(y_cont)
    first_half[x>=xe]=-99


    second_half = copy.deepcopy(y_cont)
    second_half[x<xe]=-99

    first_maximum = np.max(first_half)
    second_maximum = np.max(second_half)

    if np.all(first_half==-99) or np.all(second_half==-99):
        first_maximum, second_maximum, average_maximum = np.max(y_cont),np.max(y_cont),np.max(y_cont)
    else:
        if first_maximum==np.max(y_cont):
            if second_half[np.argmax(second_half)-1]==-99:
                second_maximum = a*(c*np.abs( (xe+w-xp))**2  +1)+C
                if first_maximum<second_maximum: second_maximum=first_maximum
                if second_maximum<yerr: second_maximum=first_maximum
                
        if second_maximum==np.max(y_cont):
            if first_half[np.argmax(first_half)+1]==-99:
                first_maximum = a*(c*np.abs( (xe-w-xp))**2  +1)+C
                if first_maximum>second_maximum: first_maximum=second_maximum
                if first_maximum<yerr: first_maximum=second_maximum
                
    average_maximum = (first_maximum + second_maximum)/2
    return first_maximum, second_maximum, average_maximum

    

# functions for fitting busy function to the spectrum
def log_likelihood_busy(theta, x, y, yerr):
    xp, xe, a, w, b, c, C = theta
    model = busy(x, xp, xe, a, w, b, c, C)
    sigma2 = yerr**2
    
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior_busy(x,theta):
    xp, xe, a, w, b, c, C = theta
    if a>0 and w>=3 and 100>b>=0 and c>=0 and 0<xp<cube_channel_length-1 and 0<xe<cube_channel_length-1 :
        return 0.0

    return -np.inf

def log_probability_busy(theta, x, y, yerr):
    lp = log_prior_busy(x,theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_busy(theta, x, y, yerr)

def busy(x, xp, xe, a, w, b, c, C):
    xp, xe, a, w, b, c, C = round(xp,13), round(xe,13), round(a,13), round(w,13), round(b,13), round(c,13), round(C,13)
    return a/4*( special.erf(b*(w+x-xe)) +1) * ( special.erf(b*(w-x+xe)) +1 )*(c*(np.abs(x-xp))**2 + 1)+C

# DESMOS calculator graph to play around with: https://www.desmos.com/calculator/cq5qxngzfe
# xe - horizontal offset of the middle of the graph, should be xe ~ channel  
# xp - horizontal offset of the parabola, should be xp~xe (controls how assymetric the profile is)
# C - vertical offset of the graph, should be C~ level of the spectrum
# w - controls the width of the peaks and height of the peaks, should be w>0
# c - controls the height of the peaks, should be c>0
# for c=0 and small w the profile has single peak
# a - controls the height of the whole graph, a>0
# a and c in tandem control how deep the central through is
# b - slope of the peaks, b>0

def fit_busy_velocity_width(x,y,source):
    # starting parameters
    pos = [source_data_table['fit_xp'].values[source]-1, source_data_table['fit_xe'].values[source]-1,
           source_data_table['fit_a'].values[source], source_data_table['fit_w'].values[source],
           source_data_table['fit_b'].values[source],source_data_table['fit_c'].values[source],
           source_data_table['fit_C'].values[source]]+1e-4 * np.random.randn(32, 7)
    pos = np.abs(pos)
    nwalkers, ndim = pos.shape
    
    mean, median, yerr = sigma_clipped_stats(y, sigma=3,maxiters=None)
    #yerr=np.abs(np.max(y)-np.min(y))*0.01
    if yerr==0: yerr=0.0001
        
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_busy, args=(x, y, yerr))
    sampler.run_mcmc(pos, 3000, progress=False);
    flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)
    
    fit_xp = np.percentile(flat_samples[:, 0], [50])[0]
    fit_xe = np.percentile(flat_samples[:, 1], [50])[0]
    fit_a = np.percentile(flat_samples[:, 2], [50])[0]
    fit_w = np.percentile(flat_samples[:, 3], [50])[0]
    fit_b = np.percentile(flat_samples[:, 4], [50])[0]
    fit_c = np.percentile(flat_samples[:, 5], [50])[0]
    fit_C = np.percentile(flat_samples[:, 6], [50])[0]

    # calculate velocity widths and errors
    W50_sampled_widths, W100_sampled_widths, x0_sampled = [], [], []
    for walker in range(len(flat_samples)):
        xp = flat_samples[walker, 0]
        xe = flat_samples[walker,1]
        a = flat_samples[walker,2]
        w = flat_samples[walker,3]
        b = flat_samples[walker,4]
        c = flat_samples[walker,5]
        C = flat_samples[walker,6]
        
        try:
            W50_channels, W100_channels, x0, zmin, zmax = calculate_velocity_widths(x,xp, xe, a, w, b, c, C, yerr)
            W50_sampled_widths=np.append(W50_sampled_widths,W50_channels)
            W100_sampled_widths=np.append(W100_sampled_widths,W100_channels)
            x0_sampled = np.append(x0_sampled,x0)
        except: pass
    

    W50_channels, W100_channels, x0_channels, zmin, zmax = calculate_velocity_widths(x,fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C, yerr)
    
    half_width = (W100_channels/2)
    W50_channels_error = (np.percentile(W50_sampled_widths[1000:],84) - np.percentile(W50_sampled_widths[1000:],16))/2
    W100_channels_error = (np.percentile(W100_sampled_widths[1000:],84) - np.percentile(W100_sampled_widths[1000:],16))/2
    x0_channels_error = (np.percentile(x0_sampled[1000:],84) - np.percentile(x0_sampled[1000:],16))/2
    params = fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C
    
    return W50_channels, W50_channels_error, W100_channels, W100_channels_error, x0_channels, x0_channels_error, half_width, zmin, zmax, params
    
def measure_velocity_width(source_data_table_input):
    global source_data_table
    source_data_table = source_data_table_input

    source_data_table['W50_channels'], source_data_table['W50_channels_error'] = np.ones(len(source_data_table))*(-99), np.ones(len(source_data_table))*(-99)
    source_data_table['W100_channels'], source_data_table['W100_channels_error'] = np.ones(len(source_data_table))*(-99), np.ones(len(source_data_table))*(-99)
    source_data_table['W50_km_s'], source_data_table['W50_km_s_error'] = np.ones(len(source_data_table))*(-99), np.ones(len(source_data_table))*(-99)
    source_data_table['W100_km_s'], source_data_table['W100_km_s_error'] = np.ones(len(source_data_table))*(-99), np.ones(len(source_data_table))*(-99)
    source_data_table['dv'] = np.ones(len(source_data_table))*(-99)

    source_data_table['fit_xp'] = np.ones(len(source_data_table))*np.array(source_data_table['z_channel_center'].values)
    source_data_table['fit_xe'] = np.ones(len(source_data_table))*np.array(source_data_table['z_channel_center'].values) 
    source_data_table['fit_a'] = np.ones(len(source_data_table))
    source_data_table['fit_w']= np.ones(len(source_data_table))*10 
    source_data_table['fit_b'] = np.ones(len(source_data_table))*np.sqrt(np.pi/2)/5
    source_data_table['fit_b'].values[source_data_table['fit_b'].values>=100]=99
    source_data_table['fit_c'] = np.ones(len(source_data_table))*1e-4*1.1
    source_data_table['fit_C'] = np.ones(len(source_data_table))*0
        
    for source in range(len(source_data_table)):
        dv = 3*10**8*wcs_cube.wcs.cdelt[0]/source_data_table['frequency_Hz'].values
        source_data_table['dv'].values = dv
        
        contour_file = path_to_contours+'/contours/%s_contours_sky.data'%(source_data_table['ID'].values[source])
        with open(contour_file, 'rb') as f:
            contours_sky = pickle.load(f)

        z_channel = source_data_table['z_channel_center'].values
        wavelength_range_left, wavelength_range_right = z_channel-200, z_channel+200
        if wavelength_range_left<0: wavelength_range_left=2
        if wavelength_range_right>= cube_data.shape[0] : wavelength_range_right = cube_data.shape[0] -2
        wavelength_range_array = np.arange(wavelength_range_left,wavelength_range_right,1,dtype=int)
        
        flux = get_contour_spectrum(contour_sky,cube,wavelength_range_array)
        W50_channels, W50_channels_error, W100_channels, W100_channels_error, x0_channels, x0_channels_error, half_width, zmin, zmax, params = fit_busy_velocity_width(wavelength_range_array,flux,source)
        
        fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C = params
        source_data_table['fit_xp'].values[source] = fit_xp
        source_data_table['fit_xe'].values[source] = fit_xe
        source_data_table['fit_a'].values[source] = fit_a
        source_data_table['fit_w'].values[source] = fit_w
        source_data_table['fit_b'].values[source] = fit_b
        source_data_table['fit_c'].values[source] = fit_c
        source_data_table['fit_C'].values[source] = fit_C

        source_data_table['W50_channels'].values[source], source_data_table['W50_channels_error'].values[source] = W50_channels, W50_channels_error
        source_data_table['W100_channels'].values[source], source_data_table['W100_channels_error'].values[source] = W100_channels, W100_channels_error
        source_data_table['W50_km_s'].values[source], source_data_table['W50_km_s_error'].values[source] = W50_channels*dv, W50_channels_error*dv
        source_data_table['W100_km_s'].values[source], source_data_table['W100_km_s_error'].values[source] = W100_channels*dv, W100_channels_error*dv
    
    return source_data_table

def velocity_width_script(data_table, path_to_radio_file, path_to_contours_input, path_to_results_input,core_no_input):
    global cube, cube_data, wcs_cube, cube_channel_length, dpix, path_to_results, path_to_contours, beam_diameter_arcsec_array
 
    path_to_results = path_to_results_input
    path_to_contours = path_to_contours_input
    cube, cube_data, wcs_cube, cube_channel_length, dpix, beam_diameter_arcsec_array =read_in_radio_file(path_to_radio_file)

    
    if core_no_input == None: core_number = multiprocessing.cpu_count()
    else: core_number = core_no_input
        
    velocity_width_results = p_map(measure_velocity_width,
                                      [data_table[i:(i+1)] for i in range(len(data_table))],num_cpus=core_number)
    all_sources_dict = pd.concat(velocity_width_results)
    
    all_sources_dict.to_csv(path_to_results+'/found_sources_extent.csv',index=False)
    
    return all_sources_dict

    
