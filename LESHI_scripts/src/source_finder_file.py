import numpy as np
import pandas as pd
import emcee
import os
import gc
from p_tqdm import p_map
import multiprocessing
from collections import defaultdict

from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from spectral_cube import SpectralCube

import matplotlib
from matplotlib.colors import Normalize

from photutils.detection import find_peaks
from photutils.aperture import CircularAperture, RectangularAperture, CircularAnnulus, ApertureStats, EllipticalAnnulus

import time
import copy
import os

import warnings

# Convert RuntimeWarning into an exception
warnings.filterwarnings("error", category=RuntimeWarning)

# get sky coordinates in degrees from pixel coordinates
def sky_coord_in_deg_from_pix(x_pix_coord,y_pix_coord,wcs):
    skycoord = SkyCoord.from_pixel(xp=x_pix_coord,yp=y_pix_coord, wcs=wcs)
    RA = skycoord.ra.degree
    DEC = skycoord.dec.degree
    return RA, DEC

# get frequency form channel
def channel_to_frequency(channel,wcs):
    wave0,wavedelta,channel0 = wcs.wcs.crval[2],wcs.wcs.cdelt[2],wcs.wcs.crpix[2]
    frequency = wave0+wavedelta*(channel-channel0)
    return frequency
    
# functions for fitting gaussian function to the spectrum
def log_likelihood(theta, x, y, yerr):
    H, A, x0, sigma = theta
    model = H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(theta):
    H, A, x0, sigma = theta
    if 0.0 <= A and 0 <= x0 <= cube_channel_range and cube_channel_range>sigma>signal_persistence_threshold/4:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

def gauss(x, H, A, x0, sigma): 
            return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
def fit_gauss(x,y):
    H_0, A_0, x0_0, sigma_0 = 0,np.abs(np.nanmax(y)),x[np.nanargmax(y)],int_image_length/6
    if x0_0==0: x0_0=1
    if x0_0==cube_channel_range: x0_0=cube_channel_range-1
    pos = [H_0, A_0, x0_0, sigma_0] + 1e-5 * np.random.randn(32, 4)
    
    nwalkers, ndim = pos.shape
    yerr=0.05*np.abs(np.nanmax(y))
    if yerr==0: yerr = 0.001
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
    sampler.run_mcmc(pos, 3000, progress=False);
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    fit_H = np.percentile(flat_samples[:, 0], [50])
    fit_A = np.percentile(flat_samples[:, 1], [50])
    fit_x0 = np.percentile(flat_samples[:, 2], [50])
    fit_sigma = np.percentile(flat_samples[:, 3], [50])

    rsqr_array = np.ones(5)
    if int(fit_x0[0])<x[0]: x0_id=0
    elif int(fit_x0[0])>x[-1]: x0_id=len(x)-1
    else:x0_id = np.where(x==int(fit_x0[0]))[0][0]
        
    
    for i in range(5):
        if 6*fit_sigma[0]<30:
            left_border = np.linspace(0,x0_id -30,5)[i] 
            right_border = np.linspace(len(x)-1,x0_id +30,5)[i]
        else:
            left_border = np.linspace(0,int(x0_id -6*fit_sigma[0]),5)[i] 
            right_border = np.linspace(len(x)-1,int(x0_id +6*fit_sigma[0]),5)[i] 
        if left_border<0: left_border=0
        if right_border>=len(x): right_border=len(x)-1
        y_central = y[int(left_border):int(right_border)]
        x_central = x[int(left_border):int(right_border)]
    
        fit_y = gauss(x_central, fit_H[0], fit_A[0], fit_x0[0], fit_sigma[0]) 

        if np.nansum((y_central-np.nansum(y_central)/len(y_central))**2)!=0: 
            mean_y = fit_H[0]
            rsqr_array[i] = 1- (np.nansum((y_central-fit_y)**2))/(np.nansum((y_central-mean_y)**2))
        else: rsqr_array[i] = 0
    rsqr = np.nanmax(rsqr_array)
    
    # fit_y = gauss(x, fit_H[0], fit_A[0], fit_x0[0], fit_sigma[0]) 
    # mean_y = fit_H[0]
    #rsqr = 1- (np.nansum((y-fit_y)**2))/(np.nansum((y-mean_y)**2))
        
    
    return rsqr, fit_x0[0], fit_sigma[0], fit_A[0], fit_H[0]

def check_source_gaussian_fit(found_sources_dict):
    for source in range(len(found_sources_dict)):
        x_coord,y_coord = found_sources_dict['x_coord'].values[source],found_sources_dict['y_coord'].values[source]
        int_im_number = found_sources_dict['int_im_number'].values[source]
        integrated_image_central_channel = int(found_sources_dict['int_im_channel'].values[source])
        
        integrated_image_beam_diameter = integrated_image_beam_diameter_array[int(int_im_number)]

        while True:
            # get spectrum for beam pixels around source
            half_width = int(10*round(int_image_length))
            if half_width<100: half_width = 100
            flux_cubelet_start = integrated_image_central_channel-half_width
            flux_cubelet_end = integrated_image_central_channel+half_width
            if flux_cubelet_start<cube_start: flux_cubelet_start=cube_start
            if flux_cubelet_end>cube_end: flux_cubelet_end=cube_end
            flux = get_beam_spectrum(x_coord,y_coord,integrated_image_beam_diameter,flux_cubelet_start,flux_cubelet_end)
            
            sloped_continuum = True
            if sloped_continuum:
                x = np.arange(0,len(flux))
                z = np.polyfit(x,flux,1)
                flux = flux - (z[1]+x*z[0])
            channels=np.arange(flux_cubelet_start,flux_cubelet_end,1)
            
            # check spectrum
            found_sources_dict['rsqr'].values[source],found_sources_dict['gauss_x0'].values[source],found_sources_dict['gauss_sigma'].values[source], found_sources_dict['gauss_A'].values[source], found_sources_dict['gauss_H'].values[source] = fit_gauss(channels,flux)
            
            if found_sources_dict['rsqr'].values[source]<rsqr_threshold:
                found_sources_dict['failed_spectral_fit'].values[source] = 1
            else:
                found_sources_dict['failed_spectral_fit'].values[source] = 0
                
            break
    
    return found_sources_dict

def calculate_spectral_SNR(flux_long,flux_central):
    flux_long=sigma_clip(flux_long, sigma=3,  maxiters=5,masked=False)
    median = np.nanmedian(flux_long)
    std = np.nanstd(flux_long)
    #mean, median, std = sigma_clipped_stats(flux_long,sigma=3)
    flux_central = np.sort(flux_central)
    average_max_signal = np.nansum(flux_central[len(flux_central)-signal_persistence_threshold:])/signal_persistence_threshold
    if std==0:
        spectral_SNR_average = 0
        spectral_SNR = 0
        
    else:
        spectral_SNR_average = (average_max_signal-median)/std
        spectral_SNR = (flux_central[-1]-median)/std
    return spectral_SNR, spectral_SNR_average
    
    
def get_beam_spectrum(x_coord,y_coord,integrated_image_beam_diameter,flux_cubelet_start,flux_cubelet_end):
    if flux_cubelet_start<cube_start: flux_cubelet_start=cube_start
    if flux_cubelet_end>cube_end: flux_cubelet_end=cube_end

    radius = int(integrated_image_beam_diameter/2)-1
    if radius<2.5: radius=2.5

    flux_cubelet = np.array(data_cube.unmasked_data[int(flux_cubelet_start):int(flux_cubelet_end),
                                                y_coord-int(radius*3):y_coord+int(radius*3),
                                                x_coord-int(radius*3):x_coord+int(radius*3)],dtype=np.float32)
    rows, cols = flux_cubelet.shape[1], flux_cubelet.shape[2]
    y, x = np.ogrid[:rows, :cols]
    center_row, center_col = (rows - 1) / 2, (cols - 1) / 2
    distance = np.sqrt((x - center_col)**2 + (y - center_row)**2)
    
    flux = np.ones(int(flux_cubelet_end-flux_cubelet_start))
    for channel in range(len(flux)):
        (flux_cubelet[channel])[distance>radius] = np.nan
        flux[channel] = np.nansum(flux_cubelet[channel])
        
    return flux
    
def find_sequence(arr,seq):
    N_arr, N_seq = arr.size, seq.size
    r_seq = np.arange(N_seq)
    M = (arr[np.arange(N_arr-N_seq+1)[:,None] + r_seq] == seq).all(1)
    return M.any()

def check_persistence(x_coord,y_coord,integrated_image_beam_diameter,integrated_image_central_channel):
    # estimate frame background noise
    if integrated_image_beam_diameter<4:integrated_image_beam_diameter=4

    ypix_left, ypix_right = int(y_coord-(bg_box_size/2)), int(y_coord+(bg_box_size/2))
    if ypix_left<0: ypix_left=0
    if ypix_right>data_cube.shape[1]: ypix_right = data_cube.shape[1]-1
        
    xpix_left, xpix_right = int(x_coord-(bg_box_size/2)), int(x_coord+(bg_box_size/2))
    if xpix_left<0: xpix_left=0
    if xpix_right>data_cube.shape[2]: xpix_right = data_cube.shape[2]-1
        
    image_around_source = np.array(data_cube.unmasked_data[integrated_image_central_channel,
                                   ypix_left:ypix_right,
                                   xpix_left:xpix_right],dtype=np.float32)
    mean, median, std = sigma_clipped_stats(image_around_source[~np.isnan(image_around_source)],sigma=3)
    frame_threshold = median+std*SNR_channel_frame_threshold

    # get cubelet around source
    cubelet_start = integrated_image_central_channel-int(round(int_image_length*3/2))
    cubelet_end = integrated_image_central_channel+int(round(int_image_length*3/2))
    if cubelet_start<cube_start: cubelet_start=cube_start
    if cubelet_end>cube_end: cubelet_end=cube_end
    
    cubelet_around_source = np.array(data_cube.unmasked_data[cubelet_start:cubelet_end,
                            y_coord-int(integrated_image_beam_diameter):y_coord+int(integrated_image_beam_diameter),
                            x_coord-int(integrated_image_beam_diameter):x_coord+int(integrated_image_beam_diameter)],dtype=np.float32)
    
    rows, cols = cubelet_around_source.shape[1], cubelet_around_source.shape[2]
    y, x = np.ogrid[:rows, :cols]
    center_row, center_col = (rows - 1) / 2, (cols - 1) / 2
    distance = np.sqrt((x - center_col)**2 + (y - center_row)**2)
    signal_in_channel = np.full(cubelet_around_source.shape[0],False)
    for channel in range(cubelet_around_source.shape[0]):
        channel_frame_around_source = cubelet_around_source[channel]
        (channel_frame_around_source)[distance>int(integrated_image_beam_diameter/2)] = np.nan
        channel_frame_around_source[0][0]=0

        if np.nanmax(channel_frame_around_source)>frame_threshold: # check if max value is bigger that threshold SNR
            channel_frame_around_max_value = cubelet_around_source[channel]
            idx = np.unravel_index(np.nanargmax(channel_frame_around_source), channel_frame_around_source.shape)
            distance_from_max = np.sqrt((x - idx[1])**2 + (y - idx[0])**2)
            (channel_frame_around_max_value)[distance_from_max>int(integrated_image_beam_diameter/2)] = np.nan
            beam_area = len(channel_frame_around_max_value[~np.isnan(channel_frame_around_max_value)])
            
            # check if values within beam around the max value are bigger than half of threshold SNR with some tolerance
            if len(channel_frame_around_max_value[channel_frame_around_max_value<0.5*frame_threshold])>np.ceil(beam_area*0.1):
                signal_in_channel[channel] = False
            else: signal_in_channel[channel] = True

                
    consecutive_signal = find_sequence(signal_in_channel,np.full(signal_persistence_threshold,True))

    return consecutive_signal

def check_source_in_spectral(found_sources_dict):
    
    for source in range(len(found_sources_dict)):
        x_coord,y_coord = found_sources_dict['x_coord'].values[source],found_sources_dict['y_coord'].values[source]
        int_im_number = found_sources_dict['int_im_number'].values[source]
        integrated_image_central_channel = int(found_sources_dict['int_im_channel'].values[source])
        
        integrated_image_beam_diameter = integrated_image_beam_diameter_array[int(int_im_number)]

        while True:

            # check persistence
            if check_persistence(x_coord,y_coord,integrated_image_beam_diameter,integrated_image_central_channel)==False:
                found_sources_dict['failed_persistence'].values[source] = 1
                break
            else:
                found_sources_dict['failed_persistence'].values[source] = 0

            # check spectral SNR
            # get spectrum for beam pixels around source
            half_width = 15*int(round(int_image_length))
            if half_width<80: half_width = 80
            flux_cubelet_start_full = integrated_image_central_channel-half_width
            flux_cubelet_end_full = integrated_image_central_channel+half_width
            if flux_cubelet_start_full<cube_start: flux_cubelet_start_full=cube_start
            if flux_cubelet_end_full>cube_end: flux_cubelet_end_full=cube_end
            
            channels = np.arange(flux_cubelet_start_full,flux_cubelet_end_full)
            flux_full = get_beam_spectrum(x_coord,y_coord,integrated_image_beam_diameter,flux_cubelet_start_full,flux_cubelet_end_full)

            sloped_continuum = True
            if sloped_continuum:
                x = np.arange(0,len(flux_full))
                z = np.polyfit(x,flux_full,1)
                flux_full = flux_full - (z[1]+x*z[0])
        
            # get spectrum for beam pixels around source
            flux_cubelet_start_central = integrated_image_central_channel-int(3/2*round(int_image_length))
            flux_cubelet_end_central = integrated_image_central_channel+int(3/2*round(int_image_length))
            if flux_cubelet_start_central<cube_start: flux_cubelet_start_central=cube_start
            if flux_cubelet_end_central>cube_end: flux_cubelet_end_central=cube_end 
            flux_central = flux_full[(channels>=flux_cubelet_start_central)&(channels<flux_cubelet_end_central)]
            
            flux_long = np.append( flux_full[(channels>=flux_cubelet_start_full)&(channels<flux_cubelet_start_central)],
                                   flux_full[(channels>=flux_cubelet_end_central)&(channels<flux_cubelet_end_full)])
            
            half_width = int(7.5*round(int_image_length))
            if half_width<50: half_width = 50
            flux_cubelet_start_medium = integrated_image_central_channel-half_width
            flux_cubelet_end_medium = integrated_image_central_channel+half_width
            if flux_cubelet_start_medium<cube_start: flux_cubelet_start_medium=cube_start
            if flux_cubelet_end_medium>cube_end: flux_cubelet_end_medium=cube_end 
            flux_medium = np.append( flux_full[(channels>=flux_cubelet_start_medium)&(channels<flux_cubelet_start_central)],
                                   flux_full[(channels>=flux_cubelet_end_central)&(channels<flux_cubelet_end_medium)])
            
            spectral_SNR_1, spectral_SNR_average_1 = calculate_spectral_SNR(flux_long,flux_central)
            spectral_SNR_2, spectral_SNR_average_2 = calculate_spectral_SNR(flux_medium,flux_central)
            spectral_SNR = np.max([spectral_SNR_1,spectral_SNR_2])
            found_sources_dict['spectral_SNR'].values[source] = spectral_SNR
            
            if (spectral_SNR>SNR_spectrum_threshold and spectral_SNR_average_2>0.9*SNR_spectrum_threshold) or (spectral_SNR>SNR_spectrum_threshold and spectral_SNR_average_1>0.9*SNR_spectrum_threshold):
            # if (spectral_SNR>SNR_spectrum_threshold and spectral_SNR_average>0.9*SNR_spectrum_threshold):
                found_sources_dict['failed_spectral_SNR'].values[source] = 0
                
            else:
                found_sources_dict['failed_spectral_SNR'].values[source] = 1
                
            break
    return found_sources_dict
    
def associate_sources(found_sources_dict):
    for source in range(len(found_sources_dict)):
        beam_diameter = integrated_image_beam_diameter_array[int(found_sources_dict['int_im_number'].values[source]+chunk*int_image_load_number)]
        if found_sources_dict['associated_with'].values[source]=='no_match_yet':
            dist = np.sqrt( (found_sources_dict['x_coord'].values[source] - found_sources_dict['x_coord'].values)**2 +
                            (found_sources_dict['y_coord'].values[source] - found_sources_dict['y_coord'].values)**2)
            dist_channel = np.absolute(found_sources_dict['int_im_channel'].values[source]-found_sources_dict['int_im_channel'].values)
            match_id = np.arange(0,len(found_sources_dict))[(dist<beam_diameter/4)&(dist_channel<=int_image_length/2)]
            found_sources_dict['associated_with'].values[source] = found_sources_dict['ID'].values[match_id[np.argmax(found_sources_dict['spectral_SNR'].values[match_id])]]     
    return found_sources_dict

def associate_sources_final(found_sources_dict,max_dist_pix,max_dist_channel):

    for source in range(len(found_sources_dict)):
        dist = np.sqrt( (found_sources_dict['x_coord'].values[source] - found_sources_dict['x_coord'].values)**2 +
                        (found_sources_dict['y_coord'].values[source] - found_sources_dict['y_coord'].values)**2)
        dist_channel = np.absolute(found_sources_dict['gauss_x0'].values[source]-found_sources_dict['gauss_x0'].values)
        match_id = np.arange(0,len(found_sources_dict))[(dist<max_dist_pix)&(dist_channel<=max_dist_channel)]
        found_sources_dict['associated_with'].values[source] = found_sources_dict['ID'].values[match_id[np.argmax(found_sources_dict['spectral_SNR'].values[match_id])]]     
    return found_sources_dict

def local_background(x_coord,y_coord,integrated_image):
    
    ypix_left, ypix_right = int(y_coord-(bg_box_size/2)), int(y_coord+(bg_box_size/2))
    if ypix_left<0: ypix_left=0
    if ypix_right>integrated_image.shape[0]: ypix_right = integrated_image.shape[0]-1
        
    xpix_left, xpix_right = int(x_coord-(bg_box_size/2)), int(x_coord+(bg_box_size/2))
    if xpix_left<0: xpix_left=0
    if xpix_right>integrated_image.shape[1]: xpix_right = integrated_image.shape[1]-1
        
    image_around_source = integrated_image[ypix_left:ypix_right,xpix_left:xpix_right]
    mean, median, std = sigma_clipped_stats(image_around_source[~np.isnan(image_around_source)],sigma=3)

    return mean, std

def check_local_background(x_coord,y_coord,max_value,integrated_image,integrated_image_beam_diameter):
    bg_mean,bg_std = local_background(x_coord,y_coord,integrated_image)
    local_threshold = bg_mean + bg_std*SNR_integrated_image_threshold
    if bg_std==0:
        int_SNR=0
    else:
        int_SNR = (max_value-bg_mean)/bg_std

    passed_beamsize = check_beamsize(x_coord,y_coord,integrated_image,integrated_image_beam_diameter,local_threshold)
    
    return int_SNR,(max_value>local_threshold and passed_beamsize)

 
def check_beamsize(x_coord,y_coord,integrated_image,integrated_image_beam_diameter,local_threshold):    
    image_around_source = integrated_image[
    int(y_coord-integrated_image_beam_diameter):int(y_coord+integrated_image_beam_diameter),
    int(x_coord-integrated_image_beam_diameter):int(x_coord+integrated_image_beam_diameter)]
    
    rows, cols = image_around_source.shape[0], image_around_source.shape[1]
    y, x = np.ogrid[:rows, :cols]
    center_row, center_col = (rows - 1) / 2, (cols - 1) / 2
    distance = np.sqrt((x - center_col)**2 + (y - center_row)**2)
    radius = int(integrated_image_beam_diameter/2)-1
    if radius<2.5: radius=2.5
    image_around_source[distance>radius] = np.nan
    
    image_around_source = image_around_source[~np.isnan(image_around_source)]
    return len(image_around_source[image_around_source>local_threshold*0.5])>=len(image_around_source)*0.9

def check_source_on_integrated_image(found_sources_dict):

    for source in range(len(found_sources_dict)):
        x_coord,y_coord = found_sources_dict['x_coord'].values[source],found_sources_dict['y_coord'].values[source]
        int_im_number = found_sources_dict['int_im_number'].values[source]
        max_value,initial_SNR = found_sources_dict['int_max_value'].values[source],found_sources_dict['int_SNR'].values[source]
        
        integrated_image = integrated_image_list[int(int_im_number)]
        integrated_image_beam_diameter = integrated_image_beam_diameter_array[int(int_im_number+chunk*int_image_load_number)]
        int_median,int_popt_std = int_median_list[int(int_im_number)], int_popt_std_list[int(int_im_number)]
        
        while True:
            if initial_SNR<10:
                int_SNR, passed_local_threshold = check_local_background(x_coord,y_coord,max_value,integrated_image,
                                                                         integrated_image_beam_diameter)
                found_sources_dict['int_SNR'].values[source] = int_SNR
                if not passed_local_threshold:
                    found_sources_dict['failed_local_threshold'].values[source] = 1
                    break
                else: found_sources_dict['failed_local_threshold'].values[source] = 0
            else: found_sources_dict['failed_local_threshold'].values[source] = 0
            
            break
    return found_sources_dict

    
# find sources on the integrated images
def search_integrated_image(int_im_number, exclusion_zone_radius,median,popt_std):
    integrated_image = integrated_image_list[int(int_im_number)]
    image_radius = int(integrated_image.shape[0]/2)

    image_width = integrated_image.shape[1]
    image_height = integrated_image.shape[0]
    
    #cent_coord = [int(integrated_image.shape[0]/2),int(integrated_image.shape[1]/2)]
    
    #find the initial sources on the integrated image 
    threshold_center = median + (exp_function(0.02,*popt_std))[0]*SNR_integrated_image_threshold
    tbl = find_peaks(integrated_image, threshold_center, box_size=exclusion_zone_radius )
    found_peaks_x_coord_array = np.array(tbl['x_peak'])
    found_peaks_y_coord_array = np.array(tbl['y_peak'])
    found_peaks_max_value = np.array(tbl['peak_value'])

    # check if the source max values are greater than threshold * estimated background for the given radius      
    radiuses = np.sqrt( np.power(found_peaks_x_coord_array-cent_coord[0],2) +  np.power(found_peaks_y_coord_array-cent_coord[1],2)*(image_width/image_height))/(image_width/2)
   
    threshold_for_each_source = median + exp_function(radiuses,*popt_std)*SNR_integrated_image_threshold*0.8
    above_threshold_mask = (found_peaks_max_value>=threshold_for_each_source)

    # calculate SNR
    initial_SNR = (found_peaks_max_value-median)/exp_function(radiuses,*popt_std)

    # filter out sources below threshold
    found_peaks_x_coord_array = found_peaks_x_coord_array[above_threshold_mask]
    found_peaks_y_coord_array = found_peaks_y_coord_array[above_threshold_mask]
    found_peaks_max_value = found_peaks_max_value[above_threshold_mask]
    initial_SNR = initial_SNR[above_threshold_mask]
    threshold_for_each_source = threshold_for_each_source[above_threshold_mask]
    
    number_of_sources = len(found_peaks_x_coord_array)

    # generate dictionary with testing results for each source
    source_dict = pd.DataFrame(data = {'ID':np.char.add(np.char.add(np.char.add('ID_',np.array(np.ones(number_of_sources)*np.round(integrated_image_central_channel_array[int(int_im_number+chunk*int_image_load_number)],0),dtype='int').astype(str)), '_'),np.arange(number_of_sources).astype(str)),
                    'x_coord':found_peaks_x_coord_array,'y_coord':found_peaks_y_coord_array,
                   'int_im_number':np.ones(number_of_sources)*int_im_number,
                   'int_im_channel':np.ones(number_of_sources)*integrated_image_central_channel_array[int(int_im_number+chunk*int_image_load_number)],
                   'int_initial_SNR':initial_SNR,'int_max_value':found_peaks_max_value,'int_SNR':-99*np.ones(number_of_sources),
                   'spectral_SNR':-99*np.ones(number_of_sources),'rsqr':-99*np.ones(number_of_sources),
                   'gauss_x0':-99*np.ones(number_of_sources),'gauss_sigma':-99*np.ones(number_of_sources), 
                   'gauss_A':-99*np.ones(number_of_sources),'gauss_H':-99*np.ones(number_of_sources),
                  'failed_beamsize':-99*np.ones(number_of_sources), 'failed_local_threshold':-99*np.ones(number_of_sources),
                  'failed_persistence':-99*np.ones(number_of_sources),'failed_spectral_SNR':-99*np.ones(number_of_sources),
                  'failed_spectral_fit':-99*np.ones(number_of_sources),
                  'associated_with':np.full(number_of_sources,'no_match_yet')})

    # filter out sources too close to the edge of the cube
    source_dict = source_dict[(source_dict['x_coord'].values>1.5*exclusion_zone_radius)&
                              (source_dict['x_coord'].values<image_width-1.5*exclusion_zone_radius)&
                              (source_dict['y_coord'].values>1.5*exclusion_zone_radius)&
                              (source_dict['y_coord'].values<image_height-1.5*exclusion_zone_radius)]
        
    return source_dict

# functions for calculating stadard deviation of the background for different radiuses from the center of the image
def exp_function(x,a,b,c,d):
    y = a*np.exp(b*x+d)+c
    return y

def log_likelihood_exp(theta, x, y, yerr):
    model = exp_function(x,*theta)
    sigma2 = yerr**2
    return -0.5 * np.nansum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior_exp(theta):
    a,b,c,d = theta
    if b>0 and a>0 and 0<d<1:
        return 0.0
    return -np.inf

def log_probability_exp(theta, x, y, yerr):
    lp = log_prior_exp(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_exp(theta, x, y, yerr)
  
def fit_exp(x,y):
    pos = [y[0],5,y[0],0.1] + 1e-5 * np.random.randn(32, 4)
    
    nwalkers, ndim = pos.shape
    yerr=0.00001
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_exp, args=(x, y, yerr))
    sampler.run_mcmc(pos, 3000, progress=False);
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    fit_a = np.percentile(flat_samples[:, 0], [50])
    fit_b = np.percentile(flat_samples[:, 1], [50])
    fit_c = np.percentile(flat_samples[:, 2], [50])
    fit_d = np.percentile(flat_samples[:, 3], [50])

    return fit_a,fit_b,fit_c,fit_d
    
#calculate flux threshold function parameters for different radiuses for each integrated image
def std_and_median_from_radius_function(image):
    rin_array = np.array([0.2,0.4,0.6,0.8])
    rout_array = np.array([0.21,0.41,0.61,0.81])

    image_width = image.shape[1]
    image_height = image.shape[0]

    std_array=np.array([])
    median_array=np.array([])
    for r in range(4):

        a_in = image_width/2*rin_array[r]
        a_out = image_width/2*rout_array[r]
        b_out = image_height/2*rout_array[r]
        
        #aperture_annulus = CircularAnnulus(cent_coord, rin,rout)
        aperture_annulus = EllipticalAnnulus(cent_coord, a_in = a_in, a_out = a_out, b_out = b_out)
        mask= aperture_annulus.to_mask(method='center')
        image_slice = mask.multiply(image,fill_value=np.nan)
        mean, median, std = sigma_clipped_stats(image_slice[~np.isnan(image_slice)], sigma=3.0)
        
        std_array = np.append(std_array,std)
        median_array = np.append(median_array,median)
    popt_std_1,popt_std_2,popt_std_3,popt_std_4 = fit_exp(rout_array, std_array)
    median = np.mean(median_array)
    return np.array([popt_std_1,popt_std_2,popt_std_3,popt_std_4]), median


def integrate_image(int_im):
    #get slab out of the cube
    if int_im == cube_slab_number-1: end_channel = cube_end
    else: end_channel = cube_start + slab_half_length*(int_im+2)
    cube_slab = np.array(data_cube.unmasked_data[ slab_half_length*int_im+cube_start : end_channel, :, :],dtype=np.float32)
    # integrate the slab
    integrated_image = np.nansum(np.array(cube_slab),0)
    return integrated_image


def source_finder_func(data_file, path_to_results,
                       SNR_integrated_image_threshold_input, SNR_channel_frame_threshold_input, 
                       signal_persistence_threshold_input,
                       SNR_spectrum_threshold_input, rsqr_threshold_input,
                       int_image_length_input,int_image_load_number_input,
                       cube_start_input,cube_end_input,beam,output_test_history,
                       bg_box_size_input,max_dist_pix_input,max_dist_channel_input ):

    
    global  SNR_integrated_image_threshold, SNR_channel_frame_threshold, signal_persistence_threshold,\
            SNR_spectrum_threshold, rsqr_threshold,\
            int_image_length,int_image_load_number,\
            cube_start,cube_end, bg_box_size,\
            data_cube, cube_width, cube_height, image_radius, cent_coord, cube_channel_range,\
            integrated_image_beam_diameter_array, integrated_image_central_channel_array,\
            cube_slab_number, slab_half_length,\
            integrated_image_list, int_popt_std_list, int_median_list, chunk,\
            max_dist_pix,max_dist_channel


    SNR_integrated_image_threshold, SNR_channel_frame_threshold, signal_persistence_threshold = SNR_integrated_image_threshold_input, SNR_channel_frame_threshold_input, signal_persistence_threshold_input
    SNR_spectrum_threshold, rsqr_threshold = SNR_spectrum_threshold_input, rsqr_threshold_input
    int_image_length,int_image_load_number = int_image_length_input,int_image_load_number_input
    cube_start, cube_end = cube_start_input, cube_end_input
    bg_box_size = bg_box_size_input
    max_dist_pix,max_dist_channel = max_dist_pix_input,max_dist_channel_input
    
    core_number = multiprocessing.cpu_count()
    
    # get cube data
    data_cube = SpectralCube.read(data_file)
    # wcs
    data_cube_wcs = data_cube.wcs
    # cube dimensions
    cube_channel_range = data_cube.shape[0]
    cube_width = data_cube.shape[2]
    cube_height = data_cube.shape[1]
    cent_coord = [int(data_cube.shape[2]/2),int(data_cube.shape[1]/2)]
    image_radius = int(data_cube.shape[1]/2)
    if cube_end==None: cube_end = cube_channel_range
    
    # cube synthesised beam diameter
    if beam==None:
        try:
            beams_pixel_diameter_array = data_cube.beams.major.value/3600/np.abs(data_cube_wcs.wcs.cdelt[0])
        except: 
            try:
                beams_pixel_diameter_array = np.ones(len(data_cube.beam.value)) * data_cube.beam.major.value/3600/np.abs(data_cube_wcs.wcs.cdelt[0])
            except: 
                print('SpectralCube failed to read the beam data, use "beam" argument to specify the beam diameter manually.')
                print(beams_pixel_diameter_array)
                
    else:
        beams_pixel_diameter_array = np.ones(cube_channel_range)*beam

    # create folder for results
    path_to_results_dir = path_to_results+'/source_finding_results_1'
    i=1
    while os.path.exists(path_to_results_dir):
        i=i+1
        path_to_results_dir = path_to_results+'/source_finding_results_%s'%(i)
    os.makedirs(path_to_results_dir)
        
    # number of integrated images
    slab_half_length = int(int_image_length/2)
    cube_slab_number = int(np.ceil((cube_end-cube_start)/int_image_length)*2-1)
    
    # number of chunks
    chunk_number = int(np.ceil(cube_slab_number/int_image_load_number))

    print('\n')
    print('for following inputs:')
    print('data_cube: %s'%(data_file))
    print('path_to_results: %s'%(path_to_results))
    print('SNR_integ: %s'%(SNR_integrated_image_threshold_input))
    print('SNR_channel: %s'%(SNR_channel_frame_threshold_input))
    print('channel_min_len: %s'%(signal_persistence_threshold_input))
    print('SNR_spec: %s'%(SNR_spectrum_threshold_input))
    print('rsqr_min: %s'%(rsqr_threshold_input))
    print('int_image_len: %s'%(int_image_length_input))
    print('int_image_load_no: %s'%(int_image_load_number_input))
    print('channel_start: %s'%(cube_start_input))
    print('channel_end: %s'%(cube_end_input))
    print('beam: %s'%(beam))
    print('bg_box_size: %s'%(bg_box_size_input))
    print('max_dist_pix: %s'%(max_dist_pix_input))
    print('max_dist_channel: %s'%(max_dist_channel_input))
    print('test_hist: %s'%(output_test_history))
    print('\n')
    
    print('Chosen width of frequency slab to be integrated: %s channels'%(int_image_length))
    print('Chosen number of channels to perform sourcefinding on: %s'%(cube_end-cube_start))
    print('Therefore (including dithered images), the number of integrated images to process: %s'%(cube_slab_number))
    print('Chosen number of integrated images to process at the same time in a chunk: %s'%(int_image_load_number))
    print('Therefore number of chunks: %s'%(chunk_number))
    print('\n')
    
    # get median beamsizes and central channel numbers for integrated images       
    integrated_image_beam_diameter_array = np.array( [np.median(beams_pixel_diameter_array[slab_half_length*int_im+cube_start : (cube_start + slab_half_length*(int_im+2) )]) for int_im in range(cube_slab_number)])
    integrated_image_central_channel_array = np.array( [int((slab_half_length*int_im+cube_start + (cube_start + slab_half_length*(int_im+2) ))/2) for int_im in range(cube_slab_number)])
    integrated_image_beam_diameter_array[-1] = np.median(beams_pixel_diameter_array[slab_half_length*(cube_slab_number-1)+cube_start : cube_end])
    integrated_image_central_channel_array[-1] = int((slab_half_length*(cube_slab_number-1)+cube_start + cube_end)/2)
    
    for chunk in range(0,chunk_number):
        print('CHUNK %s out of %s'%(chunk+1,chunk_number))
        start_time_slab = time.time()
        
    
        # integrate images
        print('integrating images')
        int_im_start = chunk*int_image_load_number
        int_im_end = (chunk+1)*int_image_load_number
        if chunk == chunk_number-1: int_im_end = cube_slab_number
        integrated_image_list = p_map(integrate_image,[int_im for int_im in range(int_im_start,int_im_end)])
    
        # estimate background
        print('estimating background')
        int_bg_estimated = p_map(std_and_median_from_radius_function,integrated_image_list)
        int_popt_std_list = [ int_bg_estimated[i][0] for i in range(int_im_end-int_im_start) ]
        int_median_list = [ int_bg_estimated[i][1] for i in range(int_im_end-int_im_start) ]
        
        # find sources
        print('finding sources')
        exclusion_zone_radius = integrated_image_beam_diameter_array[np.arange(int_im_start,int_im_end)]
        searching_results = p_map(search_integrated_image, np.arange(0,len(exclusion_zone_radius)), exclusion_zone_radius,int_median_list,int_popt_std_list)
     
        all_sources_dict = pd.concat(searching_results)
        n_sources = len(all_sources_dict)
        
        # checking sources on the integrated image
        print('checking %s sources on the integrated images using %s cores'%(n_sources,multiprocessing.cpu_count()))
        
    
        integrated_image_check_results = p_map(check_source_on_integrated_image,
                                               [all_sources_dict[i*int((n_sources)/core_number+1):(i+1)*int((n_sources)/core_number+1)] for i in range(core_number)])
        
        all_sources_dict = pd.concat(integrated_image_check_results)
        
        del integrated_image_list
        del integrated_image_check_results
        del searching_results
        gc.collect()
    
        
        # checking sources in frequency space
        filter_passed = (np.array(all_sources_dict['failed_local_threshold'])==0)
        n_sources = len(all_sources_dict[filter_passed])
        print('checking %s sources in frequency space using %s cores'%(len(all_sources_dict[filter_passed ]),core_number))
        spectral_check_results = p_map(check_source_in_spectral,
                                      [all_sources_dict[filter_passed][i*int((n_sources)/core_number+1):(i+1)*int((n_sources)/core_number+1)] for i in range(core_number)])
        
        if len(spectral_check_results)>0: all_sources_dict = pd.concat([all_sources_dict[~filter_passed],pd.concat(spectral_check_results)])
    
        # associate passing sources in neighbouring slabs
        filter_passed = (np.array(all_sources_dict['failed_spectral_SNR'])==0)
        n_sources = len(all_sources_dict[filter_passed])
        print('associating %s sources in overlapping frequency slabs'%(n_sources))  
        associated_sources_results = p_map(associate_sources,
                                      [all_sources_dict[filter_passed][i*int((n_sources)/core_number+1):(i+1)*int((n_sources)/core_number+1)] for i in range(core_number)])
        if len(associated_sources_results)>0: all_sources_dict = pd.concat([all_sources_dict[~filter_passed],pd.concat(associated_sources_results)])
        
        # check source by fitting gaussian function
        filter_passed_associated = (np.array(all_sources_dict['failed_spectral_SNR'])==0)&(all_sources_dict['ID'].values==all_sources_dict['associated_with'].values)
        n_sources = len(all_sources_dict[filter_passed_associated])
        print('fitting Gaussian function for %s sources using %s cores'%(n_sources,core_number))  
        gaussian_fit_check_results = p_map(check_source_gaussian_fit,
                                      [all_sources_dict[filter_passed_associated][i:(i+1)] for i in range(len(all_sources_dict[filter_passed_associated]))])
        if len(gaussian_fit_check_results)>0: all_sources_dict = pd.concat([all_sources_dict[~filter_passed_associated],pd.concat(gaussian_fit_check_results)])
        
        # get sky coordinates and frequency for each source
        ra_array,dec_array = sky_coord_in_deg_from_pix(all_sources_dict['x_coord'].values,all_sources_dict['y_coord'].values,data_cube_wcs)
        frequency_array = channel_to_frequency(all_sources_dict['int_im_channel'].values,data_cube_wcs)
    
        all_sources_dict['RA_deg'],all_sources_dict['Dec_deg'] = ra_array,dec_array
        all_sources_dict['frequency_Hz'] = frequency_array
    
        
        # save the results
        if output_test_history:
            if chunk==0: all_sources_dict.to_csv(path_to_results_dir+'/source_testing_history.csv',index=False,mode='a')
            else: all_sources_dict.to_csv(path_to_results_dir+'/source_testing_history.csv',index=False,mode='a',header=False)
            print('Source testing history saved to %s'%(path_to_results_dir+'/source_testing_history.csv')) 
    
        frequency_array = channel_to_frequency(all_sources_dict['gauss_x0'].values,data_cube_wcs)
        all_sources_dict['frequency_Hz'] = frequency_array
        all_sources_dict = all_sources_dict[all_sources_dict['failed_spectral_fit'].values==0]
        if chunk==0: all_sources_dict.to_csv(path_to_results_dir+'/found_sources_all.csv',index=False,mode='a')
        else: all_sources_dict.to_csv(path_to_results_dir+'/found_sources_all.csv',index=False,mode='a',header=False)
        print('Results saved to %s'%(path_to_results_dir+'/found_sources_all.csv')) 
        print('Found %s sources'%(len(all_sources_dict)))
        
        del spectral_check_results
        del gaussian_fit_check_results
        del all_sources_dict
        gc.collect()
        
        print('time for chunk: %s seconds'%(round(time.time()-start_time_slab,2)))
        print('estimated time left: %s minutes'%round(((time.time()-start_time_slab)*(int(cube_slab_number/int_image_load_number)-chunk))/60,2))
        print('\n')
        
    all_sources_dict=pd.read_csv(path_to_results_dir+'/found_sources_all.csv')
    all_sources_dict['redshift'] = (1420405751-all_sources_dict['frequency_Hz'].values)/all_sources_dict['frequency_Hz'].values
    all_sources_dict['z_channel'] = np.array(all_sources_dict['gauss_x0'].values).astype(int)
    
    all_sources_dict = associate_sources_final(all_sources_dict,max_dist_pix,max_dist_channel)
    all_sources_dict.to_csv(path_to_results_dir+'/found_sources_all.csv',index=False,columns=['ID','x_coord','y_coord','z_channel','RA_deg','Dec_deg','frequency_Hz','redshift','int_SNR','spectral_SNR','rsqr','gauss_x0','gauss_sigma','gauss_A','gauss_H','associated_with','int_im_channel'])
    
    filter_associated = (all_sources_dict['ID'].values==all_sources_dict['associated_with'].values)
    all_sources_dict = all_sources_dict[filter_associated]
    
    all_sources_dict = all_sources_dict.sort_values(by='spectral_SNR',ascending=False)
    all_sources_dict.to_csv(path_to_results_dir+'/found_sources_associated.csv',index=False,columns=['ID','x_coord','y_coord','z_channel','RA_deg','Dec_deg','frequency_Hz','redshift','int_SNR','spectral_SNR','rsqr','gauss_x0','gauss_sigma','gauss_A','gauss_H','int_im_channel'])
    print('Found %s sources'%(len(all_sources_dict)))
    all_sources_dict = pd.read_csv(path_to_results_dir+'/found_sources_associated.csv')
    return all_sources_dict

