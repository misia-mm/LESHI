import numpy as np
import pandas as pd
import cv2
import copy
import pickle
import emcee
import glob
import copy

from scipy import special
import matplotlib.pyplot as plt
from astropy.io import fits  # We use fits to open the actual data file
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.stats import sigma_clipped_stats, sigma_clip
from spectral_cube import SpectralCube
from p_tqdm import p_map
import multiprocessing

from scipy.signal import argrelextrema
import warnings

# Convert RuntimeWarning into an exception
warnings.filterwarnings("error", category=RuntimeWarning)


def contour_sky_coord_in_deg_from_pix(contour,wcs):
    contour_sky = contour.copy()
    for row in range(contour.shape[0]):
        ra,dec = sky_coord_in_deg_from_pix(contour[row][0],contour[row][1],wcs)
        contour_sky[row][0],contour_sky[row][1] = float(ra),float(dec)
    return contour_sky
    
def sky_coord_in_deg_from_pix(x_pix_coord,y_pix_coord,wcs):
    skycoord = SkyCoord.from_pixel(xp=x_pix_coord,yp=y_pix_coord, wcs=wcs)
    RA = skycoord.ra.degree
    DEC = skycoord.dec.degree
    return RA, DEC

def frequency_to_channel(channel,wcs):
    wave0,wavedelta,channel0 = wcs.wcs.crval[2],wcs.wcs.cdelt[2],wcs.wcs.crpix[2]
    channel = (frequency-wave0)/wavedelta+channel0
    return frequency
    
def channel_to_frequency(channel,wcs):
    wave0,wavedelta,channel0 = wcs.wcs.crval[2],wcs.wcs.cdelt[2],wcs.wcs.crpix[2]
    frequency = wave0+wavedelta*(channel-channel0)
    return frequency

def contour_center(contour_points):
    contour_points = contour_points.T
    x_center = int(np.mean(contour_points[0]))
    y_center = int(np.mean(contour_points[1]))
    return x_center, y_center

def contour_diameter(contour_points):
    max_distance = 0
    for point in contour_points:
        distances = np.sqrt( (point[0]-contour_points.T[0])**2 + (point[1]-contour_points.T[1])**2  )
        if max_distance < np.max(distances) : max_distance = np.max(distances)
    return max_distance
    
def contour_contains_point(contour_points, x_coord,y_coord,window_width):
    x_coord,y_coord = int(x_coord),int(y_coord)
    contour = np.array([contour_points], dtype=np.int32)
    mask = np.zeros([window_width,window_width], dtype=bool)
    temp_mask = np.zeros([window_width,window_width], dtype=np.uint8)
    cv2.drawContours(temp_mask, contour, contourIdx=-1, color=1, thickness=cv2.FILLED)
    mask[temp_mask == 1] = True

    return mask[y_coord][x_coord] 

def contour_to_mask(contour_points,shape):
    contour = np.array([contour_points], dtype=np.int32)
    mask = np.zeros(shape, dtype=bool)
    temp_mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(temp_mask, contour, contourIdx=-1, color=1, thickness=cv2.FILLED)
    cv2.drawContours(temp_mask, contour, contourIdx=-1, color=1, thickness=1)
    mask[temp_mask == 1] = True
    return mask


def moment_0_map(xpix,ypix,zchannel_left,zchannel_right,map_size,cube,cube_data,cube_channel_length):
    
    if zchannel_left<0:zchannel_left=0
    if zchannel_right>cube_channel_length: zchannel_right = -1
    zchannel_left, zchannel_right = int(zchannel_left), int(zchannel_right)
    
    ypix_left, ypix_right = int(ypix-(map_size/2)), int(ypix+(map_size/2))
    if ypix_left<0: ypix_left=0
    if ypix_right>cube_data.shape[1]: ypix_right = -1
        
    xpix_left, xpix_right = int(xpix-(map_size/2)), int(xpix+(map_size/2))
    if xpix_left<0: xpix_left=0
    if xpix_right>cube_data.shape[2]: xpix_right = -1

    cubelet=cube[zchannel_left:zchannel_right,ypix_left:ypix_right,xpix_left:xpix_right]
    
    wcs_moment_0 = 0
    moment_0=np.nansum(np.array(cubelet.unmasked_data[:,:,:]),0)
    mean, median, std = sigma_clipped_stats(moment_0, sigma=3,maxiters=None)
    
    
    return moment_0, mean, median, std, wcs_moment_0

def read_in_radio_file(radio_file):
    cube=SpectralCube.read(radio_file)
    #cube.mask_out_bad_beams
    cube_data = cube.unmasked_data[:,:,:].value
    wcs_cube = cube.wcs
    cube_channel_length = cube_data.shape[0]
    dpix=np.abs(wcs_cube.wcs.cdelt[0])*3600
    return cube, cube_data, wcs_cube, cube_channel_length,dpix


def calculate_velocity_widths(x,xp, xe, a, w, b, c, C, yerr):
    x_cont = np.arange(np.nanmin(x),np.nanmax(x),0.1)
    model_cont = busy(x_cont, xp, xe, a, w, b, c, C)
    
    first_maximum, second_maximum, average_maximum = find_peaks_of_busy_function(x, xp, xe, a, w, b, c, C, yerr)
    
    half_signal_channels = x_cont[model_cont>=( (average_maximum-C)*0.5 + C )]
    W50_channels = (half_signal_channels[-1]-half_signal_channels[0])+0.1
    x0 = int((np.nanmin(half_signal_channels) + np.nanmax(half_signal_channels))/2)

    if first_maximum>second_maximum:
        signal_channels = x_cont[model_cont>( (first_maximum-C)*0.05 + C )]
        W100_channels = (signal_channels[-1]-signal_channels[0])+0.1
    else:
        signal_channels = x_cont[model_cont>( (second_maximum-C)*0.05 + C )]
        W100_channels = (signal_channels[-1]-signal_channels[0])+0.1
            
    
    
   
    return W50_channels, W100_channels, x0


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

def fit_busy(x,y,source):
    # starting parameters
    if source_data_table['fit_xp'].values[source]>cube_channel_length-2:
        source_data_table['fit_xp'].values[source] = cube_channel_length-3
    if source_data_table['fit_xe'].values[source]>cube_channel_length-2:
        source_data_table['fit_xe'].values[source] = cube_channel_length-3
        
    pos = [source_data_table['fit_xp'].values[source],source_data_table['fit_xe'].values[source],
           source_data_table['fit_a'].values[source], source_data_table['fit_w'].values[source],
           source_data_table['fit_b'].values[source],source_data_table['fit_c'].values[source],
           source_data_table['fit_C'].values[source]] +1e-4 * np.random.randn(32, 7)
    pos = np.abs(pos)
    nwalkers, ndim = pos.shape
    #print(pos)
    mean, median, yerr = sigma_clipped_stats(y, sigma=3,maxiters=None)
    #yerr=np.abs(np.max(y)-np.min(y))*0.01
    if yerr==0: yerr=0.0001
    try:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_busy, args=(x, y, yerr))
        sampler.run_mcmc(pos, 2000, progress=False)
    
        flat_samples = sampler.get_chain(discard=200, thin=10, flat=True)
    except RuntimeWarning:
        print(pos,cube_channel_length,source_data_table['fit_xp'].values[source])

    fit_xp = np.percentile(flat_samples[:, 0], [50])[0]
    fit_xe = np.percentile(flat_samples[:, 1], [50])[0]
    fit_a = np.percentile(flat_samples[:, 2], [50])[0]
    fit_w = np.percentile(flat_samples[:, 3], [50])[0]
    fit_b = np.percentile(flat_samples[:, 4], [50])[0]
    fit_c = np.percentile(flat_samples[:, 5], [50])[0]
    fit_C = np.percentile(flat_samples[:, 6], [50])[0]
    

    W50_channels, W100_channels, x0_channels = calculate_velocity_widths(x,fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C, yerr)
    half_width = (W100_channels/2)

    
    fit_y =busy(x, fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C)

    rsqr = 1- (np.sum((y-fit_y)**2))/(np.sum((y-np.sum(y)/len(y))**2))
    # print('Rsqr: ',rsqr)
    return rsqr, fit_y, fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C, x0_channels, half_width

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
            W50_channels, W100_channels, x0 = calculate_velocity_widths(x,xp, xe, a, w, b, c, C, yerr)
            W50_sampled_widths=np.append(W50_sampled_widths,W50_channels)
            W100_sampled_widths=np.append(W100_sampled_widths,W100_channels)
            x0_sampled = np.append(x0_sampled,x0)
        except: pass
    

    W50_channels, W100_channels, x0_channels = calculate_velocity_widths(x,fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C, yerr)
    
    half_width = (W100_channels/2)
    W50_channels_error = (np.percentile(W50_sampled_widths[1000:],84) - np.percentile(W50_sampled_widths[1000:],16))/2
    W100_channels_error = (np.percentile(W100_sampled_widths[1000:],84) - np.percentile(W100_sampled_widths[1000:],16))/2
    x0_channels_error = (np.percentile(x0_sampled[1000:],84) - np.percentile(x0_sampled[1000:],16))/2

    return W50_channels, W50_channels_error, W100_channels, W100_channels_error, x0_channels, x0_channels_error, half_width



def find_source_extent(source_data_table_input):
    global source_data_table
    source_data_table = source_data_table_input

    source_data_table['half_width'] = np.ones(len(source_data_table))*np.array(source_data_table['gauss_sigma'].values)*2
    source_data_table['contour_flag'] = np.ones(len(source_data_table))
    source_data_table['contour_diameter_arc'] = np.ones(len(source_data_table))*beam_arc
    
    source_data_table['x0_busy'], source_data_table['x0_busy_error'] = np.ones(len(source_data_table))*(-99), np.ones(len(source_data_table))*(-99)
    source_data_table['W100'], source_data_table['W100_error'] = np.ones(len(source_data_table))*(-99), np.ones(len(source_data_table))*(-99)
    source_data_table['W50'], source_data_table['W50_error'] = np.ones(len(source_data_table))*(-99), np.ones(len(source_data_table))*(-99)
    
    

    source_data_table['fit_xp'] = np.ones(len(source_data_table))*np.array(source_data_table['gauss_x0'].values)
    source_data_table['fit_xe'] = np.ones(len(source_data_table))*np.array(source_data_table['gauss_x0'].values) 
    source_data_table['fit_a'] = np.ones(len(source_data_table))
    source_data_table['fit_w']= np.ones(len(source_data_table))*10 
    source_data_table['fit_b'] = np.ones(len(source_data_table))*np.sqrt(np.pi/2)/np.array(source_data_table['gauss_sigma'].values)
    source_data_table['fit_b'].values[source_data_table['fit_b'].values>=100]=99
    source_data_table['fit_c'] = np.ones(len(source_data_table))*1e-4*1.1
    source_data_table['fit_n'] = np.ones(len(source_data_table))*2
    source_data_table['fit_C'] = np.ones(len(source_data_table))*np.array(source_data_table['gauss_H'].values) 
    

    for source in range(len(source_data_table)):
        # data coordinates
        try: x_pix, y_pix = source_data_table['x_coord'].values[source], source_data_table['y_coord'].values[source]
        except: 
            ra, dec = source_data_table['RA_deg'].values[source], source_data_table['Dec_deg'].values[source]
            sky_coords =SkyCoord(ra, dec, unit="deg",frame="fk5")
            x_pix, y_pix = skycoord_to_pixel(SkyCoord(ra,dec, frame="fk5", unit="deg"),wcs_cube)
        try:z_channel, sigma = source_data_table['int_im_channel'].values[source], (source_data_table['gauss_sigma'].values[source])
        except: 
            z_channel = frequency_to_channel(source_data_table['frequency_Hz'].values[source],wcs_cube)
            sigma = 5
        x_pix,y_pix,z_channel  = int(x_pix),int(y_pix),int(z_channel)
    
        wavelength_range_left, wavelength_range_right = z_channel-200, z_channel+200
        if wavelength_range_left<0: wavelength_range_left=2
        if wavelength_range_right>= cube_data.shape[0] : wavelength_range_right = cube_data.shape[0] -2
        wavelength_range_array = np.arange(wavelength_range_left,wavelength_range_right,1,dtype=int)
    
        loop = 0
        fail_flag = 0
        
        contour_size = [-99,-99,-99,-99]
        x0=z_channel
        map_size_pix = initial_map_size_pix
        half_width=int(2*sigma)
        if half_width<1: half_width=1
        while True:
            
            # break the loop if the value does not converge after too many tries and keep initial values
            loop = loop+1
            if loop>20:
                
                fail_flag=1
                source_data_table['contour_flag'].values[source] = fail_flag
                #print('loop fail for contour %s - kept last values of parameters'%(source_data_table['ID'].values[source]))
                break
                
            contour_found=False
            if half_width<1:half_width=1
            moment_0, mean, median, std, wcs_moment_0 = moment_0_map(x_pix,y_pix,int(x0-half_width),int(x0+half_width),map_size_pix,cube,cube_data,cube_channel_length)
            moment_0 = np.nan_to_num(moment_0)
            threshold_value = median + 3*std
            _, binary_image = cv2.threshold(moment_0.astype(np.float32), threshold_value, 1, cv2.THRESH_BINARY)
        
            # Convert to uint8 for OpenCV (0 or 255)
            binary_image = (binary_image * 255).astype(np.uint8)
            contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            final_contours=[]
            for contour in contours:
                final_contour = []
                for point in contour:
                    final_contour.append(point[0])
                final_contour.append(final_contour[0])
                final_contour = np.array(final_contour)
                if contour_contains_point(final_contour, int(moment_0.shape[0]/2),int(moment_0.shape[0]/2),moment_0.shape[0]): 
                    source_contour = copy.deepcopy(final_contour)
                    contour_found=True
            if not contour_found:
                fail_flag=1
                source_data_table['contour_flag'].values[source] = fail_flag
                #print('contour fail - tried again')
                half_width=2*sigma
                x0=z_channel
                continue
            
            mask = contour_to_mask(source_contour,moment_0.shape)
            flux=np.zeros(len(wavelength_range_array))
            for i, wavelength in enumerate(wavelength_range_array):     
                x_pix_left, x_pix_right = int(x_pix-(map_size_pix/2)),int(x_pix+(map_size_pix/2))
                y_pix_left, y_pix_right = int(y_pix-(map_size_pix/2)),int(y_pix+(map_size_pix/2))

                if x_pix_left<0: x_pix_left=0
                if y_pix_left<0: y_pix_left=0
                if x_pix_right>cube_data.shape[2]: x_pix_right=cube_data.shape[2]-1
                if y_pix_right>cube_data.shape[1]: y_pix_right=cube_data.shape[1]-1
                image_slice = cube_data[wavelength,y_pix_left:y_pix_right,x_pix_left:x_pix_right]

        
                # apply mask and sum up the flux in channel image
                try:flux[i] = np.nansum(image_slice[mask])
                except: 
                    print(mask.shape,image_slice.shape,moment_0.shape)
                    print(wavelength,y_pix_left,y_pix_right,x_pix_left,x_pix_right)
            
            rsqr, fit_y, fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C, x0, half_width = fit_busy(wavelength_range_array,flux,source)
    
            # update the function parameters
            source_data_table['fit_xp'].values[source], source_data_table['fit_xe'].values[source] = fit_xp, fit_xe
            source_data_table['fit_a'].values[source], source_data_table['fit_w'].values[source] = fit_a, fit_w
            source_data_table['fit_b'].values[source] = fit_b
            source_data_table['fit_c'].values[source], source_data_table['fit_C'].values[source] = fit_c, fit_C
    
            # break the loop if the value converges
            if contour_size[-1]*0.95<len(mask[mask==True])<contour_size[-1]*1.05 and contour_size[-2]*0.95<len(mask[mask==True])<contour_size[-2]*1.05 and contour_size[-3]*0.95<len(mask[mask==True])<contour_size[-3]*1.05:
                fail_flag=0
                source_data_table['contour_flag'].values[source] = 0
                break
            if contour_size[-2]*0.95<len(mask[mask==True])<contour_size[-2]*1.05 and contour_size[-4]*0.95<len(mask[mask==True])<contour_size[-4]*1.05:
                fail_flag=0
                source_data_table['contour_flag'].values[source] = 0
                break
            contour_size.append(len(mask[mask==True]))
            #print(contour_size)
            
            # update the size, center and range of the cubelet
            x_center_contour,y_center_contour = contour_center(source_contour)
            x_pix = int(x_center_contour + x_pix - int(map_size_pix/2))
            y_pix = int(y_center_contour + y_pix - int(map_size_pix/2))
            
            map_size_pix = contour_diameter(source_contour) * 5
            if map_size_pix<initial_map_size_pix: map_size_pix = initial_map_size_pix

            spectrum_half_length = 6*half_width
            if spectrum_half_length<200: spectrum_half_length=200
            wavelength_range_left, wavelength_range_right = x0-spectrum_half_length, x0+spectrum_half_length
            if wavelength_range_left<0: wavelength_range_left=2
            if wavelength_range_right> cube_data.shape[0]-1 : wavelength_range_right = cube_data.shape[0] -1
            wavelength_range_array = np.arange(wavelength_range_left,wavelength_range_right,1,dtype=int)

            if x0> cube_data.shape[0]-1: x0 = cube_data.shape[0]-1
            if half_width<1: half_width=1

        if fail_flag==0:

            # calculate velocity widths
            W50_channels, W50_channels_error, W100_channels, W100_channels_error, x0_channels, x0_channels_error, half_width = fit_busy_velocity_width(wavelength_range_array,flux,source)

            # update parameters for each source

            new_frequency = channel_to_frequency(x0,wcs_cube)
            new_z = (1420405000-new_frequency)/new_frequency

            source_data_table['x0_busy'].values[source],source_data_table['half_width'].values[source] =  x0, half_width
            source_data_table['x0_busy_error'].values[source] = x0_channels_error
            source_data_table['W50'].values[source], source_data_table['W100'].values[source] = W50_channels, W100_channels
            source_data_table['W50_error'].values[source], source_data_table['W100_error'].values[source] = W50_channels_error, W100_channels_error
            source_data_table['frequency_Hz'].values[source],source_data_table['redshift'].values[source] = new_frequency, new_z
            source_data_table['contour_diameter_arc'].values[source] = contour_diameter(source_contour)*dpix

            source_data_table['RA_deg'].values[source],source_data_table['Dec_deg'].values[source] = sky_coord_in_deg_from_pix(x_pix,y_pix,wcs_cube)
            source_data_table['x_coord'].values[source], source_data_table['y_coord'].values[source] = x_pix, y_pix
            source_data_table['z_channel'].values[source] = int(x0)
            
        else:
            source_data_table['fit_xp'].values[source] = (-99)
            source_data_table['fit_xe'].values[source] = (-99)
            source_data_table['fit_a'].values[source] = (-99)
            source_data_table['fit_w'].values[source] = (-99)
            source_data_table['fit_b'].values[source] = (-99)
            source_data_table['fit_c'].values[source] = (-99)
            source_data_table['fit_n'].values[source] = (-99)
            source_data_table['fit_C'].values[source] = (-99)

    source_data_table['half_width'].values[source_data_table['half_width'].values<1] = 1
    source_data_table['contour_diameter_arc'].values[source_data_table['contour_diameter_arc'].values<beam_arc] = beam_arc
    
    return source_data_table

def associate_sources_final(found_sources_dict):
    found_sources_dict['associated_with'] = np.full(len(found_sources_dict),'no_match_yet')
    for source in range(len(found_sources_dict)):
        dist = np.sqrt( (found_sources_dict['x_coord'].values[source] - found_sources_dict['x_coord'].values)**2 +
                        (found_sources_dict['y_coord'].values[source] - found_sources_dict['y_coord'].values)**2)
        dist_channel = np.absolute(found_sources_dict['x0_busy'].values[source]-found_sources_dict['x0_busy'].values)
        match_id = np.arange(0,len(found_sources_dict))[(dist<found_sources_dict['contour_diameter_arc'].values[source]/3*dpix)&(dist_channel<=found_sources_dict['half_width'].values[source])]
        found_sources_dict['associated_with'].values[source] = found_sources_dict['ID'].values[match_id[np.argmax(found_sources_dict['int_SNR'].values[match_id])]]     
    return found_sources_dict

def source_extent_script(data_table, path_to_radio_file, initial_map_size_pix_input, beam_arc_input):
    global cube, cube_data, wcs_cube, cube_channel_length, initial_map_size_pix, beam_arc, dpix
    initial_map_size_pix = initial_map_size_pix_input
    cube, cube_data, wcs_cube, cube_channel_length, dpix=read_in_radio_file(path_to_radio_file)
    beam_arc = beam_arc_input
    # core_number = multiprocessing.cpu_count()
    # n_sources = len(data_table)
    # core_number = n_sources
    source_extent_results = p_map(find_source_extent,
                                      [data_table[i:(i+1)] for i in range(len(data_table))])
    all_sources_dict = pd.concat(source_extent_results)
    all_sources_dict = associate_sources_final(all_sources_dict)
    filter_associated = (all_sources_dict['ID'].values==all_sources_dict['associated_with'].values)
    all_sources_dict = all_sources_dict[filter_associated]
    
    all_sources_dict.to_csv('found_sources_extent.csv',index=False)
    
    return all_sources_dict

    
