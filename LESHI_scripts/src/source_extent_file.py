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
    wave0,wavedelta,channel0 = wcs.wcs.crval[2],wcs.wcs.cdelt[2],wcs.wcs.crpix[2]
    frequency = wave0+wavedelta*(channel-channel0)
    return frequency

def circular_contour(x_cen,y_cent,radius):
    thetas = np.linspace(0,2*np.pi,50)
    x_array = radius*np.cos(thetas)
    y_array = radius*np.sin(thetas)
    contour = np.array([x_array,y_array]).T
    contour = np.append(contour,[ [x_array[0],y_array[0]] ],axis=0)
    return contour

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
    
def contour_contains_point(contour_points, x_coord,y_coord,shape):
    x_coord,y_coord = int(x_coord),int(y_coord)
    contour = np.array([contour_points], dtype=np.int32)
    mask = np.zeros(shape, dtype=bool)
    temp_mask = np.zeros(shape, dtype=np.uint8)
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
    
    wcs_moment_0 = cubelet.wcs
    moment_0=np.nansum(np.array(cubelet.unmasked_data[:,:,:]),0)
    mean, median, std = sigma_clipped_stats(moment_0, sigma=3,maxiters=None)
    
    
    return moment_0, mean, median, std, wcs_moment_0

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
    if a>0 and w>=3 and 10>b>=0 and c>=0 and 0<xp<cube_channel_length-1 and 0<xe<cube_channel_length-1 :
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

def fit_busy(x,y,initial_params):
    # starting parameters
    if initial_params[0]>cube_channel_length-2:
        initial_params[0] = cube_channel_length-3
    if initial_params[1]>cube_channel_length-2:
        initial_params[1] = cube_channel_length-3
        
    pos = initial_params +1e-4 * np.random.randn(32, 7)
    pos = np.abs(pos)
    nwalkers, ndim = pos.shape
    mean, median, yerr = sigma_clipped_stats(y, sigma=3,maxiters=None)
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
    

    W50_channels, W100_channels, x0_channels, zmin, zmax = calculate_velocity_widths(x,fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C, yerr)
    half_width = (W100_channels/2)

    
    fit_y =busy(x, fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C)

    try:rsqr = 1- (np.nansum((y-fit_y)**2))/(np.sum((y-np.nansum(y)/len(y))**2))
    except: print(y)
    # print('Rsqr: ',rsqr)
    return rsqr, fit_y, fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C, x0_channels, half_width, zmin, zmax

def find_source_extent(source_data_table_input):
    global source_data_table
    source_data_table = source_data_table_input

    try: a = source_data_table['x_pix_center'].values
    except:
        source_data_table['x_pix_center'] =  np.ones(len(source_data_table))*(-99)
        source_data_table['x_pix_min'] =  np.ones(len(source_data_table))*(-99)
        source_data_table['x_pix_max'] =  np.ones(len(source_data_table))*(-99)
        source_data_table['y_pix_center'] =  np.ones(len(source_data_table))*(-99)
        source_data_table['y_pix_min'] =  np.ones(len(source_data_table))*(-99)
        source_data_table['y_pix_max'] =  np.ones(len(source_data_table))*(-99)
        source_data_table['z_channel_center'] =  np.ones(len(source_data_table))*(-99)
        source_data_table['z_channel_min'] =  np.ones(len(source_data_table))*(-99)
        source_data_table['z_channel_max'] =  np.ones(len(source_data_table))*(-99)
        
        source_data_table['contour_diameter_arc'] = np.ones(len(source_data_table))*(-99)
        source_data_table['contour_flag'] = np.ones(len(source_data_table))
        
    for source in range(len(source_data_table)):
        # data coordinates
        ra, dec = source_data_table['RA_deg'].values[source], source_data_table['Dec_deg'].values[source]
        x_pix, y_pix = source_data_table['x_pix_center'].values[source], source_data_table['y_pix_center'].values[source]
        if x_pix==-99: 
            sky_coords =SkyCoord(ra, dec, unit="deg",frame="fk5")
            x_pix, y_pix = skycoord_to_pixel(SkyCoord(ra,dec, frame="fk5", unit="deg"),wcs_cube)
        try:
            z_channel = source_data_table['int_im_channel'].values[source]
            sigma = source_data_table['gauss_sigma'].values[source]
            gauss_H = source_data_table['gauss_H'].values[source]
        except: 
            z_channel = frequency_to_channel(source_data_table['frequency_Hz'].values[source],wcs_cube)
            sigma = 5
            gauss_H = 0
            
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
        try: zmin,zmax = source_data_table['z_channel_min'].values[source],source_data_table['z_channel_max'].values[source]
        except:zmin,zmax = x0-int(2*sigma),x0+int(2*sigma)

        initial_params = [z_channel,z_channel,1,10,np.sqrt(np.pi/2)/sigma,1e-4*1.1,gauss_H]
        if initial_params[4]>=10: initial_params[4] = 9
        
        while True:
            
            # break the loop if the value does not converge after too many tries and keep initial values
            loop = loop+1
            if loop>20:
                
                fail_flag=1
                source_data_table['contour_flag'].values[source] = fail_flag
                #print('loop fail for contour %s - kept last values of parameters'%(source_data_table['ID'].values[source]))
                break
                
            contour_found=False
            
            moment_0, mean, median, std, wcs_moment_0 = moment_0_map(x_pix,y_pix,zmin,zmax,map_size_pix,cube,cube_data,cube_channel_length)
            x_pix_moment_0, y_pix_moment_0 = skycoord_to_pixel(SkyCoord(ra,dec, frame="fk5", unit="deg"),wcs_moment_0)
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
                if contour_contains_point(final_contour, x_pix_moment_0, y_pix_moment_0,moment_0.shape): 
                    source_contour = copy.deepcopy(final_contour)
                    contour_found=True
            if not contour_found:
                fail_flag=1
                source_data_table['contour_flag'].values[source] = fail_flag
                zmin,zmax = x0-int(2*sigma),x0+int(2*sigma)
                x0=z_channel
                continue
                
            mask = contour_to_mask(source_contour,moment_0.shape)
            contour_sky = contour_sky_coord_in_deg_from_pix(source_contour,wcs_moment_0)
            flux = get_contour_spectrum(contour_sky,cube,wavelength_range_array)
            
            rsqr, fit_y, fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C, x0, half_width, zmin, zmax = fit_busy(wavelength_range_array,flux,initial_params)
    
            # update the function parameters
            initial_params = [fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C]
            if zmax-zmin < 2:
                zmax=zmax+1
                zmin=zmin-1

            # update the center
            x_center_contour,y_center_contour = contour_center(source_contour)
            ra_center,dec_center = sky_coord_in_deg_from_pix(x_center_contour,y_center_contour,wcs_moment_0)
            x_pix, y_pix = skycoord_to_pixel(SkyCoord(ra_center,dec_center, frame="fk5", unit="deg"),wcs_cube)
    
            # break the loop if the value converges
            if (contour_size[-1]*0.95<len(mask[mask==True])<contour_size[-1]*1.05 and contour_size[-2]*0.95<len(mask[mask==True])<contour_size[-2]*1.05 and contour_size[-3]*0.95<len(mask[mask==True])<contour_size[-3]*1.05) or (contour_size[-2]*0.95<len(mask[mask==True])<contour_size[-2]*1.05 and contour_size[-4]*0.95<len(mask[mask==True])<contour_size[-4]*1.05):
                fail_flag=0
                source_data_table['contour_flag'].values[source] = 0
                break
            contour_size.append(len(mask[mask==True]))

            # update range of the cubelet
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

        contour_diameter_pix = contour_diameter(source_contour)
        if fail_flag==1 or contour_diameter_pix<min_diameter_pix:
            x_pix, y_pix = skycoord_to_pixel(SkyCoord(ra,dec, frame="fk5", unit="deg"),wcs_cube)
            source_contour = circular_contour(x_pix,y_pix,min_diameter_pix/2)
            contour_diameter_pix = contour_diameter(source_contour)
            contour_sky = contour_sky_coord_in_deg_from_pix(source_contour,wcs_cube)
            if not os.path.exists(path_to_results+'contours'):
                os.makedirs(path_to_results+'contours')
            with open(path_to_results+'/contours/'+"%s_contour.data"%(source_data_table['ID'].values[source]), 'wb') as f:
                pickle.dump(contour_sky, f)

            #reset 
            wavelength_range_left, wavelength_range_right = z_channel-200, z_channel+200
            if wavelength_range_left<0: wavelength_range_left=2
            if wavelength_range_right>= cube_data.shape[0] : wavelength_range_right = cube_data.shape[0] -2
            wavelength_range_array = np.arange(wavelength_range_left,wavelength_range_right,1,dtype=int)
            initial_params = [z_channel,z_channel,1,10,np.sqrt(np.pi/2)/sigma,1e-4*1.1,gauss_H]
            if initial_params[4]>=100: initial_params[4] = 99

        flux = get_contour_spectrum(contour_sky,cube,wavelength_range_array)
        rsqr, fit_y, fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C, x0, half_width, zmin, zmax = fit_busy(wavelength_range_array,flux,initial_params)
    
        if zmax-zmin < 2:
            zmax=zmax+1
            zmin=zmin-1

        new_frequency = channel_to_frequency(x0,wcs_cube)
        new_z = (1420405000-new_frequency)/new_frequency

        source_data_table['frequency_Hz'].values[source],source_data_table['redshift'].values[source] = new_frequency, new_z
        contour_diameter_pix = contour_diameter(source_contour)
        source_data_table['contour_diameter_arc'].values[source] = contour_diameter_pix*dpix

        source_data_table['RA_deg'].values[source],source_data_table['Dec_deg'].values[source] = sky_coord_in_deg_from_pix(x_pix,y_pix,wcs_cube)
        source_data_table['x_pix_center'].values[source], source_data_table['y_pix_center'].values[source] = x_pix, y_pix
        
        source_data_table['z_channel_center'].values[source] = int(x0)
        source_data_table['z_channel_min'].values[source] = int(zmin)
        source_data_table['z_channel_max'].values[source] = int(zmax)

        if not os.path.exists(path_to_results+'contours'):
            os.makedirs(path_to_results+'contours')
        contour_sky = contour_sky_coord_in_deg_from_pix(source_contour,wcs_moment_0)
        with open(path_to_results+'/contours/'+"%s_contour.data"%(source_data_table['ID'].values[source]), 'wb') as f:
            pickle.dump(contour_sky, f)
            
        ra_max,dec_max = np.nanmax(contour_sky.T[0]),np.nanmax(contour_sky.T[1])
        ra_min,dec_min = np.nanmin(contour_sky.T[0]),np.nanmin(contour_sky.T[1])
        x_pix_max, y_pix_max = skycoord_to_pixel(SkyCoord(ra_max,dec_max, frame="fk5", unit="deg"),wcs_cube)
        x_pix_min, y_pix_min = skycoord_to_pixel(SkyCoord(ra_min,dec_min, frame="fk5", unit="deg"),wcs_cube)
        source_data_table['x_pix_min'].values[source], source_data_table['y_pix_min'].values[source] = x_pix_min, y_pix_min
        source_data_table['x_pix_min'].values[source], source_data_table['y_pix_min'].values[source] = x_pix_max, y_pix_max
            
        # plot quick plots
        if not os.path.exists(path_to_results+'/extent_quick_plots'):
            os.makedirs(path_to_results+'/extent_quick_plots')
        fig = plt.figure(figsize=(15,5))
        fig.suptitle(source_data_table['ID'].values[source], fontsize=16)

        gs = fig.add_gridspec(1,2, hspace=0, wspace=0,width_ratios = [1,2])
        ax = gs.subplots()
        ax[0].imshow(moment_0,origin='lower')
        if 3*contour_diameter_pix < 60:
            ax[0].set_xlim(int(moment_0.shape[0]/2)-30,int(moment_0.shape[0]/2)+30)
            ax[0].set_ylim(int(moment_0.shape[0]/2)-30,int(moment_0.shape[0]/2)+30)
        else:
            ax[0].set_xlim(int(moment_0.shape[0]/2)-1.5*contour_diameter_pix,int(moment_0.shape[0]/2)+1.5*contour_diameter_pix)
            ax[0].set_ylim(int(moment_0.shape[0]/2)-1.5*contour_diameter_pix,int(moment_0.shape[0]/2)+1.5*contour_diameter_pix)
        ax[0].plot(source_contour.T[0],source_contour.T[1],c='white',ls='--')
        ax[1].step(wavelength_range_array,flux,c='k',alpha=0.6,label='spectrum')
        ax[1].plot(wavelength_range_array,fit_y,c='blue',label='busy fit')
        ax[1].set_xlabel('channels [pix]')
        ax[1].set_ylabel('Flux')
        ax[1].axvline(  x0,color='gray',ls='--',alpha=0.5,lw=1,label='center')
        ax[1].axvline(  zmin,color='gray',ls='--',alpha=0.3,lw=1,label='full width')
        ax[1].axvline(  zmax,color='gray',ls='--',alpha=0.3,lw=1)
        ax[1].legend(loc='upper right')
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()
        if 8*half_width < 100:
            ax[1].set_xlim(x0-50,x0+50)
        else:
            ax[1].set_xlim(x0-4*half_width,x0+4*half_width)
           
        fig.savefig(path_to_results+'/extent_quick_plots/'+"%s_plot.jpg"%(source_data_table['ID'].values[source]))
                
    return source_data_table

def associate_sources_final(found_sources_dict):
    found_sources_dict['associated_with'] = np.full(len(found_sources_dict),'no_match_yet')
    for source in range(len(found_sources_dict)):
        dist = np.sqrt( (found_sources_dict['x_coord'].values[source] - found_sources_dict['x_coord'].values)**2 +
                        (found_sources_dict['y_coord'].values[source] - found_sources_dict['y_coord'].values)**2)
        dist_channel = np.absolute(found_sources_dict['x0_busy'].values[source]-found_sources_dict['x0_busy'].values)
        match_id = np.arange(0,len(found_sources_dict))[(dist<found_sources_dict['contour_diameter_arc'].values[source]/3*dpix)&(dist_channel<=(found_sources_dict['zmax'].values[source]-found_sources_dict['zmin'].values[source])/2)]
        found_sources_dict['associated_with'].values[source] = found_sources_dict['ID'].values[match_id[np.argmax(found_sources_dict['z_channel'].values[match_id])]]     
    return found_sources_dict

def source_extent_script(data_table, path_to_radio_file, initial_map_size_pix_input, min_diameter_arc_input,path_to_results_input,core_no_input ):
    global cube, cube_data, wcs_cube, cube_channel_length, initial_map_size_pix, dpix, path_to_results, min_diameter_pix, beam_diameter_arcsec_array
    initial_map_size_pix = initial_map_size_pix_input
    path_to_results = path_to_results_input
    cube, cube_data, wcs_cube, cube_channel_length, dpix, beam_diameter_arcsec_array =read_in_radio_file(path_to_radio_file)
    if min_diameter_arc_input == None:
        min_diameter_pix = int(np.median(beam_diameter_arcsec_array)/dpix)*2
    else:
        min_diameter_pix = min_diameter_arc_input/dpix
    
    if core_no_input == None: core_number = multiprocessing.cpu_count()
    else: core_number = core_no_input 
        
    source_extent_results = p_map(find_source_extent,
                                     [data_table[i:(i+1)] for i in range(len(data_table))],num_cpus=core_number)
    all_sources_dict = pd.concat(source_extent_results)
    all_sources_dict = associate_sources_final(all_sources_dict)
    #filter_associated = (all_sources_dict['ID'].values==all_sources_dict['associated_with'].values)
    #all_sources_dict = all_sources_dict[filter_associated]
    
    all_sources_dict.to_csv(path_to_results+'/found_sources_extent.csv',index=False)
    
    return all_sources_dict

    
