import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import glob
import cv2
import copy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.patches import Ellipse

import emcee
from scipy import special

from astropy.io import fits 
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.visualization import make_lupton_rgb
from astropy import units as u 
from astropy.stats import sigma_clipped_stats, sigma_clip
from astrocut import FITSCutout
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from p_tqdm import p_map
import multiprocessing
import traceback

from spectral_cube import SpectralCube


def sky_coord_in_deg_from_pix(x_pix_coord,y_pix_coord,wcs):
    skycoord = SkyCoord.from_pixel(xp=x_pix_coord,yp=y_pix_coord, wcs=wcs)
    RA = skycoord.ra.degree
    DEC = skycoord.dec.degree
    return RA, DEC

# get frequency form channel
def channel_to_frequency(channel,wave0,wavedelta,channel0):
    frequency = wave0+wavedelta*(channel-channel0)
    return frequency

def gauss(x, H, A, x0, sigma): 
            return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def contour_contains_point(contour_points, x_coord,y_coord,window_width):
    x_coord,y_coord = int(x_coord),int(y_coord)
    contour = np.array([contour_points], dtype=np.int32)
    mask = np.zeros([window_width,window_width], dtype=bool)
    temp_mask = np.zeros([window_width,window_width], dtype=np.uint8)
    cv2.drawContours(temp_mask, contour, contourIdx=-1, color=1, thickness=cv2.FILLED)
    mask[temp_mask == 1] = True

    return mask[y_coord][x_coord] 

def contour_center(contour_points):
    contour_points = contour_points.T
    x_center = int(np.mean(contour_points[0]))
    y_center = int(np.mean(contour_points[1]))
    return x_center, y_center

def contour_to_mask(contour_points,shape):
    contour = np.array([contour_points], dtype=np.int32)
    mask = np.zeros(shape, dtype=bool)
    temp_mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(temp_mask, contour, contourIdx=-1, color=1, thickness=cv2.FILLED)
    cv2.drawContours(temp_mask, contour, contourIdx=-1, color=1, thickness=1)
    mask[temp_mask == 1] = True
    return mask

def contour_to_mask_thick(contour_points,shape):
    contour = np.array([contour_points], dtype=np.int32)
    mask = np.zeros(shape, dtype=bool)
    temp_mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(temp_mask, contour, contourIdx=-1, color=1, thickness=cv2.FILLED)
    cv2.drawContours(temp_mask, contour, contourIdx=-1, color=1, thickness=3)
    mask[temp_mask == 1] = True
    return mask

def moment_0_map(xpix,ypix,zchannel_left,zchannel_right,map_size,cube,cube_data,cube_channel_length):
    
    if zchannel_left<0:zchannel_left=0
    if zchannel_right>cube_channel_length: zchannel_right = cube_channel_length-1
    zchannel_left, zchannel_right = int(zchannel_left), int(zchannel_right)
    
    ypix_left, ypix_right = int(ypix-(map_size/2)), int(ypix+(map_size/2))
    if ypix_left<0: ypix_left=0
    if ypix_right>cube_data.shape[1]: ypix_right = cube_data.shape[1]-1
        
    xpix_left, xpix_right = int(xpix-(map_size/2)), int(xpix+(map_size/2))
    if xpix_left<0: xpix_left=0
    if xpix_right>cube_data.shape[2]: xpix_right = cube_data.shape[2]-1

    cubelet=cube[zchannel_left:zchannel_right,ypix_left:ypix_right,xpix_left:xpix_right]

    cubelet.beam_threshold = [10000]
    moment_0 = cubelet.moment(order=0) 
    wcs_moment_0 = moment_0.wcs
    moment_0 = moment_0.value
    
    moment_0=np.nansum(np.array(cubelet.unmasked_data[:,:,:]),0)
    mean, median, std = sigma_clipped_stats(moment_0, sigma=3,maxiters=None)
    
    
    return moment_0, mean, median, std, wcs_moment_0


def ini_plot(x_pix_coord,y_pix_coord,ra,dec,source_ID):
    #prepare the figure and its layout
    fig = plt.figure(figsize=(8,8*11.5/10))
    gs = fig.add_gridspec(5,11, hspace=0, wspace=0,height_ratios = [2,1,5,0.01,3],width_ratios = [1,1,1,1,1,0.1,1,1,1,1,1])
    
    axs = gs.subplots(sharex=True, sharey=True)        

    for ax in axs[0, 0:11]:
        ax.remove()
    axcoordinfo = fig.add_subplot(gs[0, 0:3])
    axspecfitinfo = fig.add_subplot(gs[0, 3:8])
    axflaginfo = fig.add_subplot(gs[0, 8:11])

    for ax in axs[1, 0:]:
        ax.remove() 
    for ax in axs[2, 0:5]:
        ax.remove()

    wcs_radio_image = WCS(naxis=2)
    wcs_radio_image.wcs.crpix = [data_cube_wcs.wcs.crpix[0]-(x_pix_coord-image_pixel_width/2), data_cube_wcs.wcs.crpix[1]-(y_pix_coord-image_pixel_width/2)]
    wcs_radio_image.wcs.crval = [data_cube_wcs.wcs.crval[0], data_cube_wcs.wcs.crval[1]]
    wcs_radio_image.wcs.cunit = [data_cube_wcs.wcs.cunit[0], data_cube_wcs.wcs.cunit[1]]
    wcs_radio_image.wcs.ctype = [data_cube_wcs.wcs.ctype[0], data_cube_wcs.wcs.ctype[1]]
    wcs_radio_image.wcs.cdelt = [data_cube_wcs.wcs.cdelt[0], data_cube_wcs.wcs.cdelt[1]]

    axradioimage = fig.add_subplot(gs[2, 0:5],projection=wcs_radio_image)

    try:
        file_g = glob.glob(path_to_optical_images+'/optical_images_filter_R/*%s_*.fits'%(source_ID))[0]
        g=fits.open(file_g)
        wcs_vis = WCS(g[1].header)
        ra = round(ra,4)
        dec = round(dec,4)
        width_arc = round(image_arc_width,6)
        
        center_coord = SkyCoord(ra,dec, unit="deg")
        cutout_size = [width_arc/wcs_vis.wcs.cd[1][1]/3600, width_arc/3600/wcs_vis.wcs.cd[1][1]]
        output_files = FITSCutout(input_files=[file_g],
                                 coordinates=center_coord,
                                 cutout_size=cutout_size,
                                 single_outfile=False)
        wcs_vis = WCS(output_files.fits_cutouts[0][1].header)
    except: wcs_vis = wcs_radio_image

            
    for ax in axs[2, 5:]:
        ax.remove()
    axvisimage = fig.add_subplot(gs[2, 6:],projection=wcs_vis)

    for ax in axs[3, 0:]:
        ax.remove() 

    for ax in axs[4, 0:]:
        ax.remove()
    axspectrumwide = fig.add_subplot(gs[4, 0:])

    
    # set ticks and labels
    axflaginfo.set_xticks([])
    axflaginfo.set_yticks([])
    axspecfitinfo.set_xticks([])
    axspecfitinfo.set_yticks([])
    axcoordinfo.set_xticks([])
    axcoordinfo.set_yticks([])

    axvisimage.set_title('optical image [deg] \n \n \n')
    axradioimage.set_title('radio moment 0 map [deg] \n \n \n')
    axspectrumwide.set_xlabel('Frequency [MHz]')
    axspectrumwide.set_ylabel('Integrated Flux [Jy Hz]')
    axspectrumwide.xaxis.set_minor_locator(AutoMinorLocator())
    axspectrumwide.yaxis.set_minor_locator(AutoMinorLocator())

    
    ra_axis = axvisimage.coords[0]
    dec_axis = axvisimage.coords[1]

    ra_axis.set_major_formatter('d.ddd')
    dec_axis.set_major_formatter('d.ddd')

    ra_axis.set_ticks(number=4)
    dec_axis.set_ticks(number=4)

    ra_axis.set_axislabel('RA [deg]')
    dec_axis.set_axislabel('DEC [deg]')

    dec_axis.set_ticks_position('r')
    dec_axis.set_ticklabel_position('r')
    dec_axis.set_axislabel_position('r')
    ra_axis.set_ticks_position('t')
    ra_axis.set_ticklabel_position('t')
    ra_axis.set_axislabel_position('t')

    ra_axis = axradioimage.coords[0]
    dec_axis = axradioimage.coords[1]

    ra_axis.set_major_formatter('d.ddd')
    dec_axis.set_major_formatter('d.ddd')

    ra_axis.set_ticks(number=4)
    dec_axis.set_ticks(number=4)

    ra_axis.set_axislabel('RA [deg]')
    dec_axis.set_axislabel('DEC [deg]')

    dec_axis.set_ticks_position('l')
    dec_axis.set_ticklabel_position('l')
    dec_axis.set_axislabel_position('l')
    ra_axis.set_ticks_position('t')
    ra_axis.set_ticklabel_position('t')
    ra_axis.set_axislabel_position('t')

    return fig, axs, axflaginfo, axspecfitinfo, axcoordinfo, axradioimage, axvisimage, axspectrumwide
    
# functions for fitting busy function to the spectrum
def log_likelihood_busy(theta, x, y, yerr):
    xp, xe, a, w, b, c, C = theta
    model = busy(x, xp, xe, a, w, b, c, C)
    sigma2 = yerr**2
    
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior_busy(x,theta):
    xp, xe, a, w, b, c, C = theta
    if a>0 and w>=3 and 5>b>=0 and c>=0 and 0<xp<cube_channel_range-1 and 0<xe<cube_channel_range-1 :
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

def fit_busy(x,y,source,sources_dict):
    # starting parameters
    if sources_dict['fit_xp'].values[source]>cube_channel_range-2:
        sources_dict['fit_xp'].values[source] = cube_channel_range-3
    if sources_dict['fit_xe'].values[source]>cube_channel_range-2:
        sources_dict['fit_xe'].values[source] = cube_channel_range-3
        
    pos = [sources_dict['fit_xp'].values[source],sources_dict['fit_xe'].values[source],
           sources_dict['fit_a'].values[source], sources_dict['fit_w'].values[source],
           sources_dict['fit_b'].values[source],sources_dict['fit_c'].values[source],
           sources_dict['fit_C'].values[source]] +1e-4 * np.random.randn(32, 7)
    pos = np.abs(pos)
    nwalkers, ndim = pos.shape
    #print(pos)
    mean, median, yerr = sigma_clipped_stats(y, sigma=3,maxiters=None)
    #yerr=np.abs(np.max(y)-np.min(y))*0.01
    yerr=0.0001
    if yerr==0: yerr=0.0001
    try:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_busy, args=(x, y, yerr))
        sampler.run_mcmc(pos, 7000, progress=False)
    
        flat_samples = sampler.get_chain(discard=200, thin=10, flat=True)
    except RuntimeWarning:
        print(pos,cube_channel_length,sources_dict['fit_xp'].values[source])

    fit_xp = np.percentile(flat_samples[:, 0], [50])[0]
    fit_xe = np.percentile(flat_samples[:, 1], [50])[0]
    fit_a = np.percentile(flat_samples[:, 2], [50])[0]
    fit_w = np.percentile(flat_samples[:, 3], [50])[0]
    fit_b = np.percentile(flat_samples[:, 4], [50])[0]
    fit_c = np.percentile(flat_samples[:, 5], [50])[0]
    fit_C = np.percentile(flat_samples[:, 6], [50])[0]
    
    fit_y =busy(x, fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C)
    return fit_y, fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C
    
def get_beam_spectrum(x_coord,y_coord,beam_diameter,flux_cubelet_start,flux_cubelet_end):
    if flux_cubelet_start<0: flux_cubelet_start=0
    if flux_cubelet_end>cube_channel_range: flux_cubelet_end=cube_channel_range


    flux_cubelet = np.array(data_cube.unmasked_data[int(flux_cubelet_start):int(flux_cubelet_end),
                                                y_coord-int(beam_diameter):y_coord+int(beam_diameter),
                                                x_coord-int(beam_diameter):x_coord+int(beam_diameter)],dtype=np.float32)
    rows, cols = flux_cubelet.shape[1], flux_cubelet.shape[2]
    y, x = np.ogrid[:rows, :cols]
    center_row, center_col = (rows - 1) / 2, (cols - 1) / 2
    distance = np.sqrt((x - center_col)**2 + (y - center_row)**2)
    
    flux = np.ones(int(flux_cubelet_end-flux_cubelet_start))
    for channel in range(len(flux)):
        radius = int(beam_diameter/2)-1
        if radius<2.5: radius=2.5
        (flux_cubelet[channel])[distance>radius] = np.nan
        flux[channel] = np.nansum(flux_cubelet[channel])
    return flux
    
def channel_to_frequency(channel,wcs):
    wave0,wavedelta,channel0 = wcs.wcs.crval[2],wcs.wcs.cdelt[2],wcs.wcs.crpix[2]
    frequency = wave0+wavedelta*(channel-channel0)
    return frequency
    
    
def wide_spectrum_plot(axspectrumwide,wavelength_range_array_wide,x_coord,y_coord,beam_diameter,sources_dict,source,bmpix,contour_mask,map_size_pix,cube_data):
    cube_data = data_cube.unmasked_data[:,:,:]
    
    # get flux for the contour mask
    flux=np.zeros(len(wavelength_range_array_wide))
    for i, wavelength in enumerate(wavelength_range_array_wide):     
        x_pix_left, x_pix_right = int(x_coord-(map_size_pix/2)),int(x_coord+(map_size_pix/2))
        y_pix_left, y_pix_right = int(y_coord-(map_size_pix/2)),int(y_coord+(map_size_pix/2))

        if x_pix_left<0: x_pix_left=0
        if y_pix_left<0: y_pix_left=0
        if x_pix_right>cube_data.shape[2]: x_pix_right=cube_data.shape[2]-1
        if y_pix_right>cube_data.shape[1]: y_pix_right=cube_data.shape[1]-1
        image_slice = cube_data[wavelength,y_pix_left:y_pix_right,x_pix_left:x_pix_right]
        image_slice = np.array(image_slice)

        # apply mask and sum up the flux in channel image
        flux[i] = np.nansum(image_slice[contour_mask])

    # plot flux vs wavelength
    xdata = channel_to_frequency(wavelength_range_array_wide,data_cube_wcs)/1E6
    ydata = flux*bmpix*dfreq # flux in Jy Hz
    
    try:axspectrumwide.set_ylim((np.nanmin(ydata)-0.1*np.absolute(np.nanmax(ydata)-np.nanmin(ydata))),np.nanmax(ydata)+0.15*np.absolute(np.nanmax(ydata)-np.nanmin(ydata)))
    except:
        #plt.scatter(xdata,ydata)
        #plt.show()
        print(ydata)
        print(flux)
        print(x_coord,y_coord,sources_dict['z_channel_center'].values[source])
        print(wavelength_range_array_wide)
        print(x_coord,y_coord,beam_diameter,wavelength_range_array_wide[0],wavelength_range_array_wide[-1]+1)
    axspectrumwide.set_xlim(np.nanmin(xdata),np.nanmax(xdata))
  
    #axspectrumwide.step(xdata,ydata,color = 'black',alpha=0.45,lw=0.7)
    axspectrumwide.step(xdata,ydata,color = 'black',alpha=0.5,lw=0.75,label='spectrum')
    # busy_fit, fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C = fit_busy(wavelength_range_array_wide,ydata/(bmpix*dfreq),source,sources_dict)
    
    
    # xdata = np.arange(wavelength_range_array_wide[0],wavelength_range_array_wide[-1]+1,0.2)
    # busy_func = busy(xdata,fit_xp, fit_xe, fit_a, fit_w, fit_b, fit_c, fit_C)*bmpix*dfreq
    # xdata = channel_to_frequency(xdata,data_cube_wcs)/1E6
    # axspectrumwide.plot(xdata,busy_func,color='blue',label='busy function',lw=0.75,alpha=0.9)
    #axspectrumwide.plot(channel_to_frequency(xdata,data_cube_wcs)/1E6,bmpix*dfreq*gauss(xdata,sources_dict['H'].values[source],sources_dict['A'].values[source],sources_dict['x0'].values[source],sources_dict['sigma'].values[source],),color='blue',label='Gaussian function fit',lw=0.75,alpha=0.9)
    z_channel = sources_dict['z_channel_center'].values[source]

    try:
        axspectrumwide.axvline(  1420.406/(sources_dict['z_spec'].values+1),color='gray',ls='--',alpha=0.2,lw=1,label='z$_{opt}$')
    except:
        pass
        
    # axspectrumwide.axvline(channel_to_frequency(z_channel+sources_dict['W50'].values[source]/2-0.5,data_cube_wcs)/1E6,color='gray',
    #                        ls='--',alpha=0.6,lw=1)
    # axspectrumwide.axvline(channel_to_frequency(z_channel-sources_dict['W50'].values[source]/2+0.5,data_cube_wcs)/1E6,
    #                        color='gray',ls='--',alpha=0.6,lw=1,label=r'$W_{50}$')
    axspectrumwide.legend(loc='upper right')
    
    return ydata 

def get_contour(moment_0):
    moment_0 = np.nan_to_num(moment_0)
    mean, median, std = mean_wide, median_wide, std_wide
    
    max_std = (np.max(moment_0)-median)/(std)
    threshold_value = median + 3*std
    _, binary_image = cv2.threshold(moment_0.astype(np.float32), threshold_value, 1, cv2.THRESH_BINARY)

    # Convert to uint8 for OpenCV (0 or 255)
    binary_image = (binary_image * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    final_contours=[]
    contour_found=False
    for contour in contours:
        final_contour = []
        for point in contour:
            final_contour.append(point[0])
        final_contour.append(final_contour[0])
        final_contour = np.array(final_contour)
        
        x_contour_center,y_contour_center = contour_center(final_contour)
        dist = np.sqrt((int(moment_0.shape[0]/2)-x_contour_center)**2+(int(moment_0.shape[0]/2)-y_contour_center)**2)
        
        if contour_contains_point(final_contour, int(moment_0.shape[0]/2),int(moment_0.shape[0]/2),moment_0.shape[0]) or dist<3: 
            source_contour = copy.deepcopy(final_contour)
            contour_found=True
    if not contour_found:
        
        pass
        
    contour_mask_thick = contour_to_mask_thick(source_contour,moment_0.shape)
    contour_mask = contour_to_mask(source_contour,moment_0.shape)
    moment_0_masked = copy.deepcopy(moment_0)
    moment_0_masked[~contour_mask_thick]=0
    contour_levels = np.linspace(median+3*std,median+max_std*std,6)

    return contour_mask,moment_0_masked, contour_levels

def get_beam_contour(moment_0,beam_radius_pixel):
    
    moment_0 = np.nan_to_num(moment_0)
    
    rows, cols = moment_0.shape[0], moment_0.shape[1]
    y, x = np.ogrid[:rows, :cols]
    center_row, center_col = (rows - 1) / 2, (cols - 1) / 2
    distance = np.sqrt((x - center_col)**2 + (y - center_row)**2)
    
    radius = beam_radius_pixel
    if radius<2.5: radius=2.5
    contour_mask = np.full(moment_0.shape,True)
    contour_mask[distance>radius] = False

    return contour_mask

def hi_image_plot(x_coord,y_coord,wavelength_range_left,wavelength_range_right,axradioimage,beam_radius_pixel):
    global mean_wide, median_wide, std_wide
    # HI IMAGE
    # create moment zero map from channels defined by the wavelength range
    moment_0, mean, median, std, wcs_moment_0 = moment_0_map(x_coord,y_coord,wavelength_range_left,wavelength_range_right,image_pixel_width,data_cube,data_cube.unmasked_data[:,:,:],cube_channel_range)

    slit_image = np.array(moment_0)

    # get stats
    moment_0_wide, mean_wide, median_wide, std_wide, wcs_moment_0_wide = moment_0_map(x_coord,y_coord,wavelength_range_left,wavelength_range_right,image_pixel_width*3,data_cube,data_cube.unmasked_data[:,:,:],cube_channel_range)


    # plot the cross hair
    axradioimage.axvline((slit_image.shape[0]-1)/2+0.5,color='gray',ls='--',alpha=0.3)
    axradioimage.axhline((slit_image.shape[1]-1)/2+0.5,color='gray',ls='--',alpha=0.3)

    # plot the image
    axradioimage.imshow(slit_image,origin='lower')

    #plot beam
    if beam_radius_pixel/image_pixel_width*1.1<1/8: circle1 = plt.Circle((slit_image.shape[0]/8, slit_image.shape[0]/8), slit_image.shape[0]/image_pixel_width*beam_radius_pixel, color='white',fill=False,ls='--')
    else: circle1 = plt.Circle((beam_radius_pixel/image_pixel_width*1.1*slit_image.shape[0], beam_radius_pixel/image_pixel_width*1.1*slit_image.shape[0]), slit_image.shape[0]/image_pixel_width*beam_radius_pixel, color='white',fill=False,ls='--')
    axradioimage.add_patch(circle1)

    return moment_0, wcs_moment_0

def optical_image_plot(axvisimage,sources_dict,source,beam_radius_pixel):

    #try:
    ra = sources_dict['RA_deg'].values[source]
    dec = sources_dict['Dec_deg'].values[source]
    source_ID = sources_dict['ID'].values[source]

    width_arc = round(image_pixel_width*dpix,6)
    
    try:
        file_g = glob.glob(path_to_optical_images+'/optical_images_filter_R/*%s_*.fits'%(source_ID))[0]
   
        file_b = glob.glob(path_to_optical_images+'/optical_images_filter_G/*%s_*.fits'%(source_ID))[0]
        file_r = glob.glob(path_to_optical_images+'/optical_images_filter_I/*%s_*.fits'%(source_ID))[0]
        
        r = fits.open(file_r)
        g=fits.open(file_g)
        b = fits.open(file_b)
        
    
        wcs_vis = WCS(g[1].header)
        center_coord = SkyCoord(ra,dec, unit="deg")
        cutout_size = [width_arc/wcs_vis.wcs.cd[1][1]/3600, width_arc/3600/wcs_vis.wcs.cd[1][1]]
    
        output_files = FITSCutout(input_files=[file_g,file_b,file_r],
                                 coordinates=center_coord,
                                 cutout_size=cutout_size,
                                 single_outfile=False)
        r_image=np.sqrt(np.absolute(output_files.fits_cutouts[2][1].data))
    
        g_image=np.sqrt(np.absolute(output_files.fits_cutouts[0][1].data))
        b_image=np.sqrt(np.absolute(output_files.fits_cutouts[1][1].data))
        wcs_vis = WCS(output_files.fits_cutouts[2][1].header)
    
        # r_image=np.sqrt(np.absolute(r[1].data))
        # g_image=np.sqrt(np.absolute(g[1].data))
        # b_image=np.sqrt(np.absolute(b[1].data))
        
        
        min_r,min_b,min_g = np.min(r_image),np.min(b_image),np.min(g_image)
        max_r,max_b,max_g = np.max(r_image),np.max(b_image),np.max(g_image)
        
        
        rgb = make_lupton_rgb(r_image*0.9,g_image,b_image*1.3,Q=2.5,stretch=1.1)
        axvisimage.imshow(rgb,origin='lower')
    except Exception:
        traceback.print_exc()
        print(path_to_optical_images+'/optical_images_filter_R/*%s_*.fits'%(source_ID))
        print(output_files.fits_cutouts)
    
    # mark beam
    # circle1 = plt.Circle((r_image.shape[0]/2, r_image.shape[0]/2), r_image.shape[0]/image_pixel_width*beam_radius_pixel, color='white',fill=False,ls='--')
    # axvisimage.add_patch(circle1)
        

    # plot the cross hair
    axvisimage.axvline((rgb.shape[0]-1)/2,color='gray',ls='--',alpha=0.3)
    axvisimage.axhline((rgb.shape[1]-1)/2,color='gray',ls='--',alpha=0.3)


def info_about_target(source_dict,source,axflaginfo, axspecfitinfo, axcoordinfo,beam_diameter):
    # coordinates
    
    axcoordinfo.text(0.03,0.9,'DETECTION COORDINATES')
    sky_coords = SkyCoord(source_dict['RA_deg'].values[source],source_dict['Dec_deg'].values[source], frame="fk5", unit="deg")
    x_pix_coord, y_pix_coord = skycoord_to_pixel(sky_coords,data_cube_wcs)
    axcoordinfo.text(0.03,0.75,'xpix ypix channel: '+str(int(x_pix_coord))+' '+str(int(y_pix_coord))+' '+str(round(source_dict['z_channel_center'].values[source])))
    
    if sky_coords.dec.dms.d<0: zero_non_zero = '-'
    else: zero_non_zero='+'
    hex_coords = str(int(sky_coords.ra.hms.h)).zfill(2)+'h'+str(int(sky_coords.ra.hms.m)).zfill(2)+'m'+str(round(sky_coords.ra.hms.s,1)).zfill(4)+'s '+zero_non_zero+str(int(abs(sky_coords.dec.dms.d))).zfill(2)+'d'+str(int(abs(sky_coords.dec.dms.m))).zfill(2)+'m'+str(int(round(abs(sky_coords.dec.dms.s)))).zfill(2)+'s'
    axcoordinfo.text(0.03,0.60,'deg: '+str(round(source_dict['RA_deg'].values[source],5))+' '+str(round(source_dict['Dec_deg'].values[source],5)))
    axcoordinfo.text(0.03,0.50,'hex: '+hex_coords)
    axcoordinfo.text(0.03,0.40,  'frequency: '+str(round(source_dict['frequency_Hz'].values[source]/1E6,2))+' [MHz]')
    
    redshifthi = (1420 - source_dict['frequency_Hz'].values[source]/1E6)/(source_dict['frequency_Hz'].values[source]/1E6)
    redshiftoh = (1665 - source_dict['frequency_Hz'].values[source]/1E6)/(source_dict['frequency_Hz'].values[source]/1E6)
    
    axcoordinfo.text(0.03,0.30,  'redshift:   HI: '+str(round(redshifthi,3))+'  OH: '+str(round(redshiftoh,3)))
    axcoordinfo.text(0.03,0.05,  'beam diameter: '+str(round(beam_diameter,2))+' arcsec')

    
    # # spectral fitting
    # axspecfitinfo.text(0.03,0.9,'HI EMISSION')
    
    # axspecfitinfo.text(0.03,0.75,r'log(M$_{HI}$/M$_{☉}$) = %s ± %s'%(str(round(source_dict['MHI_prism'].values[source],3)),str(round((source_dict['MHI_prism_error']).values[source],3))))
    # axspecfitinfo.text(0.03,0.65, r'F$_{tot}$ [Jy Hz] = %s ± %s'%(str(round((source_dict['flux_Jy_Hz']).values[source],3)),str(round((source_dict['flux_Jy_Hz_error']).values[source],3))))
    # axspecfitinfo.text(0.03,0.55, r'SNR$_{3D}$ = '+str(round((source_dict['integrated_SNR']).values[source],3)))
    # if (source_dict['confidence_flag']).values[source]==0 and (source_dict['confused_flag']).values[source]==1: axspecfitinfo.text(0.03,0.05, 'confident detection, confused source')
    # elif (source_dict['confidence_flag']).values[source]==1 and (source_dict['blended_flag']).values[source]==0: axspecfitinfo.text(0.03,0.05, 'not confident detection, single source')
    # elif (source_dict['confidence_flag']).values[source]==1 and (source_dict['blended_flag']).values[source]==1: axspecfitinfo.text(0.03,0.05, 'not confident detection, blended source')
    # elif (source_dict['confidence_flag']).values[source]==0 and (source_dict['blended_flag']).values[source]==0: axspecfitinfo.text(0.03,0.05, 'confident detection, single source')
    # elif (source_dict['confidence_flag']).values[source]==0 and (source_dict['blended_flag']).values[source]==1: axspecfitinfo.text(0.03,0.05, 'confident detection, blended source')
    
    # axflaginfo.text(0.03,0.9,'HI SPECTRAL PROFILE')
    # dv = dfreq/source_dict['frequency_Hz'].values[source]*300000
    # axflaginfo.text(0.03,0.75,r'W$_{50}$ [km/s] = '+str(round((source_dict['W50']).values[source]*dv,3))+'±'+str(round((source_dict['W50_error']).values[source]*dv,3)))
    # axflaginfo.text(0.03,0.65,r'W$_{50}$ [channels] = '+str(round((source_dict['W50']).values[source],2)) +'±'+str(round((source_dict['W50_error']).values[source],2)) )
    # axflaginfo.text(0.03,0.55,r'W$_{100}$ [km/s] = '+str(round((source_dict['W100']).values[source]*dv,3))+'±'+str(round((source_dict['W100_error']).values[source]*(source_dict['dv']).values[source],3))) 
    # axflaginfo.text(0.03,0.45,r'W$_{100}$ [channels] = '+str(round((source_dict['W100']).values[source],2))+'±'+str(round((source_dict['W100_error']).values[source],2)) )
    # axflaginfo.text(0.03,0.35,r'Δv$_{channel}$ [km/s] = '+str(round(dv,1)) )
    

def read_in_radio_file(radio_file):
    
    # get cube data
    data_cube = SpectralCube.read(radio_file)
    # wcs
    data_cube_wcs = data_cube.wcs
    # channel range
    cube_channel_range = data_cube.shape[0]
    
    
    # create 2D WCS object
    data_cube_wcs_2d = WCS(naxis=2)
    data_cube_wcs_2d.wcs.crpix = [data_cube_wcs.wcs.crpix[0], data_cube_wcs.wcs.crpix[1]]
    data_cube_wcs_2d.wcs.crval = [data_cube_wcs.wcs.crval[0], data_cube_wcs.wcs.crval[1]]
    data_cube_wcs_2d.wcs.cunit = [data_cube_wcs.wcs.cunit[0], data_cube_wcs.wcs.cunit[1]]
    data_cube_wcs_2d.wcs.ctype = [data_cube_wcs.wcs.ctype[0], data_cube_wcs.wcs.ctype[1]]
    data_cube_wcs_2d.wcs.cdelt = [data_cube_wcs.wcs.cdelt[0], data_cube_wcs.wcs.cdelt[1]]

            
    
    return data_cube,data_cube_wcs,data_cube_wcs_2d,cube_channel_range


def make_plot(sources_dict):
    for source in range(0,len(sources_dict)):
        global beams_arc_diameter_array, image_pixel_width,image_arc_width, axvisimage, axradioimage
        #print('source: ',source,' out of ',len(sources_dict))
        # data coordinates
        ra, dec = (sources_dict['RA_deg'].values)[source], (sources_dict['Dec_deg'].values)[source]
        sky_coords =SkyCoord(ra, dec, unit="deg",frame="fk5")
        x_pix_coord, y_pix_coord = skycoord_to_pixel(sky_coords,data_cube_wcs)
        z_channel, sigma =  (sources_dict['z_channel_center'].values)[source], ((sources_dict['z_channel_max'].values)[source]-(sources_dict['z_channel_min'].values)[source])/4
        if (sources_dict['z_channel_center'].values)[source]==-99: z_channel = (sources_dict['gauss_x0'].values)[source]
        x_pix_coord,y_pix_coord,z_channel  = int(x_pix_coord),int(y_pix_coord),int(z_channel)
        
        image_pixel_width = int(sources_dict['contour_diameter_arc'].values[source]*2/dpix)
        if sources_dict['contour_diameter_arc'].values[source]>200:image_pixel_width = int(sources_dict['contour_diameter_arc'].values[source]*1.1/dpix)
        if image_pixel_width<image_pixel_width_min: image_pixel_width = image_pixel_width_min
        image_arc_width = image_pixel_width*dpix
        
        spectrum_length = sigma*4*6
        if spectrum_length<spectrum_length_min: spectrum_length = spectrum_length_min
        
        

        # spectrum range
        wavelength_range_left, wavelength_range_right = int(z_channel-spectrum_length/2), int(z_channel+spectrum_length/2)
        if wavelength_range_left<0: wavelength_range_left=2
        if wavelength_range_right>= cube_channel_range : wavelength_range_right = cube_channel_range -2
        wavelength_range_array = np.arange(wavelength_range_left,wavelength_range_right+1,1,dtype=int)

        
        # beam diameter
        if beam==None:
            try:
                beams_arc_diameter_array = data_cube.beams.major.value
            except: 
                try:
                    beams_arc_diameter_array = np.ones(len(data_cube.beam.value)) * data_cube.beam.major.value
                except: 
                    print('SpectralCube failed to read the beam data, use "beam" argument to specify the beam diameter manually.')
        else:
            beams_arc_diameter_array = np.ones(cube_channel_range)*(beam*np.abs(data_cube_wcs.wcs.cdelt[0])*3600)

        beams_pixel_diameter_array =  beams_arc_diameter_array/3600/np.abs(data_cube_wcs.wcs.cdelt[0])

        beam_diameter_arc = np.median(beams_arc_diameter_array)
        #print('beam diameter: ',beam_diameter_arc)
        beam_radius_pixel = np.abs(round(beam_diameter_arc/2/np.abs(data_cube_wcs.wcs.cdelt[0]*3600),3))
        #print('beamradius: ',beam_radius_pixel)
        
        
        beam_diameter_pixel =  np.median(beams_pixel_diameter_array)
    
        omega = np.pi*beam_diameter_arc**2/(4*np.log(2)) # beam area in arcsec
        bmpix=dpix**2/omega
    
        # MAKE THE PLOT
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)
            
        # initialize the plot
        fig, axs, axflaginfo, axspecfitinfo, axcoordinfo, axradioimage, axvisimage, axspectrumwide = ini_plot(x_pix_coord,y_pix_coord,ra,dec,sources_dict['ID'].values[source])  
        
        # plot HI image
        left_channel = int(np.floor(z_channel-2*sigma))
        right_channel = int(np.ceil(z_channel+2*sigma))

        if left_channel<0: left_channel=0
        if right_channel>cube_channel_range: right_channel=cube_channel_range

        moment_0,wcs_moment_0 = hi_image_plot(x_pix_coord,y_pix_coord, left_channel, right_channel,axradioimage,beam_radius_pixel)

        # plot optical image
        optical_image_plot(axvisimage,sources_dict,source,beam_radius_pixel)

        # plot contours
        try:
            contour_mask, moment_0_masked, contour_levels = get_contour(moment_0)
            if len(contour_mask[contour_mask==True])<beam_radius_pixel**2*np.pi:
                contour_mask = get_beam_contour(moment_0,beam_radius_pixel)
                moment_0_masked = moment_0
                moment_0_masked[~contour_mask]=-99
                contour_levels=[-99]
                beam_contour = Ellipse((ra,dec),width=beam_diameter_arc/3600,height=beam_diameter_arc/3600,angle=0,ls='-',lw=0.8,alpha=0.9,color='white',fc='None',transform=axradioimage.get_transform('world'))
                axradioimage.add_patch(beam_contour)

                beam_contour = Ellipse((ra,dec),width=beam_diameter_arc/3600,height=beam_diameter_arc/3600,angle=0,ls='-',lw=0.8,alpha=0.9,color='white',fc='None',transform=axvisimage.get_transform('world'))
                axvisimage.add_patch(beam_contour)
                
            else:
                axradioimage.contour(moment_0_masked, contour_levels[1:], colors='white',transform=axradioimage.get_transform(wcs_moment_0),linewidths=0.8, alpha=0.9 )
                axradioimage.contour(moment_0_masked, [contour_levels[0]], colors='white',transform=axradioimage.get_transform(wcs_moment_0),linewidths=0.8, alpha=0.9,linestyles='--' )
                axvisimage.contour(moment_0_masked, contour_levels, colors='white',transform=axvisimage.get_transform(wcs_moment_0),linewidths=0.8, alpha=0.5 )
                
        except:
            contour_mask = get_beam_contour(moment_0,beam_radius_pixel)
            moment_0_masked = moment_0
            moment_0_masked[~contour_mask]=-99
            contour_levels=[-99]

            beam_contour = Ellipse((ra,dec),width=beam_diameter_arc/3600,height=beam_diameter_arc/3600,angle=0,ls='-',lw=0.8,alpha=0.9,color='white',fc='None',transform=axradioimage.get_transform('world'))
            axradioimage.add_patch(beam_contour)

            beam_contour = Ellipse((ra,dec),width=beam_diameter_arc/3600,height=beam_diameter_arc/3600,angle=0,ls='-',lw=0.8,alpha=0.9,color='white',fc='None',transform=axvisimage.get_transform('world'))
            axvisimage.add_patch(beam_contour)
            
        

        # INFO ABOUT TARGET
        info_about_target(sources_dict,source,axflaginfo, axspecfitinfo, axcoordinfo,beam_diameter_arc) 
        
        
        # plot wide spectrum
        flux = wide_spectrum_plot(axspectrumwide,wavelength_range_array,x_pix_coord,y_pix_coord,beam_diameter_pixel,sources_dict,source,bmpix,contour_mask,image_pixel_width,data_cube)
        
    
        axspecfitinfo.set_title(sources_dict['ID'].values[source],fontsize=12)
        
            
        #out_file = path_to_results_dir+sources_dict['ID'].values[source]+'.pdf'
        #fig.savefig(out_file,bbox_inches='tight')
        out_file = path_to_results_dir+sources_dict['ID'].values[source]+'.%s'%(filetype)
        fig.savefig(out_file,bbox_inches='tight')
        plt.close(fig)
        #except: print('fail')


def emission_plot_script(data_table,path_to_data_cube,path_to_optical_images_input,path_to_results,image_arc_width_input,spectrum_length_input,beam_input,filetype_input):
    
    # globals
    global data_cube,data_cube_wcs,data_cube_wcs_2d,cube_channel_range, image_arc_width,spectrum_length_min, \
    path_to_optical_images, dpix,dfreq,image_pixel_width_min, cube_start, cube_end,path_to_results_dir,filetype,beam
    

    # results directory
    path_to_results_dir = path_to_results+'/emission_plots_1/'
    i=1
    while os.path.exists(path_to_results_dir):
        i=i+1
        path_to_results_dir = path_to_results+'/emission_plots_%s/'%(i)
    os.makedirs(path_to_results_dir)
    
    

    path_to_optical_images = path_to_optical_images_input
    image_arc_width_min,spectrum_length_min, beam,filetype = image_arc_width_input, spectrum_length_input, beam_input,filetype_input
    
    # data array
    sources_dict_full = data_table
    
    # data cube
    data_cube,data_cube_wcs,data_cube_wcs_2d,cube_channel_range=read_in_radio_file(path_to_data_cube)
    image_pixel_width_min = int(np.array(image_arc_width_min/( np.abs(data_cube_wcs.wcs.cdelt[0])*3600) ))
    if np.mod(image_pixel_width_min,2)!=0: image_pixel_width_min=image_pixel_width_min+1
    dpix=np.abs(data_cube_wcs.wcs.cdelt[0])*3600 # width of pixel in arcsec
    dfreq=np.abs(data_cube_wcs.wcs.cdelt[2]) # width of channel in Hz
    
    #plot params
    plt.rcParams.update({'font.size': 7})
    
    
    cube_start=0
    cube_end=cube_channel_range
    core_number = multiprocessing.cpu_count()
    n_sources=len(sources_dict_full)

    
    results = p_map(make_plot,[sources_dict_full[i:(i+1)] for i in range(len(data_table))])
    



