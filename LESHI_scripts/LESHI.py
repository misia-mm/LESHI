from .src import eye_check_file
from .src import source_finder_file
from .src import get_opt_cutout_file
from .src import detection_plot_file

def source_finder(data_cube, path_to_results,
                       SNR_integ=3.5, SNR_channel=2.5, 
                       channel_min_len=3,
                       channel_max_len=100,
                       SNR_spec=3.5, rsqr_min=0.35,
                       int_image_len=10, int_image_load_no=10,
                       channel_start=0, channel_end=None, beam=None, test_hist=False, 
                       bg_box_size=100, max_dist_pix=10, max_dist_channel=10 ):
    """Start the LESHI Source Finder. 
    
    Parameters
    ----------
    data_file : str
        Path to the datacube to perform source finding on.

    path_to_results : str
        Path to the directory where the results will be saved.

    SNR_integ : int or float
        Threshold signal-to-noise ratio of each source on the integrated image (default = 3.5)

    SNR_channel : int or float
        Threshold signal-to-noise ratio of each source on the channel image (default = 2.5)

    channel_min_len : int
        Minimum number of channels that the signal persists for (default = 3)

    channel_max_len : int
        Maximal acceptable width of the signal in channels (default = 100)

    SNR_spec : int or float
        Threshold signal-to-noise ratio of each source on the spectrum (default = 3.5)

    rsqr_min : int or float
        Minimum value of the R^2 parameter (raning from 0 to 1) quantifying how well the fitted Gaussian
        function fits the spectrum (default = 0.35)

    int_image_len : even int
        Channel width of the cube slab to integrate into a moment-0 map. Has to be an even number. 
        Should be equal to the width of the weakest expected signals (default = 10)
    
    int_image_load_no : int
        Number of images to process at the same time, is limited by the available RAM (default = 10)

    channel_start : int
        Channel of the data cube where sourcefinding should start (default = 0)

    channel_end : int
        Channel of the data cube where sourcefinding should end, if None, the source finding will end 
        at the end of the data cube (default = None)

    beam : int or float
        Diameter of the synthesised beam of the data cube in arcseconds, only needed if this information cannot
        be read from the data by SpectralCube package (defulat = None).

    test_hist : bool
        Whether to output the full testing history or not, showing where each source failed or passed, useful for troubleshooting
        (defualt = False)

    bg_box_size : int
        Width of the box in pixels used to calculate local background noise (default = 100)

    max_dist_pix : int
        Maximal distance in pixels between two sources to be associated (defualt = 10)

    max_dist_channel : int
        Maximal distance in channels between two sources to be associated (defualt = 10)
    
    Returns
    ------
    output : none
        Function does not return anything, all results are saved in the specified csv file.
    """

    if int_image_len % 2 == 0:
        source_finder_file.source_finder_func(data_cube, path_to_results,
                       SNR_integ, SNR_channel, 
                       channel_min_len,
                       channel_max_len,
                       SNR_spec, rsqr_min,
                       int_image_len,int_image_load_no,
                       channel_start,channel_end,beam,test_hist, 
                       bg_box_size,max_dist_pix,max_dist_channel )
    else:
        print('int_image_len should be an even number')

def get_opt_cutout(ID,ra,dec,width_arc = 100, filters = ['G','R','I'], survey = 'LegacySurvey', path_to_images = './'):
    """Start the LESHI Source Finder. 
    
    Parameters
    ----------
    ID : str or str array
        ID of the source, optical images will be saved under this name.
        
    ra : float or float array
        Right acsension in degrees of the coordinates of the center of the cutouts to download.
        
    dec : float or float array
        Declination in degrees of the coordinates of the center of the cutouts to download.
    
    width_arc : float or float_array
        Width of the cutouts in arcseconds (default = 100)

    filters : str array
        Array with filter names of cutouts to download (default = ['G','R','I'])

    survey :  'HSC' or 'LegacySurvey'
        Which survey cutouts to download (default = 'LegacySurvey')

    path_to_images : str
        Path to directory where the images should be saved (default is in the current directory)
    
    Returns
    ------
    output : none
        Function does not return anything, all results are saved in the specified directory.
    """
    
    if survey == 'LegacySurvey':
        if not hasattr(ra, '__iter__'):
            ra,dec, ID, width_arc = np.array([ra]),np.array([dec]),np.array([ID]),np.array([width_arc]), 
        get_opt_cutout_file.get_LG_image(ID,ra,dec,width_arc, filters, path_to_images)
        
    if survey == 'HSC':
        if hasattr(ra, '__iter__'):
            get_opt_cutout_file.get_HSC_image_in_bulk(ID,ra,dec,width_arc, filters,path_to_images)
        else:
            get_opt_cutout_file.get_HSC_image(ID,ra,dec,width_arc, filters,path_to_images)


        
def eye_check(data_frame,path_to_figures,path_to_results):
    eye_check_file.eye_check_func(data_frame,path_to_figures,path_to_results)

def detection_plot(data_table, path_to_data_cube, path_to_optical_images, path_to_results,
                   image_arc_width,spectrum_length,beam):
    
    detection_plot_file.detection_plot_func(image_arc_width,spectrum_length,beam,
                        path_to_optical_images,path_to_results,data_table,path_to_data_cube)
    
        