
def source_finder(data_cube, path_to_results = "./",
                       SNR_integ=3.5, SNR_channel=2.5, 
                       channel_min_len=3,
                       SNR_spec=3.5, rsqr_min=0.35,
                       int_image_len=10, int_image_load_no=10,
                       channel_start=0, channel_end=None, beam=None, 
                       bg_box_size=100, sloped_baseline = False,
                       max_dist_pix=10, max_dist_channel=10, 
                       core_no=None, test_hist=False ):
    """Start the LESHI Source Finder. 
    
    Parameters
    ----------
    data_file : str
        Path to the datacube to perform source finding on.

    path_to_results : str
        Path to the directory where the results will be saved (default = "./").

    SNR_integ : int or float
        Threshold signal-to-noise ratio of each source on the integrated image (default = 3.5).

    SNR_channel : int or float
        Threshold signal-to-noise ratio of each source on the channel image (default = 2.5).

    channel_min_len : int
        Minimum number of channels that the signal persists for (default = 3).

    SNR_spec : int or float
        Threshold signal-to-noise ratio of each source on the spectrum (default = 3.5).

    rsqr_min : int or float
        Minimum value of the R^2 parameter (ranging from 0 to 1) quantifying how well the fitted Gaussian
        function fits the spectrum (default = 0.35).

    int_image_len : even int
        Channel width of the cube slab to integrate into a moment-0 map. Has to be an even number. 
        Should be equal to the width of the weakest expected signals in channels (default = 10).
    
    int_image_load_no : int
        Number of  moment-0 maps (integarted images) to process at the same time, limited by the available RAM (default = 10).

    channel_start : int
        Channel of the data cube where source finding should start (default = 0).

    channel_end : int
        Channel of the data cube where source finding should end, if None, the source finding will end 
        at the end of the data cube (default = None).

    beam : int or float
        Diameter of the synthesised beam of the data cube in pixels, only needed if this information cannot
        be read from the data by SpectralCube package (defulat = None).

    bg_box_size : int
        Width of the box in pixels used to calculate local background noise (default = 100).

    sloped_baseline : bool
        If set to True, the code will attempt to subtract the fitted noise baseline (default = False).

    max_dist_pix : int
        Maximal distance in pixels between two sources to be associated into one source (defualt = 10).

    max_dist_channel : int
        Maximal distance in channels between two sources to be associated into one source (defualt = 10).

    test_hist : bool
        Whether to output the full testing history or not, showing where each source failed or passed, useful for troubleshooting, 
        however may result in very large tables (defualt = False).

    core_no : int
        Number of cores to use for parallelisation 
    
    Returns
    ------
    output : pandas data frame
        Function returns the data table with found associated sources, the table is also saved in the specified directory.
    """



          
    print( '    @   @                                            @.  ..    ')  
    print( '@   @@  @.                                           @. .@.  .@')  
    print( '@@   @@.@.        .@.                     @:         @@@@.  .@.')  
    print( ' @@.  @@@.          @.                   @.          @@@.  .@. ')  
    print( '  @@.  @@.          .@.                 @@           @@.  @@.  ')  
    print( '   .@@@@@@@          @@@.             .@@          .@@@@@@@    ')  
    print( '       .@@@@@@.....@@@@@@@@@@.  .@@@@@@@@@@.    .@@@@@..       ')  
    print( '            .@@@@@@@..           .  . ....@@@@@@@@.            ') 
    print(r"       _____     ________   ______   ____  ____  _____ ") 
    print(r"      |_   _|   |_   __  |.' ____ \ |_   ||   _||_   _|") 
    print(r"        | |       | |_ \_|| (___ \_|  | |__| |    | |  ") 
    print(r"        | |   _   |  _| _  _.____`.   |  __  |    | |  ") 
    print(r"       _| |__/ | _| |__/ || \____) | _| |  | |_  _| |_ ") 
    print(r"      |________||________| \______.'|____||____||_____|") 
    print( '         Line    Emisssion  Source -  Hunting  Integrator')
    print('\n')
    print('for more information see: https://github.com/misia-mm/LESHI')
    print('please report any issues to: michalina.maksymowicz.maciata@gmail.com')
    print('\n')
    print('STARTING THE LESHI SOURCEFINDER')
    from .src import source_finder_file
    if int_image_len % 2 == 0:
        df = source_finder_file.source_finder_func(data_cube, path_to_results,
                       SNR_integ, SNR_channel, 
                       channel_min_len,
                       SNR_spec, rsqr_min,
                       int_image_len,int_image_load_no,
                       channel_start,channel_end,beam,test_hist, 
                       bg_box_size,max_dist_pix,max_dist_channel,sloped_baseline,core_no )
        return df
    else:
        print('int_image_len should be an even number')
        return None

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

    filters : string array
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
    from .src import get_opt_cutout_file
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
    from .src import eye_check_file
    eye_check_file.eye_check_func(data_frame,path_to_figures,path_to_results)

def detection_plot(data_table, path_to_data_cube, path_to_optical_images, path_to_results,
                   image_arc_width,spectrum_length,beam):
    from .src import detection_plot_file
    detection_plot_file.detection_plot_func(image_arc_width,spectrum_length,beam,
                        path_to_optical_images,path_to_results,data_table,path_to_data_cube)

def source_velocity_width(data_table, path_to_radio_file, path_to_contours='./', path_to_results='./', core_no=None):
    from .src import velocity_width_file

    df = velocity_width_file.velocity_width_script(data_table, path_to_radio_file, path_to_contours, path_to_results, core_no)
    return df

def source_hi_mass(data_table, data_cube, path_to_contours='./', path_to_results='./', core_no=None):
    from .src import hi_mass_file

    df = hi_mass_file.hi_mass_script(data_table, data_cube, path_to_contours, path_to_results, core_no)
    return df
    
def source_extent(data_table, data_cube, min_window_width_pix=100, min_diameter_pix=None,path_to_results='./',core_no=None):
    from .src import source_extent_file
  
    df = source_extent_file.source_extent_script(data_table, data_cube, min_window_width_pix, min_diameter_pix, path_to_results, core_no)
    return df

def emission_plot(data_table, data_cube, path_to_optical_images='./optical_images/', path_to_results='./',
                   image_arc_width=100,spectrum_length=200,beam=None,filetype='png'):
    from .src import emission_plot_file
    emission_plot_file.emission_plot_script(data_table,data_cube,path_to_optical_images,path_to_results,image_arc_width,spectrum_length,beam,filetype)

