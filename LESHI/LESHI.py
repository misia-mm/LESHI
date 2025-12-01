from .src import eye_check_file
from .src import source_finder_file
from .src import get_opt_cutout_file
from .src import detection_plot_file

def source_finder(data_file, path_to_results,
                       SNR_integrated_image_threshold_input, SNR_channel_frame_threshold_input, 
                       signal_persistence_threshold_input,
                       signal_length_max_input,
                       SNR_spectrum_threshold_input, rsqr_threshold_input,
                       int_image_length_input,int_image_load_number_input,
                       cube_start_input,cube_end_input,beam,output_test_history, 
                       bg_box_size_input,max_dist_pix_input,max_dist_channel_input ):

    
    source_finder_file.source_finder_func(data_file, path_to_results,
                       SNR_integrated_image_threshold_input, SNR_channel_frame_threshold_input, 
                       signal_persistence_threshold_input,
                       signal_length_max_input,
                       SNR_spectrum_threshold_input, rsqr_threshold_input,
                       int_image_length_input,int_image_load_number_input,
                       cube_start_input,cube_end_input,beam,output_test_history, 
                       bg_box_size_input,max_dist_pix_input,max_dist_channel_input )

def get_opt_cutout(ID,ra,dec,width_arc,filters,survey,path_to_images):
    if survey == 'LegacySurvey':
        get_opt_cutout_file.get_LG_image(ID,ra,dec,width_arc, filters,path_to_images)
    if survey == 'HSC':
        if hasattr(ra, '__iter__'):
            get_opt_cutout_file.get_HSC_image_in_bulk(ID,ra,dec,width_arc, filters,path_to_images)
        else:
            get_opt_cutout_file.get_HSC_image(ID,ra,dec,width_arc, filters,path_to_images)


        
def eye_check(data_frame,path_to_figures,path_to_results):
    eye_check_file.eye_check_func(data_frame,path_to_figures,path_to_results)


def detection_plot(image_arc_width_input,spectrum_length,beam_input,
                   path_to_optical_images,path_to_results,data_table,path_to_data_cube):
    detection_plot_file.detection_plot_func(image_arc_width_input,spectrum_length,beam_input,
                        path_to_optical_images,path_to_results,data_table,path_to_data_cube)
    
        