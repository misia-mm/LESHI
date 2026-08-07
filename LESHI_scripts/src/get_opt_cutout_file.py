import pandas as pd
import numpy as np
import os
import shutil
import glob

# get LS image
def get_LS_image(ID_array,ra_array,dec_array,width_arc_array, filters,path_to_data):   
    for source in range(len(ra_array)):
        ID = ID_array[source]
        ra = round(ra_array[source],4)
        dec = round(dec_array[source],4) # coordinates of the cutout
        width_arc = width_arc_array[source]
        semi_width_deg = round(width_arc/3600/2,6)
        pixscale=0.26
        size = width_arc/pixscale
        
        if size>3000:
            size=3000
            pixscale = width_arc/3000

        for photo_filter in filters:
            # formulate the link from which wget will download the image, the link depends on the coordinates and size of the cutout, for example:
            #       'http://www.legacysurvey.org/viewer/fits-cutout?ra=3.5167&dec=-23.1827&layer=ls-dr10&pixscale=0.262&size=5000&bands=g'
            link = 'http://www.legacysurvey.org/viewer/fits-cutout?ra=%s&dec=%s&layer=ls-dr10&pixscale=%s&size=%s&bands=%s'%(ra,dec,pixscale,int(size),photo_filter.lower())
                   
            # formulate the file name in which wget will save the file, for example:
            # G_image_cutout_ID.fits
            file_name = '%s/LS_images_filter_%s/%s_%s_image_cutout.fits'%(path_to_data,photo_filter,photo_filter,ID)
            
            # create directories for the filter images
            path_exist = os.path.exists('%s/LS_images_filter_%s'%(path_to_data,photo_filter))
            if not path_exist:
               os.makedirs('%s/LS_images_filter_%s'%(path_to_data,photo_filter))
    
            # download the images
            os.system('wget --no-verbose -O %s "%s"'%(file_name,link))       
        print('Finished dowloading cutouts')


# get HSC image
def get_HSC_image(ID,ra,dec,width_arc, filters,path_to_data):   
    ra = round(ra,4)
    dec = round(dec,4) # coordinates of the cutout
    semi_width_deg = round(width_arc/3600/2,6)
       
    for photo_filter in filters:
        
        # formulate the link from which wget will download the image, the link depends on the coordinates and size of the cutout, for example:
        #       https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/cgi-bin/cutout?ra=150.3135&dec=2.3063&sw=0.013889&sh=0.013889&type=coadd&image=on&filter=HSC-G&tract=&rerun=pdr3_dud_rev
        link = 'https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout?ra=%s&dec=%s&sw=%s&sh=%s&type=coadd&image=on&filter=HSC-%s&tract=&rerun=pdr2_wide'%(ra,dec,semi_width_deg,semi_width_deg,photo_filter)
               
        # formulate the file name in which wget will save the file, for example:
        # G_image_cutout_ID.fits
        file_name = '%s/HSC_images_filter_%s/%s_%s_image_cutout.fits'%(path_to_data,photo_filter,photo_filter,ID)
        
        # create directories for the filter images
        path_exist = os.path.exists('%s/optical_images_filter_%s'%(path_to_data,photo_filter))
        if not path_exist:
           os.makedirs('%s/HSC_images_filter_%s'%(path_to_data,photo_filter))

        # download the images
        username = 'leshi' # these login details are mine, however it is very straightforward to register
        password = 'i+3kxud2jExtsQhdpYoE5J8FG7yzc6OlTABXdZ14'
        os.system('wget --no-verbose -O %s --user=%s --password=%s "%s"'%(file_name,username,password,link))       
    print('Finished dowloading cutouts')

def get_HSC_image_in_bulk(ID_array,ra_array,dec_array,width_arc_array, filters,path_to_data):   
    target_table = pd.DataFrame(data={'rerun':np.full(len(ra_array),'pdr2_wide'),'ra':np.round(ra_array,4),'dec':np.round(dec_array,4),'sw':np.round(width_arc_array/2/3600,6),'sh':np.round(width_arc_array/2/3600,6)})
    
    for photo_filter in filters:
        # create directories for the filter images
        path_exist = os.path.exists('%s/HSC_images_filter_%s'%(path_to_data,photo_filter))
        if not path_exist:
           os.makedirs('%s/HSC_images_filter_%s'%(path_to_data,photo_filter))

        # prepare a table to upload coordinates
        target_table['filter'] = np.full(len(ra_array),'HSC-'+photo_filter)
        target_table.to_csv('temp_target_table.txt',index=False,sep='\t')
        with open('temp_target_table.txt','r+') as file:
            file_data = file.read()
            file.seek(0, 0)
            file.write('#?' + '\t' + file_data)
            file.close()
            
        # download the images
        username = 'leshi' # these login details are mine, however it is very straightforward to register
        password = 'i+3kxud2jExtsQhdpYoE5J8FG7yzc6OlTABXdZ14'
        curl_command = 'curl https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout --form list=@temp_target_table.txt --user %s:%s | tar xvf -'%(username,password)
        os.system(curl_command)   

        # move and change file names appropriately
        folder = glob.glob('./*arch*')[0]
        for source in range(len(ra_array)):
            try:
                file = glob.glob(folder+'/%s-cutout*'%(source+2))[0]

                # formulate the file name to which save the file, for example:
                # G_ID_12_1234_image_cutout.fits
                file_name = '%s/HSC_images_filter_%s/%s_%s_image_cutout.fits'%(path_to_data,photo_filter,photo_filter,ID_array[source])
                os.system('mv %s %s'%(file,file_name))
            except:
                print('missing file')
                file_name = '%s/HSC_images_filter_%s/%s_%s_image_cutout_fail.txt'%(path_to_data,photo_filter,photo_filter,ID_array[source])
                with open(file_name, "w") as text_file:
                    text_file.write("failed to download the image")
                
        # clean up
        os.system('rm -rf %s'%(folder))
        os.system('rm temp_target_table.txt')

    print('Finished dowloading cutouts')



