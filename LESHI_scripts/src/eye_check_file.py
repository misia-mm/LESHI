import cv2
import numpy as np
import glob
import pandas as pd
import string  
import copy
import os

def save_results():
    path_to_results_dir = path_to_results+'flagging_results_1'
    i=1
    while os.path.exists(path_to_results_dir):
        i=i+1
        path_to_results_dir = path_to_results+'flagging_results_%s'%(i)
    os.mkdir(path_to_results_dir)
    data_frame.to_csv(path_to_results_dir+'/flagging_results_table.csv',index=False)
            
    current_image = image.copy()
    cv2.rectangle(current_image, (int(image.shape[0]*0.01),int(image.shape[0]*0.95)),
                  (int(image.shape[0]*0.64),int(image.shape[0]*0.92)), (0,0,0), -1)
    cv2.putText(current_image, 'results saved to %s'%(path_to_results_dir), 
                (int(image.shape[0]*0.015),int(image.shape[0]*0.94)), cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, 
                (255, 255, 255), 1,cv2.LINE_AA)
    cv2.imshow(WINDOW_NAME, current_image)
    wait = cv2.waitKey(2500)
    cv2.imshow(WINDOW_NAME, image)
   

def keyboard_input():
    text = ""
    letters = string.ascii_lowercase + string.digits
    while True:
        key = cv2.waitKey(1)
        for letter in letters:
            if key == ord(letter):
                text = text + letter
        if key == ord("\n") or key == ord("\r"): # Enter Key
            break
    return text
      
    
def galaxy_info_text():
    global data_frame
    galaxy_table_number = images_to_browse_ids[image_number]
    
    cv2.putText(image, 'progress: %s/%s'%(image_number+1,len(images_to_browse_ids)), 
                (int(image.shape[0]*0.605),int(image.shape[0]*0.03)), cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, 
                (255, 255, 255), 1,cv2.LINE_AA)
    cv2.putText(image, 'flag: %s'%(data_frame['flag'].values[images_to_browse_ids[image_number]]), 
                (int(image.shape[0]*0.605),int(image.shape[0]*0.05)), cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, 
                (255, 255, 255), 1,cv2.LINE_AA)
    
    return


def browse_menu_text():
    cv2.putText(image, 'p - go forward', (int(image.shape[0]*0.015),int(image.shape[0]*0.03)), 
                cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, (255, 255, 255), 1,cv2.LINE_AA)
    cv2.putText(image, 'o - go backward', (int(image.shape[0]*0.015),int(image.shape[0]*0.05)), 
                cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, (255, 255, 255), 1,cv2.LINE_AA)
    cv2.putText(image, '0-9 - change flag and go forward', (int(image.shape[0]*0.015),int(image.shape[0]*0.07)), 
                cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, (255, 255, 255), 1,cv2.LINE_AA)
    cv2.putText(image, 'r - return to previous menu', (int(image.shape[0]*0.015),int(image.shape[0]*0.09)), 
                cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, (255, 255, 255), 1,cv2.LINE_AA)
    cv2.putText(image, 's - save current changes', (int(image.shape[0]*0.015),int(image.shape[0]*0.11)), 
                cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, (255, 255, 255), 1,cv2.LINE_AA)
    cv2.putText(image, 'q - quit everything', (int(image.shape[0]*0.015),int(image.shape[0]*0.13)), 
                cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, (255, 255, 255), 1,cv2.LINE_AA)

    return
    
def browse_menu():
    global image,image_number,toggle_contour,image_scale,resize_scale, toggle_help, data_frame
    image = cv2.imread(figures_sorted[images_to_browse_ids[image_number]])
    image = cv2.copyMakeBorder(image, int(window_width/10), 0, 0, 0,cv2.BORDER_CONSTANT,value=(0, 0, 0))
    browse_menu_text()
    galaxy_info_text()
    cv2.imshow(WINDOW_NAME, image)
    
    while True:
        image = cv2.imread(figures_sorted[images_to_browse_ids[image_number]])
        image = cv2.copyMakeBorder(image, int(window_width/10), 0, 0, 0,cv2.BORDER_CONSTANT,value=(0, 0, 0))
        browse_menu_text()
        galaxy_info_text()
        cv2.imshow(WINDOW_NAME, image)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('p'):    
            image_number = image_number+1
            if image_number>=len(images_to_browse_ids): image_number = image_number-len(images_to_browse_ids)
            if image_number<0: image_number = len(images_to_browse_ids)+image_number
            
                
        elif key == ord('o'):
            image_number = image_number-1
            if image_number>=len(images_to_browse_ids): image_number = image_number-len(images_to_browse_ids)
            if image_number<0: image_number = len(images_to_browse_ids)+image_number       
            

       # modify flag
        elif key == ord('0') or key == ord('1') or key == ord('2') or key == ord('3') or key == ord('4') or key == ord('5') or key == ord('6') or key == ord('7') or key == ord('8') or key == ord('9'):
            data_frame['flag'].values[images_to_browse_ids[image_number]]=chr(key)
            image_number = image_number+1
            if image_number>=len(images_to_browse_ids): image_number = image_number-len(images_to_browse_ids)
            if image_number<0: image_number = len(images_to_browse_ids)+image_number 
            
            
        elif key == ord('s'):
            save_results()
            
        elif key == ord('r'):
            break
            
        elif key == ord('q'):
            np.make_error()

    return


def image_number_menu():
    global image_number
    image = cv2.imread(figures_sorted[images_to_browse_ids[image_number]])
    image = cv2.copyMakeBorder(image, int(window_width/10), 0, 0, 0,cv2.BORDER_CONSTANT,value=(0, 0, 0))
    cv2.putText(image, 'Type to which image to jump to and click return key (enter)', 
                (int(image.shape[0]*0.015),int(image.shape[0]*0.03)), cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, 
                (255, 255, 255), 1,cv2.LINE_AA)
    cv2.putText(image, '(image numbers span the range from 0 to %s)'%(len(figures_sorted)-1), 
                (int(image.shape[0]*0.015),int(image.shape[0]*0.05)), cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, 
                (255, 255, 255), 1,cv2.LINE_AA)
    cv2.imshow(WINDOW_NAME, image)
    text = keyboard_input()
    
    try: 
        image_number_new=int(text)
        if image_number_new>=len(figures_sorted): np.make_error()
        image_number=image_number_new
        
    except: 
        image = cv2.imread(figures_sorted[images_to_browse_ids[image_number]])
        image=cv2.resize(image,(int(image_orig.shape[0]*resize_scale),int(image_orig.shape[1]*resize_scale)))
        cv2.rectangle(image, (int(image.shape[0]*0.01),int(image.shape[0]*0.04)), 
                      (int(image.shape[0]*0.25),int(image.shape[0]*0.01)), (0,0,0), -1)
        cv2.putText(image, 'WRONG NUMBER TYPED', (int(image.shape[0]*0.015),int(image.shape[0]*0.03)), 
                    cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, (0,0,255), 1,cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, image)
        wait = cv2.waitKey(2000)
        image_number = -99
    return 
        
def start_menu():
    global image, image_number, images_to_browse_ids, contours_to_browse,resize_scale
    
    while True:
        
        image = cv2.imread(figures_sorted[image_number])
        image = cv2.copyMakeBorder(image, int(window_width/10), 0, 0, 0,cv2.BORDER_CONSTANT,value=(0, 0, 0))
        
        cv2.putText(image, 'What would you like to do?', (int(image.shape[0]*0.015),int(image.shape[0]*0.03)), 
                    cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, (255, 255, 255), 1,cv2.LINE_AA)
        cv2.putText(image, '1 - browse through all figures', (int(image.shape[0]*0.015),int(image.shape[0]*0.05)), 
                    cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, (255, 255, 255), 1,cv2.LINE_AA)
        cv2.putText(image, '2 - go to specific figure', (int(image.shape[0]*0.015),int(image.shape[0]*0.07)), 
                    cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, (255, 255, 255), 1,cv2.LINE_AA)
        cv2.putText(image, 's - save current changes', (int(image.shape[0]*0.015),int(image.shape[0]*0.11)), 
                    cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, (255, 255, 255), 1,cv2.LINE_AA)
        cv2.putText(image, 'q - quit everything', (int(image.shape[0]*0.015),int(image.shape[0]*0.13)), 
                    cv2.FONT_HERSHEY_SIMPLEX,0.6*image.shape[0]/950, (255, 255, 255), 1,cv2.LINE_AA)
        
        cv2.imshow(WINDOW_NAME, image)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('1') :
            
            images_to_browse_ids = np.arange(len(figures_sorted))
            # go to browse menu
            browse_menu()
                
        
        elif key == ord('2'):
            image_number_menu()
            images_to_browse_ids = np.arange(len(figures_sorted))
            
            # go to browse menu
            if image_number!=-99: browse_menu()
            else: image_number = 0
        elif key == ord('s'):
            save_results()
            
        elif key == ord('q'):
            np.make_error()
    return   
    
def welcome_menu():
    blank_image_orig=np.zeros((int(window_width),int(window_width*0.8)))
    loop=0
    while True:
        blank_image=blank_image_orig.copy()
        key = cv2.waitKey(300) & 0xFF
        cv2.putText(blank_image, 'WELCOME', (int(blank_image.shape[0]*0.3),int(blank_image.shape[0]*0.2)), 
                    cv2.FONT_HERSHEY_SIMPLEX,2.5*blank_image.shape[0]/950, (255,255,255), 1,cv2.LINE_AA)
        cv2.putText(blank_image, 'code written by Michalina Maksymowicz-Maciata', 
                    (int(blank_image.shape[0]*0.23),int(blank_image.shape[0]*0.25)), 
                    cv2.FONT_HERSHEY_SIMPLEX,0.6*blank_image.shape[0]/950, (255,255,255), 1,cv2.LINE_AA)
        cv2.putText(blank_image, 'press ENTER to continue', (int(blank_image.shape[0]*0.32),int(blank_image.shape[0]*0.38)), 
                    cv2.FONT_HERSHEY_SIMPLEX,0.8*blank_image.shape[0]/950, (255,255,255), 1,cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, blank_image)
        if key == ord("\n") or key == ord("\r"):
            start_menu()
        elif key == ord('q'):
            np.make_error()



def eye_check_func(data_frame_input,path_to_figures,path_to_results_input):
    global image, image_number, toggle_contour, image_scale, images_to_browse_ids, data_frame, window_width,WINDOW_NAME,figures_sorted, path_to_results

    data_frame = data_frame_input
    path_to_results = path_to_results_input
    data_frame['flag'] = np.ones(len(data_frame))*(-99)
    
    figures_sorted=[]
    for source in range(len(data_frame)):
        ID = data_frame['ID'].values[source]
        try:figures_sorted.append(glob.glob(path_to_figures+'*%s*.png'%(ID))[0])
        except:pass
    image_number = 0
    # initialise the window
    WINDOW_NAME='INTERACTIVE FLAGGING'
    cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_NORMAL)
    window_width = 1600
    cv2.startWindowThread()
    
    # start the app
    try:welcome_menu()
    except: pass
        
    cv2.destroyAllWindows()
    # these are for some reason necessary, otherwise the window might not close for macs
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)


