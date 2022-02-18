# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:10:54 2019

@author: tiger


Installation notes:
    1) Anaconda
    2) pip install (everything) + 
    pip install matplotlib scipy scikit-image pillow numpy natsort opencv-python tifffile keras pandas    
    pip install csbdeep numba

    mahotas? - failed
    conda config --add channels conda-forge
    conda install mahotas
    
    pip install skan *** NEW!!! allows skeleton analysis
    
    Graphics card driver
    CUDA Toolkit ==> needs VIsual studio (base package sufficient???)
    CuDnn SDK ==> 
    
    Ignore step about putting cudnn with Visual Studio

"""
""" Install instructions for Nature Methods:
        1. pip install tensorflow
    
"""
"""
     SimpleElastix installation:
               - install in path name WITHOUT SPACES (and not too many characters either!!!)
               - have visual studio C++ package
               - ***install target language dependencies first
                    sudo apt-get install cmake swig monodevelop r-base r-base-dev ruby ruby-dev python python-dev tcl tcl-dev tk tk-dev
               - run windows 64x native shell as ADMINISTRATOR
               - follow the GUI instructions to the DOT!!! Even deselecting all other options in the GUI to only keep Python wrappings
               - remember to go to SUPERBUILD folder within simpleElastix when setting cMake path
               - at the end, must python install
                    ==> will need to move file
                         _SimpleITK.pyd from the ...\Python\ directory to ...\Python\Packaging. 
                    and then run the installation INSIDE CONDA prompt!!!
                         or else will not install to anaconda Spyder
                    
               
               

"""


""" VirtualEnv

        conda activate XTCenv
"""


#-------------------------------------------------------------------------------------------------------------------------------
# Load dependencies and connect to GPU
#-------------------------------------------------------------------------------------------------------------------------------     

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
gpu = 1
tf.config.experimental.set_memory_growth(gpus[gpu], True)
tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean


from functional.plot_functions_CLEANED import *
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.registration_functions import *
from functional.GUI import *

from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import tifffile as tiff

from csbdeep import data
from csbdeep import io
from keras.models import load_model
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE


import matlab.engine
import shutil




#-------------------------------------------------------------------------------------------------------------------------------
# Define user inputs and input path
#-------------------------------------------------------------------------------------------------------------------------------          
""" (1) Read in timeseries
        - should be default 0.095 um/px in XY and 1 um/px in Z
        ***switch to memmap tiff if files are too large...

"""
input_path = os.getcwd() + '/demo/';

#print("Input must be uint8 ")
XY_expected = 0.095;                            ### expected default resolution in microns/px
Z_expected = 1;                                 ### expected default resolution in microns/px
#list_folder, XY_res, Z_res = XTC_track_GUI(default_XY=str(XY_expected), default_Z=str(Z_expected))    ### opens a GUI and prompts user for the metadata of the file


### for Code Ocean, explicitly state the resolution and input path
list_folder = [input_path]
XY_res = XY_expected
Z_res = 3


sys.path.append('/media/user/storage/ilastik-1.3.3post3-Linux/ilastik-meta/ilastik')
import shlex, subprocess
# ILASTIK_path = '/media/user/storage/ilastik-1.3.3post3-Linux/run_ilastik.sh'
# ### newly trained without median filter!
# ILASTIK_detector ='/media/user/storage/Data/(3) Huganir_lab_CARE/ILASTIK_models/For ILASTIK training NO median filter/CARE_invivo_trained_SEP_seg_NO_MEDIAN_FILTER.ilp'
# ILASTIK_tracker = '/media/user/storage/Data/(3) Huganir_lab_CARE/ILASTIK_models/Tracking_WITH_LEARNING_CARE_trained_SEP_seg.ilp'
    


ILASTIK_path = os.getcwd() + '/ilastik-1.3.3post3-Linux/run_ilastik.sh'
### newly trained without median filter!
ILASTIK_detector = os.getcwd() + '/ILASTIK_models/CARE_invivo_trained_SEP_seg_NO_MEDIAN_FILTER.ilp'
ILASTIK_tracker = os.getcwd() + '/ILASTIK_models/Tracking_WITH_LEARNING_CARE_trained_SEP_seg.ilp'
    

for input_path in list_folder:    

    #-------------------------------------------------------------------------------------------------------------------------------
    # REGISTRATION
    #-------------------------------------------------------------------------------------------------------------------------------

    """ (2) Perform registration with SimpleElastix of each timepoint to the preceding timepoint
    after it is registered to original fixed image
    
        ***REMOVE THE "RAW" image for demo here
    """
    foldername = input_path.split('/')[-2]
    sav_dir = input_path + '/' + foldername + '_OUTPUT'
    
    try:
        # Create target Directory
        os.mkdir(sav_dir);    print("\nSave directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("\nSave directory " , sav_dir ,  " already exists, will overwrite")
        
    sav_dir = sav_dir + '/'
    
    
    print('Loading input timeseries')
    images = glob.glob(os.path.join(input_path,'*.tif'))
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    
    all_adj_im = tiff.imread(images[0])

    filename = images[0].split('/')[-1].split('.')[0:-1]
    filename = '.'.join(filename)
        
    ### set first frame to be the fixed one (no need to register)
    all_reg = np.zeros(np.shape(all_adj_im));       all_reg[0] = all_adj_im[0]       
    ### register the rest to the previous frame
    print('Beginning image registration')
    for i in range(1, len(all_adj_im)):
        
        ### For first frame, set fixed as baseline
        if i == 1:
            fixed_im = all_adj_im[0]
            first_frame_fixed = np.copy(fixed_im)
            
        moving_im = all_adj_im[i]
            
        ### (A) First do registration overall
        registered_im, transformix = register_ELASTIX(fixed_im, moving_im, reg_type='affine')  ### can also be "nonrigid"
        
        ### (B) Then do registration slice by slice
        reg_slices, reapply_slices, transformix_slice = register_by_SLICE(first_frame_fixed, registered_im, reapply_im=[], reg_type='affine')
        reg_slices = np.vstack(reg_slices)


        all_reg[i] = reg_slices

        ### reset the fixed_im to be registered_im
        fixed_im = np.copy(registered_im)
            
        print('Registering slice: ' + str(i) + ' of total: ' + str(len(all_adj_im)))
    
    
    tiff.imwrite(sav_dir + filename + '_REGISTERED_adj.tif', np.asarray(np.expand_dims(all_reg, axis=2), dtype=np.uint8),
                  imagej=True, resolution=(10.5263157895, 10.5263157895),
                  metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})


    #-------------------------------------------------------------------------------------------------------------------------------
    # XTC restoration
    #-------------------------------------------------------------------------------------------------------------------------------
    
    """ (2) Now scale Z-dimension and apply XTC restoration """
    model = CARE(config=None, name='./Checkpoints_TF/4_HUGANIR_LIVE_FULLY_down_COMBINED')
    model.load_weights('weights_best.h5')
    

    ### Scale images to default resolution if user resolution does not matching training
    XY_scale = float(XY_res)/XY_expected
    if XY_scale < 1: print('Image XY resolution does not match training resolution, and will be downsampled by: ' + str(round(1/XY_scale, 2)))
    elif  XY_scale > 1: print('Image XY resolution does not match training resolution, and will be upsampled by: ' + str(round(XY_scale, 2)))


    Z_scale = float(Z_res)/Z_expected
    if Z_scale < 1: print('Image Z resolution does not match training resolution, and will be downsampled by: ' + str(round(1/Z_scale, 2)))
    elif  Z_scale > 1: print('Image Z resolution does not match training resolution, and will be upsampled by: ' + str(round(Z_scale, 2)))
    
    

    all_CARE = np.zeros([all_reg.shape[0], int(all_reg.shape[1] * Z_scale), int(all_reg.shape[2] * XY_scale), int(all_reg.shape[3] * XY_scale)])
    for frame_id in range(len(all_reg)):
                
        input_im = all_reg[frame_id]

        if XY_scale != 1 or Z_scale != 1:
            print('Rescaling before XTC restoration')
            input_im = rescale(input_im, [Z_scale, XY_scale, XY_scale], anti_aliasing=True)   ### rescale the images
            input_im = ((input_im - input_im.min()) * (1/(input_im.max() - input_im.min()) * 255)).astype('uint8')   ### rescale to 255
            print('Successfully rescaled, beginning XTC restoration')
            
        else:
            print('No rescaling required, beginning XTC restoration')
                
       
        """ Run XTC model inference """
        pred_med_snr = model.predict(input_im, 'ZYX', n_tiles=(2,4,4))
    
        # Gets rid of weird donut holes if you normalize it           
        pred_med_snr[pred_med_snr < 0] = 0   ### set negative values to be 0
        pred_med_snr[pred_med_snr >= 255] = 255             
         
        all_CARE[frame_id] = pred_med_snr
        
    ### must epxand dimensions to save properly!
    tiff.imwrite(sav_dir + filename + '_REGISTERED_CARE_processed.tif', np.asarray(np.expand_dims(all_CARE, axis=2), dtype=np.uint8),
                 imagej=True, resolution=(10.5263157895, 10.5263157895),
                 metadata={'spacing': 0.33, 'unit': 'um', 'axes': 'TZCYX'})
    

    #-------------------------------------------------------------------------------------------------------------------------------
    # ILASTIK synapse detection
    #-------------------------------------------------------------------------------------------------------------------------------    
    
    """ (3) Run ILASTIK headless for synapse detection 
    ************CANNOT HAVE ILASTIK BE RUNNING AT THE SAME TIME in GUI form!!! """    
    
    sav_dir_tmp = input_path + '/' + foldername + '_tmp_ILASTIK'
    try:
        # Create target Directory
        os.mkdir(sav_dir_tmp)
        print("\nSave directory " , sav_dir_tmp ,  " Created ") 
    except FileExistsError:
        print("\nSave directory " , sav_dir_tmp ,  " already exists")
    sav_dir_tmp = sav_dir_tmp + '/'
    
    
    all_ILASTIK = np.zeros(np.shape(all_CARE))
    for frame_id in range(len(all_CARE)):
                
        input_im = all_CARE[frame_id]
        tmp_name = sav_dir_tmp + filename + '_tmp_for_ILASTIK_' + str(frame_id) + '.tif'
        
        ### save a temporary file
        tiff.imsave(tmp_name, np.asarray(input_im, dtype=np.uint8))
        
        ### run ILASTIK detector
        subprocess.run([
             ILASTIK_path,
            '--headless',
            '--export_source=simple segmentation',
            '--project=' + ILASTIK_detector,
            '--raw_data=' + tmp_name,
            '--output_filename_format=' + sav_dir_tmp + filename + '_tmp_for_ILASTIK_' + str(frame_id) + '_ILASTIK_seg'
        ])
        
        ### invert image
        output_ILASTIK = tiff.imread(sav_dir_tmp + filename + '_tmp_for_ILASTIK_' + str(frame_id) +  '_ILASTIK_seg.tiff')
        output_ILASTIK[output_ILASTIK == 2] = 0
        output_ILASTIK[output_ILASTIK == 1] = 255
        
  
        ### NEED TO DELETE PREVIOUS 2 temporary image files!!!
        os.remove(tmp_name)
        os.remove(sav_dir_tmp + filename + '_tmp_for_ILASTIK_' + str(frame_id) +  '_ILASTIK_seg.tiff')

        tiff.imwrite(sav_dir_tmp + filename +  '_tmp_for_ILASTIK_'  + str(frame_id) + '_REGISTERED_detect_ILASTIK.tif', np.asarray(output_ILASTIK, dtype=np.uint8))
    


    #-------------------------------------------------------------------------------------------------------------------------------
    # MATLAB watershed
    #-------------------------------------------------------------------------------------------------------------------------------    
    
    """ (4) Use command line to call matlab, will then find all the ILASTIK detections in temporary folder above.
            Will automatically then run watershed segmentation. Output is a series of float32 TIFFs
    """
    print('Running watershed in MATLAB')
    eng = matlab.engine.start_matlab()
    s = eng.genpath('./MATLAB_functions/')
    eng.addpath(s, nargout=0)
    eng.main_Huganir_watershed_SEP_func(sav_dir_tmp, nargout=0)
    eng.quit()

    ###  Combine output into a single TIFF stack for subsequent analysis. Also delete temporary folder tmp_ILASTIK
    images_w = glob.glob(os.path.join(sav_dir_tmp,'*_watershed_seg.tif'))
    images_w.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    
    watershed_stack = []
    for im_name in images_w:
        im = tiff.imread(im_name)
        watershed_stack.append(im)
        
    watershed_stack = np.asarray(watershed_stack)
    tiff.imwrite(sav_dir + filename + '_REGISTERED_WATERSHED.tif', np.asarray(np.expand_dims(watershed_stack, axis=2), dtype=np.float32),
                 imagej=True, resolution=(10.5263157895, 10.5263157895),
                 metadata={'spacing': 0.33, 'unit': 'um', 'axes': 'TZCYX'})
    

 

    #-------------------------------------------------------------------------------------------------------------------------------
    # ILASTIK synapse tracking
    #-------------------------------------------------------------------------------------------------------------------------------   
    
    """ (5) Run synapse tracking 
     ^^^need GUROBI installed???
    """
    images = glob.glob(os.path.join(sav_dir,'*_adj.tif'))
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,CARE=i.replace('_adj.tif','_CARE_processed.tif'), watershed=i.replace('_adj.tif','_WATERSHED.tif')) for i in images]
     
    CARE_name = examples[0]['CARE']      
    RAW_name = examples[0]['input']  
    watershed_name = examples[0]['watershed']  
    
    """ (7) Perform tracking, using ILASTIK """     
    """ ************CANNOT HAVE ILASTIK BE RUNNING AT THE SAME TIME in GUI form!!! """
    subprocess.run([
         ILASTIK_path,
        '--headless',
        '--project=' + ILASTIK_tracker,
        '--raw_data=' + CARE_name,
        '--binary_image=' + watershed_name,
    ])
    tracking_result = tiff.imread(sav_dir + '/' + filename +  '_Tracking-Result.tiff')
        
    ### must epxand dimensions to save properly!
    tiff.imwrite(sav_dir +  '/' + filename + '_Tracking-Result.tif', np.asarray(np.expand_dims(tracking_result, axis=2), dtype=np.float32),
                 imagej=True, resolution=(10.5263157895, 10.5263157895),
                 metadata={'spacing': 0.33, 'unit': 'um', 'axes': 'TZCYX'})
    
    
    
    ### then delete the temporary file
    os.remove(sav_dir + '/' + filename +  '_Tracking-Result.tiff')
    
    
    ### Delete temporary folder tmp_ILASTIK
    shutil.rmtree(sav_dir_tmp)  
    
    













