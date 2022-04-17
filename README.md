# XTC synapse tracking

## Overview:
Main script for running all modules of the synapse tracking pipeline. 
The modular layout below means that individual components of the pipeline can
be easily replaced/improved. Currently, this includes the following sequentially:

1. Timeseries registration with ITK-Elastix package. 
           Volumetric affine registration is first performed, followed by slice-by-slice registration.
            Alternative non-rigid registration methods are also available.
2. XTC image restoration. 
            Loads trained 3D U-Net model for image enhancement.
            Requires GPU for optimal speed. If the input volume is acquired at a different
            voxel resolution, the volume is automatically rescaled (interpolated) to 
            the resolution of the training data.
3. ilastik synapse detection model. 
            Voxel-based classification to identify XTC enhanced synapses
            across all timepoints in the timeseries.
4. ***NON-FUNCTIONAL in Code Capsule environment*** Matlab-based watershed segmentation.
            Separates adjoining synapses. Currently unavailable
            due to licensing issues of installing Matlab in Code capsule environment.
5. ilastik synapse tracking model.
            Runs ilastik headless and assigns unique integer to each synapse that is
            tracked across all timepoints
6. Blood vessel detection for masking possibly obstructed synapses.
            Uses thresholding and some morphological operations to generate a binary
            segmentation of dark space (blood vessels). This masks out synapses that
            are likely to be obstructed by blood vessels. 

Expected input:

    /data/"__demo_crop_2.tif"
                *Single multipage-TIFF file (uint8) used to demo the image restoration and tracking pipeline.
                Contains a small cropped volume from a longitudinal in vivo imaging experiment
                visualizing SEP-GluA2 labelled synapses. 
Output:
    
    /results/"__quick_reference_output.png"
                * a pyplot PNG figure that contains a quick view of a small subvolume
                of the demo dataset to show how the data looks before/after restoration
                and after synapse tracking.
    /results/"___RESCALED_adj.tif"
                * single multipage-TIFF (uint8) with entire timeseries that has been
                registered and rescaled to match the resolution of the training data.
    /results/"___XTC_processed.tif"
                * single multipage-TIFF (uint8) after registeration and processing with XTC.
    /results/"___Tracking-Result.tif"
                * single multipage-TIFF (uint32) with segmentation of all synapses.
                Each tracked synapse has a unique assigned value that is consistent
                across all timepoints. For optimal visualization, we suggest loading
                this file into ImageJ/FIJI (https://imagej.net/software/fiji/) and 
                selecting the "glasbey on dark" LUT followed by adjusting the contrast
                histogram (Image>>Adjust>>Brightness/Contrast) to see the full spectrum
                of tracked synapses.
    /results/"__blood_vessel_mask.tiff"
                * single multipage-TIFF (uint8) with mask of blood vessels that is
                used in the analysis to remove obstructed synapses after tracking.



## Installation instructions:
This project requires a couple of core dependencies. Currently this has only been tested on a Linux system.

### A. Create a virtual environment for the project (highly recommended)

1. First install Anaconda: https://www.anaconda.com/products/distribution

2. Then create virtual environment

           conda create -n XTCSynEnv python=3.7 anaconda
           source activate XTCSynEnv    ### use this command to activate the virtual enviornment

3. Install Python dependencies within virtual environment

           pip install natsort itk-elastix pysimplegui csbdeep matplotlib scipy scikit-image pillow numpy natsort tifffile
    
    
### B. Installing Tensorflow (with GPU compatibility) highly recommended for efficient processing
        
1. First install tensorflow, check version needed for specific GPUs (https://www.tensorflow.org/install/gpu). 
        pip install tensorflow==2.6.0
        
2. Then install dependencies to enable Tensorflow communication with GPU. Follow instructions here: https://www.tensorflow.org/install/gpu. Note: CUDA 11.2 and cuDNN 8.1 work for tensorflow 2.6.0.


    
### C. Installing MATLAB and including path from python

1. To install MATLAB engine, first figure out where matlab is installed by entering "matlabroot" in the MATLAB cmd, then:
            
            cd /usr/local/MATLAB/R2021a/extern/engines/python
            python setup.py install
            
            
            
### D. Download checkpoint file, demo data, and ILASTIK models

1. Download checkpoint file, demo data, and ILASTIK models here: https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/yxu130_jh_edu/EnSmiQYFidVMlwfUNDvCWgQB3_mv-CeusZ3dkolZw71hwQ?e=xtWTH3
2. Then copy/paste the entire folders into the XTC main directory



## Run demo

1. Run the main function:

           python ./Synapse_detection_and_tracking_FULL_NM.py

2. Follow directions of GUI and select demo folder. Enter in voxel resolution and press continue.



            
            
