#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 13:36:16 2022

@author: user
"""


import sys,os
#sys.path.append('/home/user/build/SimpleITK-build/Wrapping/Python/')
#sys.path.append('./SimpleITK-build/Wrapping/Python/')
#import SimpleITK as sitk
import numpy as np

import itk as itk






""" Updated all code to ITK-elastix which is so much easier to install and use"""
def register_ELASTIX_ITK(fixed_im, moving_im, reg_type='affine'):

    fixed_image = np.asarray(fixed_im, dtype=np.float32)
    moving_image = np.asarray(moving_im, dtype=np.float32)
    
    
    
    
    parameter_object = itk.ParameterObject.New()
    parameter_map_rigid = parameter_object.GetDefaultParameterMap('translation')
    parameter_object.AddParameterMap(parameter_map_rigid)    
        
    
    if reg_type == 'affine':
        parameter_map_rigid = parameter_object.GetDefaultParameterMap('affine')
        parameter_object.AddParameterMap(parameter_map_rigid)   
    
    else:
        print('required reg_type: "affine" or None')
            
    # Call registration function
    registered_im, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object)    
        
    
    ### or do better normalization than this?
    registered_im[registered_im < 0] = 0   ### set negative values to be 0
    registered_im[registered_im >= 255] = 255       
    registered_im = np.asarray(registered_im, dtype=np.uint8)
    
    
    return registered_im, result_transform_parameters





""" For slice-by-slice registration

            ***reapply with transformix currently does not work

 """
def register_by_SLICE_ITK(fixed_im, moving_im, reg_type='affine', reapply_im=[]):
    reg_slices = []
    reapply_slices = []
    for slice_id in range(len(fixed_im)):
        
        
        slice_fixed = fixed_im[slice_id, :, :]
        slice_move = moving_im[slice_id, :, :]
        
        
        reg_im, transformix = register_ELASTIX_ITK(slice_fixed, slice_move, reg_type='affine')  ### can also be "nonrigid"

        reg_im = np.expand_dims(reg_im, 0)
        reg_slices.append(reg_im)
        
        if len(reapply_im) > 0:  ### if nothing to reapply to
            slice_reapply = reapply_im[slice_id, :, :]
            
            reapply_reg = reapply_transform_map(slice_reapply, transformix)
            reapply_reg = np.expand_dims(reapply_reg, 0)
            reapply_slices.append(reapply_reg)


    return reg_slices, reapply_slices, transformix
    


""" To map registration onto another volume """
def reapply_transform_map_ITK(raw_moving, transformix):

    transformix.SetMovingImage(sitk.GetImageFromArray(raw_moving))
    transformix.Execute()
    raw_reg = sitk.GetArrayFromImage(transformix.GetResultImage())
    raw_reg[raw_reg < 0] = 0   ### set negative values to be 0
    raw_reg[raw_reg >= 255] = 255      


    return raw_reg



# """ For normal registration """
# def register_ELASTIX(fixed_im, moving_im, reg_type='affine'):

#     fixedImage = sitk.GetImageFromArray(fixed_im)
#     movingImage = sitk.GetImageFromArray(moving_im)
    
#     parameterMapVector = sitk.VectorOfParameterMap()
    
#     ### for rigid
#     parameterMapVector.append(sitk.GetDefaultParameterMap('translation'))
    
    
#     ### for affine only
#     if reg_type == 'affine':
#         parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    
#     elif reg_type == 'nonrigid':        
#         ### for non-rigid
#         parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
#         parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
        
#     else:
#         print('required reg_type: "affine" or "nonrigid"')
    
#     elastixImageFilter = sitk.ElastixImageFilter()
#     elastixImageFilter.SetFixedImage(fixedImage)
#     elastixImageFilter.SetMovingImage(movingImage)
#     elastixImageFilter.SetParameterMap(parameterMapVector)
#     elastixImageFilter.LogToFileOff()
#     elastixImageFilter.LogToConsoleOff()
#     elastixImageFilter.SetLogToConsole(False)

#     elastixImageFilter.Execute()
    
#     resultImage = elastixImageFilter.GetResultImage()
    
#     registered_im = sitk.GetArrayFromImage(resultImage)
    
    
#     registered_im[registered_im < 0] = 0   ### set negative values to be 0
#     registered_im[registered_im >= 255] = 255       
    
    
#     ### also return transform so can do on new image
#     transformParameterMap = elastixImageFilter.GetTransformParameterMap()

#     transformix = sitk.TransformixImageFilter()
#     transformix.SetTransformParameterMap(transformParameterMap)
        
    
#     return registered_im, transformix




# """ For slice-by-slice registration """
# def register_by_SLICE(fixed_im, moving_im, reg_type='affine', reapply_im=[]):
#     reg_slices = []
#     reapply_slices = []
#     for slice_id in range(len(fixed_im)):
        
        
#         slice_fixed = fixed_im[slice_id, :, :]
#         slice_move = moving_im[slice_id, :, :]
        
        
#         reg_im, transformix = register_ELASTIX(slice_fixed, slice_move, reg_type='affine')  ### can also be "nonrigid"

#         reg_im = np.expand_dims(reg_im, 0)
#         reg_slices.append(reg_im)
        
#         if len(reapply_im) > 0:  ### if nothing to reapply to
#             slice_reapply = reapply_im[slice_id, :, :]
            
#             reapply_reg = reapply_transform_map(slice_reapply, transformix)
#             reapply_reg = np.expand_dims(reapply_reg, 0)
#             reapply_slices.append(reapply_reg)


#     return reg_slices, reapply_slices, transformix
    


# """ To map registration onto another volume """
# def reapply_transform_map(raw_moving, transformix):

#     transformix.SetMovingImage(sitk.GetImageFromArray(raw_moving))
#     transformix.Execute()
#     raw_reg = sitk.GetArrayFromImage(transformix.GetResultImage())
#     raw_reg[raw_reg < 0] = 0   ### set negative values to be 0
#     raw_reg[raw_reg >= 255] = 255      


#     return raw_reg














            