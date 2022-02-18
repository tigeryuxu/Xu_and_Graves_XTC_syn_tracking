function [L] = DAPIcount_3D(intensityValueDAPI, DAPIsize, DAPImetric, enhance, DAPI_bb_size, binary)

% Cell DAPI nuclei counter by performing standard binarization and watershed operations:
% - removes high intensity pixels (> 0.15) to get better binarization
% - "optional" enhacement ==> imadjust
% - edge exclusion (no DAPI along edges of image
% - watershed (optional imimposemin)
% - roundness exclusion criteria
% 
% Also includes:
% - artifiact removal
% 
% Inputs:
% 
% 
% Outputs:
%             mat == matrix of centroids of identified DAPI objects
%             objDAPI == PixelIdxList of all identified DAPI objects
%             bw == logical image of DAPI objects

%% Smooth
if binary == 'N'
    I = imgaussfilt3(intensityValueDAPI, 1);
    
    %% Threshold
    %% Subtract background:
    if enhance == 'Y'
        background = imopen(I,strel('disk',DAPI_bb_size));
        I2 = imsubtract(I, background);
        I = I2;
        I = histeq(I);
    else
        background = imopen(I,strel('disk',DAPI_bb_size));
        I2 =  imsubtract(I, background);
        I = I2;
    end
else
    I = intensityValueDAPI;
end

%% Binarize
bw = binarize_3D_otsu(I);
%bw_global = imbinarize(I, 0.3);
%figure; volshow(im2double(bw_global));

%thresh = adaptthresh(I, 0.1, 'NeighborhoodSize', 2*floor(size(I)/100)+1, 'Statistic', 'gaussian');
%bw = imbinarize(I, thresh);
%figure; volshow(im2double(bw))


%% Find min mask
%bw = ~bwareaopen(~bw, 10);  % clean
D = -bwdist(~bw);  % EDT
%mask = imregionalmin(D);   % Extended minima
mask = imextendedmin(D, 0.5);   % Extended minima


%% Watershed segmentation by imposing minima (NO NEED FOR imposing minima) probably b/c gaussfilt renders this useless
D2 = imimposemin(D, mask);

Ld2 = watershed(D2);
bw3 = bw;
bw3(Ld2 == 0) = 0;
bw = bw3;

L = bwlabeln(bw);


end

