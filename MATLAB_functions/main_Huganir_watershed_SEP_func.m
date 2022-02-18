
function [] = main_Huganir_watershed_SEP_func(foldername)

%% Need to add for 3D:
% 1) ridges2lines ==> needs to separate into 3 types of angles (x,y,z) ==>
% OR just take it out completely???
% 2) must fix rest of code to adapt to nanofiber cultures
% 3) must fix all "disk" dilations to "spheres"



% IF WANT TO ADJUST/Elim background, use ImageJ
% ==> 1) Split channels, 2) Select ROI, 3) go to "Edit/Clear outside"
% 4) Merge Channels, 5) Convert "Stack to RGB", 6) Save image

%***Note for Annick: ==> could also use ADAPTHISTEQ MBP for area at end...
%but too much???

% ***ADDED AN adapthisteq to imageAdjust.mat... 2019-01-24

%% Main function to run heuristic algorithm
%opengl hardware;
close all;

cur_dir = pwd;
addpath(strcat(cur_dir))  % adds path to functions
cd(cur_dir);

%% Initialize
%foldername = uigetdir();   % get directory

%% Run Analysis
cd(foldername);   % switch directories
nameCat = '*tif*';
fnames = dir(nameCat);

trialNames = {fnames.name};
numfids = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently

%% Read in images
empty_file_idx_sub = 0;
for fileNum = 1 : numfids
    
        disp(['Watershed on volume: ', int2str(fileNum), ' of total: ', int2str(numfids)]);
        cd(cur_dir);
        natfnames=natsort(trialNames);
        filename_raw = natfnames{fileNum};
        %% Decide if want to load individual channels or single image
        cd(foldername);
        [gray] = load_3D_gray(filename_raw);
      
        %figure(1); volshow(gray);
        
        %% TIGER - CHANGED "enhance", "Human_OL", and "cropping size" ==> all for Daryan's stuff
        DAPIsize = 10;
        DAPImetric = 0.2;
        enhance_DAPI = 'N';
        DAPI_bb_size = 10;
        binary = 'Y';
        [labels] = DAPIcount_3D(gray, DAPIsize, DAPImetric, enhance_DAPI, DAPI_bb_size, binary);  % function
               
        %% save label image
        labels = uint32(labels);
        z_size = length(labels(1, 1, :));
        for k = 1:z_size
            t = Tiff(strcat(filename_raw,'_watershed_seg.tif'), 'a');
            tagstruct.ImageLength = size(labels, 1);
            tagstruct.ImageWidth = size(labels, 2);
            tagstruct.Compression = Tiff.Compression.None;
            %tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
            tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
            tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
            tagstruct.BitsPerSample = 32;
            tagstruct.SamplesPerPixel = 1;
            tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            t.setTag(tagstruct);
            t.write(labels(:, :, k));
            t.close();
        end
        
        


        
        
        
        
        
end

end


