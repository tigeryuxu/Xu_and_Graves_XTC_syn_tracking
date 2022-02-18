function [red_3D] = load_3D_gray(filename_raw, natfnames)
%iptsetpref('VolumeViewerUseHardware',false);   % HAVE TO USE THIS b/c problem with openGL currently
%iptsetpref('VolumeViewerUseHardware',true)

info = imfinfo(filename_raw);
num_images = numel(info);
im_size = [info(1).Height, info(1).Width];
gray_scale_size = im_size(1:2);
%green_3D = zeros([gray_scale_size, num_images]);
red_3D = zeros([gray_scale_size, num_images]);
%blue_3D = zeros([gray_scale_size, num_images]);
for k = 1:num_images
    A = imread(filename_raw, k, 'Info', info);
    %A = A;
    % ... Do something with image A ...
    %figure(888); imshow(A);
    red = A(:, :, 1);
    %green = A(:, :, 2);
    
    red_3D(:, :, k) = im2double(red);
    %green_3D(:, :, k) = im2double(green);
    %blue_3D(:, :, k) = im2double(green);
end

%volumeViewer(red_3D);
%figure(400); volshow(red_3D,  'BackgroundColor', [0,0,0]);
%figure(401); volshow(green_3D,  'BackgroundColor', [0,0,0]);






