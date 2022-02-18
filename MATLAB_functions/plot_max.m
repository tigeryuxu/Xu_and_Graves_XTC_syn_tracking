function [mip] = plot_max(vol)

    axis = 3;
    mip = max(vol, [], axis);
    figure(); imshow(mip);

end