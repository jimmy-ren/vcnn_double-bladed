addpath applications/Shepard_CNN/Shepard_super_res/
addpath applications/Shepard_CNN/utility/
addpath applications/image_denoise/utility/
addpath utils/
addpath cuda/
addpath mem/
addpath layers/
addpath layers_adapters/
addpath pipeline/

clearvars -global config;
clearvars -global mem;
clear gen_mask_patch_cat_idx_for_super_res;
clear;
global config;

% set5
% -- baby bird butterfly head woman
%I = im2double(imread('applications/Shepard_CNN/Shepard_super_res/images/Set5/bird_GT.bmp'));
% set 14
% -- baboon barbara bridge coastguard comic face flowers foreman lenna man monarch pepper ppt3 zebra
I = im2double(imread('applications/Shepard_CNN/Shepard_super_res/images/Set14/baboon.bmp'));
if(size(I,3) == 3)
    I = rgb2ycbcr(I);
end

I = modcrop(I, 4);
I = padarray(I, [4 4], 'replicate');

% downsample the image
D = imresize(I(:,:,1), [size(I,1)/2 size(I,2)/2], 'bicubic');

% super-resolution x2
U = imresize(D, [size(I,1) size(I,2)], 'nearest');
prepare_sr_net(size(U, 1), size(U, 2), 'applications/Shepard_CNN/Shepard_super_res/results/shepard_layer_x2/sr_x2.mat');
mask = config.NEW_MEM([1 0; ...
                       0 0]);
mask = repmat(mask, size(U, 1)/2, size(U, 2)/2, 1);

final_output = apply_net_with_mask(config.NEW_MEM(U) .* mask, mask);
final_output = gather(final_output);
final_output = shave(final_output, [2 2]);
GT = shave(I(:,:,1), [2 2]);
input = shave(U.*mask, [2 2]);

GT = GT(5:size(GT,1)-4,5:size(GT,2)-4);
final_output = final_output(5:size(final_output,1)-4,5:size(final_output,2)-4);
input = input(5:size(input,1)-4,5:size(input,2)-4);
figure, imshow([GT input final_output]);
drawnow();

% calculate psnr according to previous studies
[psnr, mse] = cal_psnr(uint8(GT*255), uint8(final_output*255));
fprintf('psnr: %f, mse: %f\n', psnr, mse);



