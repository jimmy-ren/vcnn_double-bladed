%--------------------------------------------------------------------------------------------------------
% The system is created based on the principles described in the following paper
% Jimmy SJ. Ren, Li Xu, Qiong Yan, Wenxiu Sun, "Shepard Convolutional Neural Networks", 
% Advances in Neural Information Processing Systems (NIPS 2015)
% email: jimmy.sj.ren@gmail.com
%--------------------------------------------------------------------------------------------------------
addpath applications/Shepard_CNN/
addpath applications/Shepard_CNN/utility/
addpath applications/Shepard_CNN/Shepard_inpainting/imgs/
addpath applications/Shepard_CNN/Shepard_inpainting/results/
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

% sample 1
% I = im2double(imread('color_inpainting_1.png'));
% mask = im2double(imread('color_inpainting_mask_1.png')) + 0.0001;
% sample 2
I = im2double(imread('color_inpainting_2.png'));
mask = im2double(imread('color_inpainting_mask_2.png')) + 0.0001;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mask = mask(:,:,1);
I = padarray(I, [4 4], 'replicate');
mask = padarray(mask, [4 4], 'replicate');

global config;
prepare_net(size(I, 1), size(I, 2), 'applications/Shepard_CNN/Shepard_inpainting/results/sh_inpaint.mat');

tic
final_output_r = apply_net_with_mask(config.NEW_MEM(I(:,:,1)).*mask, mask);
final_output_g = apply_net_with_mask(config.NEW_MEM(I(:,:,2)).*mask, mask);
final_output_b = apply_net_with_mask(config.NEW_MEM(I(:,:,3)).*mask, mask);
toc
final_output_r = gather(final_output_r);
final_output_g = gather(final_output_g);
final_output_b = gather(final_output_b);
final_output = cat(3, final_output_r, final_output_g, final_output_b);

I = I(5:size(I,1)-4,5:size(I,2)-4,:);
final_output = final_output(5:size(final_output,1)-4,5:size(final_output,2)-4,:);
figure, imshow([I final_output]);
drawnow();




