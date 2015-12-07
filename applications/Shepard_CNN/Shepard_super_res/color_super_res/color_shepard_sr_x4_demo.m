%--------------------------------------------------------------------------------------------------------
% The system is created based on the principles described in the following paper
% Jimmy SJ. Ren, Li Xu, Qiong Yan, Wenxiu Sun, "Shepard Convolutional Neural Networks", 
% Advances in Neural Information Processing Systems (NIPS 2015)
% email: jimmy.sj.ren@gmail.com
%--------------------------------------------------------------------------------------------------------
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
I = im2double(imread('applications/Shepard_CNN/Shepard_super_res/images/Set5/butterfly_GT.bmp'));
% set 14
% -- baboon barbara bridge coastguard comic face flowers foreman lenna man monarch pepper ppt3 zebra
%I = im2double(imread('applications/Shepard_CNN/Shepard_super_res/images/Set14/pepper.bmp'));
if(size(I,3) == 3)
    I = rgb2ycbcr(I);
end

I = modcrop(I, 4);
I = padarray(I, [4 4], 'replicate');

% downsample the image
D3chs = imresize(I, [size(I,1)/4 size(I,2)/4], 'bicubic');
D = D3chs(:,:,1);

% super-resolution x3
U = imresize(D, [size(I,1) size(I,2)], 'nearest');
prepare_sr_net(size(U, 1), size(U, 2), 'applications/Shepard_CNN/Shepard_super_res/results/shepard_layer_x4/sr_x4.mat');
mask = config.NEW_MEM([1 0 0 0; ...
                       0 0 0 0; ...
                       0 0 0 0; ...
                       0 0 0 0]);
mask = repmat(mask, size(U, 1)/4, size(U, 2)/4, 1);

final_output = apply_net_with_mask(config.NEW_MEM(U) .* mask, mask);
final_output = gather(final_output);
D3chs = imresize(D3chs, [size(final_output,1),size(final_output,2)]);
final_output = shave(final_output, [4 4]);
D3chs = shave(D3chs, [4 4]);

final_output = cat(3, final_output, D3chs(:,:,2:3));
final_output = ycbcr2rgb(double(final_output));
I_bicubic = ycbcr2rgb(D3chs);

figure('name', 'bicubic'), imshow(I_bicubic);
figure('name', 'Sheperd CNN'), imshow(final_output);
drawnow();





