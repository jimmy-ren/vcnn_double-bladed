addpath applications/
addpath utils/
addpath cuda/
addpath mem/
addpath layers/
addpath layers_adapters/
addpath pipeline/

clearvars -global config;
clearvars -global mem;
clear gen_mask_patch_cat_idx_for_super_res;

global config;
%shepard_sr_x2_configure();
%shepard_sr_x3_configure();
shepard_sr_x4_configure();
init(0);

% for gradient checking
img = im2double(imread('applications/image_denoise/sample1.png'));
img = rgb2ycbcr(img);
img = img(:,:,1);
startpt = 150;
image = config.NEW_MEM((img(startpt:startpt+config.input_size(1)-1, startpt:startpt+config.input_size(2)-1, :)));
mask = config.NEW_MEM([1 0 0 0;0 0 0 0;0 0 0 0;0 0 0 0]);
mask = repmat(mask, config.input_size(1)/4, config.input_size(2)/4, config.chs);
%mask = config.NEW_MEM([1 0 0; 0 0 0; 0 0 0]);
%mask = repmat(mask, config.input_size(1)/3, config.input_size(2)/3, config.chs);
%mask = config.NEW_MEM([1 0; 0 0]);
%mask = repmat(mask, config.input_size(1)/2, config.input_size(2)/2, config.chs);
label = config.NEW_MEM((image(1:config.output_size(1), 1:config.output_size(2), :)));

image = repmat(image, 1,1,1,config.batch_size);
mask = repmat(mask, 1,1,1,config.batch_size);
label = repmat(label, 1,1,1,config.batch_size);


% gradient checking
mask_li = {};
mask_li{1} = mask;
op_train_pipe_with_mask(image, mask_li, label);
computeNumericalGradient_with_mask(image, mask_li, label, 1);


%{
% speed test
loop = 100;
tic
for m = 1:loop
    fprintf('%d\n', m);
    op_train_pipe_with_mask(image, mask_li, label);
end
elapse = toc/loop/config.batch_size;
num = 1 / elapse;
fprintf('Process %f samples per second.\n', num);
%}



