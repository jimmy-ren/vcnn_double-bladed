%--------------------------------------------------------------------------------------------------------
% The system is created based on the principles described in the following paper
% Jimmy SJ. Ren and Li Xu, "On Vectorization of Deep Convolutional Neural Networks for Vision Tasks", 
% The 29th AAAI Conference on Artificial Intelligence (AAAI-15). Austin, Texas, USA, January 25-30, 2015
% email: jimmy.sj.ren@gmail.com
%--------------------------------------------------------------------------------------------------------
addpath applications/image_denoise/
addpath applications/image_denoise/utility/
addpath utils/
addpath cuda/
addpath mem/
addpath layers/
addpath layers_adapters/
addpath pipeline/

clearvars -global config;
clearvars -global mem;
clear;
I = im2double(imread('sample2.png'));
global config;
prepare_net(size(I, 1), size(I, 2), 'w_gaussian.mat');
final_output = apply_net(config.NEW_MEM(I));
final_output = gather(final_output);
figure, imshow([I final_output]);
drawnow();

