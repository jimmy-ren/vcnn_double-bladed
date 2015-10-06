%--------------------------------------------------------------------------------------------------------
% The system is created based on the principles described in the following paper
% Jimmy SJ. Ren and Li Xu, "On Vectorization of Deep Convolutional Neural Networks for Vision Tasks", 
% The 29th AAAI Conference on Artificial Intelligence (AAAI-15). Austin, Texas, USA, January 25-30, 2015
% email: jimmy.sj.ren@gmail.com
%--------------------------------------------------------------------------------------------------------
addpath applications/
addpath utils/
addpath cuda/
addpath mem/
addpath layers/
addpath layers_adapters/
addpath pipeline/
addpath data/MNIST/

clearvars -global config;
clearvars -global mem;

global config;
denoise_configure();
init(0);

% for gradient checking
img = im2double(imread('sample1.png'));
startpt = 150;
image = config.NEW_MEM((img(startpt:startpt+config.input_size(1)-1, startpt:startpt+config.input_size(2)-1, :)));
label = config.NEW_MEM((image(1:config.output_size(1), 1:config.output_size(2), :)));

image = repmat(image, 1,1,1,config.batch_size);
label = repmat(label, 1,1,1,config.batch_size);

% gradient checking
op_train_pipe(image, label);
computeNumericalGradient(image, label, 1);

% speed test
loop = 100;
tic
for m = 1:loop
    op_train_pipe(image, label);
end
elapse = toc/loop/config.batch_size;
num = 1 / elapse;
fprintf('Process %f samples per second.\n', num);


