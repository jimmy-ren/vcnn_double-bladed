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
mnist_configure();
init(0);

%verify_mode = 'gradient';
verify_mode = 'speed';

if strcmp(verify_mode, 'gradient')
    % for gradient checking
    image = config.NEW_MEM(rand(32, 32, 1, config.batch_size)) / (32*32);
    label = config.NEW_MEM((ones(10, config.batch_size) * 2 - 1)) ./ 10;
    op_train_pipe(image, label);
    computeNumericalGradient(image, label, 1);
elseif strcmp(verify_mode, 'speed')
    % for speed testing
    I = gpuArray(single(rand(32,32,1,config.batch_size)));
    y = gpuArray(single(rand(10, config.batch_size)));
    loop = 100;
    tic
    for m = 1:loop
        op_train_pipe(I, y);
    end
    elapse = toc/loop/config.batch_size;
    num = 1 / elapse;
    fprintf('Process %f samples per second.\n', num);
end


