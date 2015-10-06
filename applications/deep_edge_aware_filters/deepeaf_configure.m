%--------------------------------------------------------------------------------------------------------
% The system is created based on the principles described in the following papers
% [1] Li Xu, Jimmy SJ. Ren, Qiong Yan, Renjie Liao, Jiaya Jia, "Deep Edge-Aware Filters", 
% The 32nd International Conference on Machine Learning (ICML 2015). Lille, France, July 6-11, 2015
% [2] Jimmy SJ. Ren and Li Xu, "On Vectorization of Deep Convolutional Neural Networks for Vision Tasks", 
% The 29th AAAI Conference on Artificial Intelligence (AAAI-15). Austin, Texas, USA, January 25-30, 2015
%--------------------------------------------------------------------------------------------------------
function deepeaf_configure()
    global config;
    % general configurations for both training and testing
    config.input_size = [64 64];
    config.chs = 3;
    config.forward_pass_scheme = {'conv_v', 'conv_v', 'conv_f', 'out'};
    config.nonlinearity = 'relu';   % 'relu', 'tanh', 'sigmoid'
    config.output_activation = 'nil';   % 'softmax', 'inherit', 'nil'
    config.cost_function = 'L2 norm'; % 'cross entropy', 'L2 norm'
    config.kernel_size = [16 16; 1 1; 8 8];
    config.conv_hidden_size = [512 512];
    config.full_hidden_size = [];
    config.output_size = [56 56 3];
    config.batch_size = 10;
    config.compute_device = 'GPU';
    
    % the following items are only for training
    config.learning_rate = 0.02;
    config.weight_range = 0.01;
    config.decay = 5e-7 / 10;
    config.normalize_init_weights = 0;
    config.dropout_full_layer = 0;
    config.optimization = 'adagrad';
end
