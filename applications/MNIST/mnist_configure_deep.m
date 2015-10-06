%--------------------------------------------------------------------------------------------------------
% The system is created based on the principles described in the following paper
% Jimmy SJ. Ren and Li Xu, "On Vectorization of Deep Convolutional Neural Networks for Vision Tasks", 
% The 29th AAAI Conference on Artificial Intelligence (AAAI-15). Austin, Texas, USA, January 25-30, 2015
% email: jimmy.sj.ren@gmail.com
%--------------------------------------------------------------------------------------------------------
function mnist_configure_deep()    
    global config;
    % general configurations for both training and testing
    config.input_size = [32 32];
    config.chs = 1;
    config.forward_pass_scheme = {'conv_v', 'conv_v', 'pool', 'conv_v', 'conv_v', 'pool', 'conv_v', 'full', 'full', 'full', 'out'};
    config.nonlinearity = 'relu';   % 'relu', 'tanh', 'sigmoid'
    config.output_activation = 'softmax';   % 'softmax', 'inherit'
    config.cost_function = 'cross entropy'; % 'cross entropy', 'L2 norm'
    config.kernel_size = [3 3; 3 3; 3 3; 3 3; 5 5];
    config.conv_hidden_size = [6 6 20 20 150];
    config.full_hidden_size = [150 150];
    config.output_size = [1 1 10];
    config.batch_size = 100;
    config.compute_device = 'GPU';
    
    % the following items are only for training
    config.learning_rate = 0.01;
    config.weight_range = 2;
    config.decay = 5e-7 / 10;
    config.normalize_init_weights = 1;
    config.dropout_full_layer = 1;
    config.optimization = 'adagrad';
end
