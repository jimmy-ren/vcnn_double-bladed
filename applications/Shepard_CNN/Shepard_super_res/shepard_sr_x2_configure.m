function shepard_sr_x2_configure()    
    global config;
    % general configurations for both training and testing
    config.input_size = [48 48];
    config.chs = 1;
    config.forward_pass_scheme = {'conv_v_mask_norm', 'conv_v', 'conv_v', 'conv_f', 'out'};
    config.mask_for_SR = 'true';
    config.misc.mask_type = 4; % 4x super-resolution
    
    config.nonlinearity = 'relu';   % 'relu', 'tanh', 'sigmoid'
    config.output_activation = 'nil';   % 'softmax', 'inherit', 'nil'
    config.cost_function = 'L2 norm'; % 'cross entropy', 'L2 norm'
    config.kernel_size = [8 8; 9 9; 1 1; 8 8];
    config.conv_hidden_size = [16 512 512];
    config.full_hidden_size = [];
    config.output_size = [40 40 1];
    config.batch_size = 10;
    config.compute_device = 'GPU';
    
    % the following items are only for training
    config.learning_rate = 0.001;
    config.weight_range = 0.03;
    config.decay = 5e-7 / 10;
    config.normalize_init_weights = 0;
    config.dropout_full_layer = 0;
    config.optimization = 'adagrad';
end


