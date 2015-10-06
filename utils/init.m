function init(flag)
    % flag: 0 for training, 1 for testing
    global config;
    
    config.GEN_OUTPUT = @gen_output_copy;
    if strcmp(config.compute_device, 'GPU')
        init_gpu(1);
        config.NEW_MEM = @to_gpu;
        config.IM2COL = @im2col_gpu;
    else
        config.NEW_MEM = @to_cpu;
        config.IM2COL = @im2col;
    end
    
    if strcmp(config.nonlinearity, 'relu')
        config.NONLINEARITY = @relu;
    elseif strcmp(config.nonlinearity, 'tanh')
        config.NONLINEARITY = @tanh;
    elseif strcmp(config.nonlinearity, 'sigmoid')
        config.NONLINEARITY = @sigmoid;
    else
        config.NONLINEARITY = @tanh;
        fprintf('nonlinearity spec error, use tanh by default\n');
    end
    
    if strcmp(config.output_activation, 'softmax')
        config.OUT_ACT = @softmax;
    elseif strcmp(config.output_activation, 'inherit')
        config.OUT_ACT = config.NONLINEARITY;
    elseif strcmp(config.output_activation, 'nil')
        config.OUT_ACT = @nonlinearity_nil;
    else
        config.OUT_ACT = @softmax;
        fprintf('output_activation spec error, use softmax by default\n');
    end
    
    if strcmp(config.cost_function, 'cross entropy')
        config.COST_FUN = @cross_entropy;
    elseif strcmp(config.cost_function, 'L2 norm')
        config.COST_FUN = @L2_norm;
    else
        config.COST_FUN = @cross_entropy;
        fprintf('cost_function spec error, use cross_entropy by default\n');
    end
    
    config.cost = 0;
    config.misc.current_layer = 1;
    
    % initialize weights and calculate some statistics
    r = config.weight_range;    
    conv_layer_c = 0;
    pool_layer_c = 0;
    full_layer_c = 0;
    layer_num = length(config.forward_pass_scheme)-1;
    config.layer_num = layer_num;
    config.feature_map_sizes = {};
    config.weights = {};
    for idx = 1:layer_num
        if idx == 1
            conv_layer_c = conv_layer_c + 1;
            config.feature_map_sizes{idx} = [config.input_size(1)-config.kernel_size(1,1)+1 config.input_size(2)-config.kernel_size(1,2)+1 ...
                                             config.conv_hidden_size(conv_layer_c)];
            config.misc.mask_type = 16;     % hard code here for now
            %config.misc.mask_type = 4;
            if strcmp(config.forward_pass_scheme{idx}, 'conv_v_sr')
                config.weights{idx} = {};
                for t = 1:config.misc.mask_type
                    config.weights{idx}{t} = config.NEW_MEM(randn(config.feature_map_sizes{idx}(3), ...
                                              config.kernel_size(conv_layer_c, 1)*config.kernel_size(conv_layer_c, 2)*config.chs)*r);
                end
                % create mask and generate conv index
                mask_mem();
                %mask = config.NEW_MEM([1 0;0 0]);
                mask = config.NEW_MEM([1 0 0 0;0 0 0 0;0 0 0 0;0 0 0 0]);
                mask = repmat(mask, config.input_size(1)/sqrt(config.misc.mask_type), config.input_size(2)/sqrt(config.misc.mask_type), config.chs);
                mask = repmat(mask, 1,1,1,config.batch_size);
                mask2conv(mask);
            elseif strcmp(config.forward_pass_scheme{idx}, 'conv_v')
                config.weights{idx} = config.NEW_MEM(randn(config.feature_map_sizes{idx}(3), ...
                                              config.kernel_size(conv_layer_c, 1)*config.kernel_size(conv_layer_c, 2)*config.chs)*r);
                if config.normalize_init_weights
                    config.weights{idx} = config.weights{idx} / sqrt(config.kernel_size(conv_layer_c, 1) * config.kernel_size(conv_layer_c, 2) * config.conv_hidden_size(conv_layer_c));
                end
            elseif strcmp(config.forward_pass_scheme{idx}, 'conv_v_mask_norm')
                config.weights{idx} = config.NEW_MEM(randn(config.feature_map_sizes{idx}(3), ...
                                              config.kernel_size(conv_layer_c, 1)*config.kernel_size(conv_layer_c, 2)*config.chs)*r) + r;
                if config.normalize_init_weights
                    config.weights{idx} = config.weights{idx} / sqrt(config.kernel_size(conv_layer_c, 1) * config.kernel_size(conv_layer_c, 2) * config.conv_hidden_size(conv_layer_c));
                end
            end
        elseif strcmp(config.forward_pass_scheme{idx}, 'conv_v')
            conv_layer_c = conv_layer_c + 1;
            config.feature_map_sizes{idx} = [config.feature_map_sizes{idx-1}(1)-config.kernel_size(conv_layer_c,1)+1 ...
                                             config.feature_map_sizes{idx-1}(2)-config.kernel_size(conv_layer_c,2)+1 ...
                                             config.conv_hidden_size(conv_layer_c)];
            config.weights{idx} = config.NEW_MEM(randn(config.feature_map_sizes{idx}(3), ...
                                          config.kernel_size(conv_layer_c, 1)*config.kernel_size(conv_layer_c, 2)*config.feature_map_sizes{idx-1}(3))*r);
            if config.normalize_init_weights
                config.weights{idx} = config.weights{idx} / sqrt(config.kernel_size(conv_layer_c, 1) * config.kernel_size(conv_layer_c, 2) * config.conv_hidden_size(conv_layer_c));
            end
        elseif strcmp(config.forward_pass_scheme{idx}, 'conv_f')
            conv_layer_c = conv_layer_c + 1;
            if idx == layer_num
                config.weights{idx} = config.NEW_MEM(randn(config.kernel_size(conv_layer_c, 1)*config.kernel_size(conv_layer_c, 2)*config.output_size(3), config.conv_hidden_size(conv_layer_c-1))*r);
                if config.normalize_init_weights
                    config.weights{idx} = config.weights{idx} / sqrt(config.kernel_size(conv_layer_c, 1) * config.kernel_size(conv_layer_c, 2) * size(config.weights{idx}, 1));
                end
                config.GEN_OUTPUT = @gen_output_from_conv_f;
            else
                fprintf('in init(): conv_f layer in the hidden layer not supported yet.\n');
            end
        elseif strcmp(config.forward_pass_scheme{idx}, 'pool')
            pool_layer_c = pool_layer_c + 1;
            config.feature_map_sizes{idx} = [config.feature_map_sizes{idx-1}(1)/2 config.feature_map_sizes{idx-1}(2)/2 ...
                                             config.feature_map_sizes{idx-1}(3)];            
            config.weights{idx} = config.NEW_MEM(randn(config.feature_map_sizes{idx-1}(3), 1) * r) / 4;            
        elseif strcmp(config.forward_pass_scheme{idx}, 'full')
            full_layer_c = full_layer_c + 1;            
            if idx == layer_num
                config.weights{idx} = config.NEW_MEM(randn(config.output_size(3), config.feature_map_sizes{idx-1}(3)) * r);
                if config.normalize_init_weights
                    config.weights{idx} = config.weights{idx} / sqrt(config.output_size(3));
                end
            else
                config.feature_map_sizes{idx} = [1 1 config.full_hidden_size(full_layer_c)];
                config.weights{idx} = config.NEW_MEM(randn(config.feature_map_sizes{idx}(3), ...
                    config.feature_map_sizes{idx-1}(1)*config.feature_map_sizes{idx-1}(2)*config.feature_map_sizes{idx-1}(3)) * r);
                if config.normalize_init_weights
                    config.weights{idx} = config.weights{idx} / sqrt(config.feature_map_sizes{idx}(3));
                end
            end            
        end
    end
    
    % initialize bias
    for idx = 1:layer_num-1
        config.weights{idx+layer_num} = config.NEW_MEM(zeros(config.feature_map_sizes{idx}(3), 1)+0.01);
    end
    if strcmp(config.forward_pass_scheme{layer_num}, 'conv_f')
        config.weights{layer_num*2} = config.NEW_MEM(zeros(size(config.weights{layer_num}, 1), 1)+0.05);
    else
        config.weights{layer_num*2} = config.NEW_MEM(zeros(config.output_size(3), 1)+0.05);
    end
    
    % prepare memory
    reset_mem();
    input_mem();
    if strcmp(config.forward_pass_scheme{1}, 'conv_v_mask_norm')
        mask_mem();
    end
    if strcmp(config.forward_pass_scheme{2}, 'conv_v')
        conv2conv_mem(1);
    end
    for m = 2:layer_num
        if strfind(config.forward_pass_scheme{m}, 'conv')
            conv_mem(m);
            if strcmp(config.forward_pass_scheme{m+1}, 'out')
                conv2out_mem();
            elseif strcmp(config.forward_pass_scheme{m+1}, 'conv_v')
                conv2conv_mem(m);
            end
        elseif strcmp(config.forward_pass_scheme{m}, 'pool')
            pool_mem(m);
            if strcmp(config.forward_pass_scheme{m+1}, 'conv_v')
                pool2conv_mem(m);
            end
        elseif strcmp(config.forward_pass_scheme{m}, 'full')
            full_mem(m);
        end
    end
    
    % building pipeline
    config.pipeline_forward = {};
    config.pipeline_forward{1} = @input2conv;
    if strcmp(config.forward_pass_scheme{1}, 'conv_v_mask_norm')
        config.pipeline_forward{2} = @mask2conv;
    end
    conv_layer_c = 1;
    for idx = 1:layer_num
        if strfind(config.forward_pass_scheme{idx}, 'conv')
            conv_layer_c = conv_layer_c + 1;
            if strcmp(config.forward_pass_scheme{idx}, 'conv_v_sr')
                config.pipeline_forward{length(config.pipeline_forward)+1} = @conv_forward_SR;
            else
                config.pipeline_forward{length(config.pipeline_forward)+1} = @conv_forward;
            end
            if strcmp(config.forward_pass_scheme{idx}, 'conv_v_mask_norm')
                config.pipeline_forward{length(config.pipeline_forward)+1} = @mask_conv_forward;
                config.pipeline_forward{length(config.pipeline_forward)+1} = @mask_normalize;
            end
            if strcmp(config.forward_pass_scheme{idx+1}, 'conv_v')
                config.pipeline_forward{length(config.pipeline_forward)+1} = @nonlinearity;
                if config.kernel_size(conv_layer_c, 1) == 1 && config.kernel_size(conv_layer_c, 2) == 1
                    config.pipeline_forward{length(config.pipeline_forward)+1} = @conv2conv1by1;
                else
                    config.pipeline_forward{length(config.pipeline_forward)+1} = @conv2conv;
                end
            elseif strcmp(config.forward_pass_scheme{idx+1}, 'conv_f')
                config.pipeline_forward{length(config.pipeline_forward)+1} = @nonlinearity;
                config.pipeline_forward{length(config.pipeline_forward)+1} = @conv2conv_f;
            elseif strcmp(config.forward_pass_scheme{idx+1}, 'pool')
                config.pipeline_forward{length(config.pipeline_forward)+1} = @nonlinearity;
                config.pipeline_forward{length(config.pipeline_forward)+1} = @conv2pool;
            elseif strcmp(config.forward_pass_scheme{idx+1}, 'full')
                config.pipeline_forward{length(config.pipeline_forward)+1} = @nonlinearity;
                config.pipeline_forward{length(config.pipeline_forward)+1} = @conv2full;
            elseif strcmp(config.forward_pass_scheme{idx+1}, 'out')
                if strcmp(config.forward_pass_scheme{idx}, 'conv_f')
                    config.pipeline_forward{length(config.pipeline_forward)+1} = @conv2out;
                    config.pipeline_forward{length(config.pipeline_forward)+1} = @out_forward;
                else
                    fprintf('in init(): currently only support conv_f as the output conv layer.\n');
                end
            end
        elseif strcmp(config.forward_pass_scheme{idx}, 'pool')
            config.pipeline_forward{length(config.pipeline_forward)+1} = @pool_forward;
            config.pipeline_forward{length(config.pipeline_forward)+1} = @nonlinearity;
            if strcmp(config.forward_pass_scheme{idx+1}, 'conv_v')
                config.pipeline_forward{length(config.pipeline_forward)+1} = @pool2conv;
            elseif strcmp(config.forward_pass_scheme{idx+1}, 'pool')
                config.pipeline_forward{length(config.pipeline_forward)+1} = @pool2pool;
            elseif strcmp(config.forward_pass_scheme{idx+1}, 'full')
                config.pipeline_forward{length(config.pipeline_forward)+1} = @pool2full;
            end
        elseif strcmp(config.forward_pass_scheme{idx}, 'full')
            config.pipeline_forward{length(config.pipeline_forward)+1} = @full_forward;
            if strcmp(config.forward_pass_scheme{idx+1}, 'full')
                config.pipeline_forward{length(config.pipeline_forward)+1} = @nonlinearity;
                if config.dropout_full_layer == 1
                    config.pipeline_forward{length(config.pipeline_forward)+1} = @dropout_forward;
                end
                config.pipeline_forward{length(config.pipeline_forward)+1} = @full2full;
            elseif strcmp(config.forward_pass_scheme{idx+1}, 'out')
                config.pipeline_forward{length(config.pipeline_forward)+1} = @full2out;
                config.pipeline_forward{length(config.pipeline_forward)+1} = @out_forward;
            end
        end
    end
    
    config.SCALE_INPUT = @scale_input_nil;
    config.SCALE_OUTPUT = @scale_output_nil;
    
    if flag ~= 0
        return;
    end
    config.EXPAND_DELTA_OUT = @expand_delta_out_nil;
    if strcmp(config.nonlinearity, 'relu')
        config.DERI_NONLINEARITY = @deri_relu;
    elseif strcmp(config.nonlinearity, 'tanh')
        config.DERI_NONLINEARITY = @deri_tanh;
    elseif strcmp(config.nonlinearity, 'sigmoid')
        config.DERI_NONLINEARITY = @deri_sigmoid;
    else
        config.DERI_NONLINEARITY = @deri_tanh;        
    end
    
    if strcmp(config.output_activation, 'softmax')
        config.DERI_OUT_ACT = @deri_softmax;
    elseif strcmp(config.output_activation, 'inherit')
        config.DERI_OUT_ACT = @deri_inherit;
    elseif strcmp(config.output_activation, 'nil')
        config.DERI_OUT_ACT = @deri_nonlinearity_nil;
    else
        config.DERI_OUT_ACT = @deri_softmax;        
    end
    
    if strcmp(config.cost_function, 'cross entropy')
        config.DERI_COST_FUN = @deri_cross_entropy;
    elseif strcmp(config.cost_function, 'L2 norm')
        config.DERI_COST_FUN = @deri_L2_norm;
    else
        config.DERI_COST_FUN = @deri_cross_entropy;        
    end
    
    for m = 2:layer_num        
        if strcmp(config.forward_pass_scheme{m}, 'conv_v')            
            if strcmp(config.forward_pass_scheme{m-1}, 'pool')
                convBpool_mem(m);
            elseif strfind(config.forward_pass_scheme{m}, 'conv')
                conv_layer_id = get_conv_layer_idx_from_layer_idx(m);
                if config.kernel_size(conv_layer_id, 1) ~= 1 && config.kernel_size(conv_layer_id, 2) ~= 1
                    convBconv_mem(m);
                end
            end        
        end
    end
    
    % building pipeline for backprop
    config.pipeline_backprop = {};
    config.pipeline_backprop{1} = @out_backprop;
    for idx = layer_num+1:-1:3
        if strcmp(config.forward_pass_scheme{idx}, 'out')
            if strcmp(config.forward_pass_scheme{idx-1}, 'conv_f')
                config.EXPAND_DELTA_OUT = @expand_delta_out_for_conv_f;
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @outBconv;
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @conv_backprop;
            elseif strcmp(config.forward_pass_scheme{idx-1}, 'full')
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @outBfull;
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @full_backprop;
            else
                fprintf('in init(): backprop from the output layer to the specified layer is not yet supported.\n');
            end            
        elseif strcmp(config.forward_pass_scheme{idx}, 'conv_f')
            if strcmp(config.forward_pass_scheme{idx-1}, 'conv_v')                
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @convBconv_1by1;                
            else
                fprintf('in init(): backprop from conv_f to the specified layer is not yet supported.\n');
            end
            config.pipeline_backprop{length(config.pipeline_backprop)+1} = @conv_backprop;
        elseif strcmp(config.forward_pass_scheme{idx}, 'conv_v')
            if strfind(config.forward_pass_scheme{idx-1}, 'conv')
                conv_layer_id = get_conv_layer_idx_from_layer_idx(idx);
                if config.kernel_size(conv_layer_id, 1) == 1 && config.kernel_size(conv_layer_id, 2) == 1
                    config.pipeline_backprop{length(config.pipeline_backprop)+1} = @convBconv_1by1;
                else
                    config.pipeline_backprop{length(config.pipeline_backprop)+1} = @convBconv;
                end
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @conv_backprop;
            elseif strcmp(config.forward_pass_scheme{idx-1}, 'pool')
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @convBpool;
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @pool_backprop;
            end            
        elseif strcmp(config.forward_pass_scheme{idx}, 'pool')
            if strcmp(config.forward_pass_scheme{idx-1}, 'conv_v')
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @poolBconv;
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @conv_backprop;
            elseif strcmp(config.forward_pass_scheme{idx-1}, 'pool')
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @poolBpool;
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @pool_backprop;
            end            
        elseif strcmp(config.forward_pass_scheme{idx}, 'full')
            if strcmp(config.forward_pass_scheme{idx-1}, 'full')
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @fullBfull;
            elseif strcmp(config.forward_pass_scheme{idx-1}, 'conv_v')
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @fullBconv;
            elseif strcmp(config.forward_pass_scheme{idx-1}, 'pool')
                config.pipeline_backprop{length(config.pipeline_backprop)+1} = @fullBpool;
            end
            config.pipeline_backprop{length(config.pipeline_backprop)+1} = @full_backprop;
        end                
    end
    if strcmp(config.forward_pass_scheme{2}, 'conv_v') && config.kernel_size(2, 1) ~= 1 && config.kernel_size(2, 2) ~= 1
        config.pipeline_backprop{length(config.pipeline_backprop)+1} = @convBconv_last;
    end
    if strcmp(config.forward_pass_scheme{1}, 'conv_v_mask_norm')
        config.pipeline_backprop{length(config.pipeline_backprop)+1} = @convBinput_with_mask_accel;
        %config.pipeline_backprop{length(config.pipeline_backprop)+1} = @convBinput_with_mask;
    elseif strcmp(config.forward_pass_scheme{1}, 'conv_v_sr')
        config.pipeline_backprop{length(config.pipeline_backprop)+1} = @convBinput_SR;
    else
        config.pipeline_backprop{length(config.pipeline_backprop)+1} = @convBinput;
    end    
    
    if strcmp(config.optimization, 'adagrad')        
        config.his_grad = {};
        config.fudge_factor = 1e-6;
        if strcmp(config.forward_pass_scheme{1}, 'conv_v_sr')
            config.UPDATE_WEIGHTS = @update_weights_adagrad_SR;
            config.his_grad{1} = {};
            for m = 1:config.misc.mask_type
                config.his_grad{1}{m} = config.NEW_MEM(zeros(size(config.weights{1}{m})));
            end
            for m = 2:length(config.weights)
                config.his_grad{m} = config.NEW_MEM(zeros(size(config.weights{m})));
            end
        else
            config.UPDATE_WEIGHTS = @update_weights_adagrad;
            for m = 1:length(config.weights)
                config.his_grad{m} = config.NEW_MEM(zeros(size(config.weights{m})));
            end
        end
    else
        fprintf('optimization method not supported, use adagrad as default\n');
        config.UPDATE_WEIGHTS = @update_weights_adagrad;
    end
end




