function pool2conv()
    global config mem;
    curr_layer_idx = config.misc.current_layer;
    conv_layer_idx = get_conv_layer_idx_from_layer_idx(curr_layer_idx);
    P_reshape = reshape(mem.activations{curr_layer_idx-1}, config.feature_map_sizes{curr_layer_idx-1}(1), config.feature_map_sizes{curr_layer_idx-1}(2)*config.feature_map_sizes{curr_layer_idx-1}(3)*config.batch_size);
    P_expand = config.IM2COL(P_reshape, [config.kernel_size(conv_layer_idx, 1), config.kernel_size(conv_layer_idx, 2)]);
    P_expand(:, mem.pool2conv{curr_layer_idx}{2}) = 0;
    input_size = [0 0];
    input_size(1) = config.kernel_size(conv_layer_idx, 1)*config.kernel_size(conv_layer_idx, 2)*config.feature_map_sizes{curr_layer_idx-1}(3);
    input_size(2) = (config.feature_map_sizes{curr_layer_idx-1}(1)-config.kernel_size(conv_layer_idx, 1)+1)*(config.feature_map_sizes{curr_layer_idx-1}(2)-config.kernel_size(conv_layer_idx, 2)+1)*config.batch_size;
    mem.layer_inputs{curr_layer_idx} = reshape(accumarray(mem.pool2conv{curr_layer_idx}{1}, P_expand(:), [input_size(1)*input_size(2), 1]), input_size(1), input_size(2));
    % memory redundancy, need improve
    mem.orig_activation_size{curr_layer_idx-1} = size(mem.activations{curr_layer_idx-1});
    mem.activations{curr_layer_idx-1} = mem.layer_inputs{curr_layer_idx};
end


