function mask_conv_forward()
    global config mem;
    curr_layer_idx = config.misc.current_layer - 1;
    %mem.mask_activations{curr_layer_idx} = bsxfun(@plus, config.weights{curr_layer_idx} * mem.mask_inputs{curr_layer_idx}, config.weights{config.layer_num+curr_layer_idx});
    mem.mask_activations{curr_layer_idx} = config.weights{curr_layer_idx} * mem.mask_inputs{curr_layer_idx};
end
