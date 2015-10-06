function full_forward()
    global config mem;
    curr_layer_idx = config.misc.current_layer;
    mem.activations{curr_layer_idx} = bsxfun(@plus, config.weights{curr_layer_idx} * mem.layer_inputs{curr_layer_idx}, config.weights{config.layer_num+curr_layer_idx});
    config.misc.current_layer = curr_layer_idx + 1;
end

