function mask_normalize()
    global config mem;
    curr_layer_idx = config.misc.current_layer - 1;
    mem.activations{curr_layer_idx} = mem.activations{curr_layer_idx} ./ mem.mask_activations{curr_layer_idx};
end


