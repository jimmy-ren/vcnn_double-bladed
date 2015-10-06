function conv_backprop()
    global config mem;
    current_layer = config.misc.current_layer;
    mem.deltas{current_layer} = mem.delta_act .* config.DERI_NONLINEARITY(mem.activations{current_layer});
    config.misc.current_layer = config.misc.current_layer - 1;
end

