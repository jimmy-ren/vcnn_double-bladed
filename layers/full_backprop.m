function full_backprop()
    global config mem;
    current_layer = config.misc.current_layer;
    mem.deltas{current_layer} = config.weights{current_layer+1}' * mem.deltas{current_layer+1} ...
                                .* config.DERI_NONLINEARITY(mem.activations{current_layer});
    config.misc.current_layer = config.misc.current_layer - 1;
end

