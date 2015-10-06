function nonlinearity()
    global config mem;
    mem.activations{config.misc.current_layer-1} = config.NONLINEARITY(mem.activations{config.misc.current_layer-1});
end

