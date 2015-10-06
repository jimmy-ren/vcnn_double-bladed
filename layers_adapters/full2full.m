function full2full()
    global config mem;
    curr_layer_idx = config.misc.current_layer;
    mem.layer_inputs{curr_layer_idx} = mem.activations{curr_layer_idx-1};
end
