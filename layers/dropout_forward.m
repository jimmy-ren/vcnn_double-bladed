function dropout_forward()
    global config mem;
    curr_layer_idx = config.misc.current_layer - 1;
    if config.misc.training;
        dm = config.NEW_MEM(single(uint8(rand(size(mem.activations{curr_layer_idx})))));
        mem.activations{curr_layer_idx} = mem.activations{curr_layer_idx} .* dm;
    else
        mem.activations{curr_layer_idx} = mem.activations{curr_layer_idx} ./ 2;
    end
end

