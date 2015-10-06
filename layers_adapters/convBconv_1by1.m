function convBconv_1by1()
    global config mem;
    curr_layer_idx = config.misc.current_layer;
    mem.grads{curr_layer_idx+1} = mem.deltas{curr_layer_idx+1} * mem.activations{curr_layer_idx}';
    mem.grads{curr_layer_idx+1+config.layer_num} = sum(mem.deltas{curr_layer_idx+1}, 2);
    mem.delta_act = config.weights{curr_layer_idx+1}' * mem.deltas{curr_layer_idx+1};
end

