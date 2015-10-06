function convBinput()
    global config mem;
    curr_layer_idx = config.misc.current_layer;
    mem.grads{curr_layer_idx+1} = mem.deltas{curr_layer_idx+1} * mem.layer_inputs{curr_layer_idx+1}';
    mem.grads{curr_layer_idx+1+config.layer_num} = sum(mem.deltas{curr_layer_idx+1}, 2);
end
