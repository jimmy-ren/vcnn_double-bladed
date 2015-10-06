function convBinput_SR()
    global config mem;
    curr_layer_idx = config.misc.current_layer;
    for m = 1:config.misc.mask_type
        mem.grads{curr_layer_idx+1}{m} = mem.deltas{curr_layer_idx+1}(:,mem.mask_idx{m}) * mem.layer_inputs{curr_layer_idx+1}(:,mem.mask_idx{m})';
    end
    mem.grads{curr_layer_idx+1+config.layer_num} = sum(mem.deltas{curr_layer_idx+1}, 2);
end
