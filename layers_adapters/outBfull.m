function outBfull()
    global config mem;
    current_layer = config.misc.current_layer;
    mem.grads{current_layer+1} = mem.deltas{current_layer+1} * mem.activations{current_layer}';
    mem.grads{current_layer+1+config.layer_num} = sum(mem.deltas{current_layer+1}, 2);
end

