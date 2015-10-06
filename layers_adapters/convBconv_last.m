function convBconv_last()
    global config mem;
    curr_layer_idx = config.misc.current_layer;
    mem.deltas{curr_layer_idx+1} = reshape(accumarray(mem.convBconv{curr_layer_idx+2}, mem.deltas{curr_layer_idx+1}(:)), ...
                                            mem.orig_activation_size{curr_layer_idx+1}(2), mem.orig_activation_size{curr_layer_idx+1}(1))';
                                           %size(mem.layer_inputs{curr_layer_idx+1}, 1), size(mem.layer_inputs{curr_layer_idx+1}, 2));    
end

