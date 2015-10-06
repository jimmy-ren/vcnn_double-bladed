function conv2pool()
    global config mem;
    curr_layer_idx = config.misc.current_layer;
    mem.layer_inputs{curr_layer_idx} = reshape(accumarray(mem.pooling_matrix{curr_layer_idx}, mem.activations{curr_layer_idx-1}(:), ...
                        [size(mem.activations{curr_layer_idx-1},1)*size(mem.activations{curr_layer_idx-1},2)/4, 1]), size(mem.activations{curr_layer_idx-1}, 2)/4, size(mem.activations{curr_layer_idx-1}, 1));
end

