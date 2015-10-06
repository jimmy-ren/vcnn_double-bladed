function conv_forward_SR()
    global config mem;
    curr_layer_idx = config.misc.current_layer;
    mem.activations{1} = config.NEW_MEM(zeros(config.feature_map_sizes{1}(3), config.feature_map_sizes{1}(1)*config.feature_map_sizes{1}(2)));
    for m = 1:config.misc.mask_type
        mem.activations{curr_layer_idx}(:,mem.mask_idx{m}) = bsxfun(@plus, config.weights{curr_layer_idx}{m} * mem.layer_inputs{curr_layer_idx}(:,mem.mask_idx{m}), config.weights{config.layer_num+curr_layer_idx});
    end
    config.misc.current_layer = curr_layer_idx + 1;
end

