function pool_mem(layer_idx)
    global config mem;
    
    sub = 1:config.feature_map_sizes{layer_idx}(1)*config.feature_map_sizes{layer_idx}(2)*config.feature_map_sizes{layer_idx}(3)*config.batch_size;
    sub = reshape(sub, config.feature_map_sizes{layer_idx}(1), config.feature_map_sizes{layer_idx}(2)*config.batch_size, config.feature_map_sizes{layer_idx}(3));
    pre_sub = zeros(size(sub, 1)*2, size(sub, 2)*2, size(sub, 3));
    for m = 1:size(sub, 3)
        pre_sub(:,:,m) = kron(sub(:,:,m), ones(2));
    end
    mem.pooling_matrix{layer_idx} = reshape(pre_sub, size(pre_sub, 1)*size(pre_sub, 2), config.feature_map_sizes{layer_idx}(3))';
    mem.pooling_matrix{layer_idx} = config.NEW_MEM(mem.pooling_matrix{layer_idx}(:));
    
    %fprintf('%d\n', layer_idx);
    if strfind(config.forward_pass_scheme{layer_idx-1}, 'conv')
        mem.layer_inputs{layer_idx} = 0;%config.NEW_MEM(zeros(size(mem.activations{layer_idx-1}, 2)/4, size(mem.activations{layer_idx-1}, 1)));
        mem.activations{layer_idx} = 0;%config.NEW_MEM(zeros(size(mem.layer_inputs{layer_idx})));
    else
        print_log('in pool_mem(): pooling after pooling not supported yet');
    end
end


