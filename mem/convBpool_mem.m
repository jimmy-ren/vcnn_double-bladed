function convBpool_mem(layer_idx)
    global config mem;
    conv_layer_idx = get_conv_layer_idx_from_layer_idx(layer_idx);
    tt = reshape(1:config.feature_map_sizes{layer_idx-1}(1)*config.feature_map_sizes{layer_idx-1}(2), ...
                 config.feature_map_sizes{layer_idx-1}(1), config.feature_map_sizes{layer_idx-1}(2));
    ii = im2col(tt, [config.kernel_size(conv_layer_idx, 1), config.kernel_size(conv_layer_idx, 2)]);
    mem.convBpool{layer_idx} = {};    
    accu_idx_full = repmat(ii, config.feature_map_sizes{layer_idx-1}(3), config.batch_size);
    for m = 1:config.feature_map_sizes{layer_idx-1}(3)
        for n = 1:config.batch_size
            accu_idx_full((m-1)*size(ii, 1)+1:m*size(ii, 1), (n-1)*size(ii, 2)+1:n*size(ii, 2)) = ...
                accu_idx_full((m-1)*size(ii, 1)+1:m*size(ii, 1), (n-1)*size(ii, 2)+1:n*size(ii, 2)) + ((config.batch_size*(m-1)+n-1)*(config.feature_map_sizes{layer_idx-1}(1)*config.feature_map_sizes{layer_idx-1}(2)));
        end
    end
    mem.convBpool{layer_idx}{1} = config.NEW_MEM(accu_idx_full(:));
    
    up = 1:config.feature_map_sizes{layer_idx-1}(1) * config.feature_map_sizes{layer_idx-1}(2) * config.feature_map_sizes{layer_idx-1}(3) * config.batch_size;
    up = reshape(up, config.feature_map_sizes{layer_idx-1}(1), config.feature_map_sizes{layer_idx-1}(2)*config.batch_size, config.feature_map_sizes{layer_idx-1}(3));
    post_up = zeros(size(up, 1)*2, size(up, 2)*2, config.feature_map_sizes{layer_idx-1}(3));
    for m = 1:config.feature_map_sizes{layer_idx-1}(3)
        post_up(:,:,m) = kron(up(:,:,m), ones(2));
    end
    mem.convBpool{layer_idx}{2} = reshape(post_up, size(post_up, 1)*size(post_up, 2), config.feature_map_sizes{layer_idx-1}(3));
    mem.convBpool{layer_idx}{2} = config.NEW_MEM(mem.convBpool{layer_idx}{2});
end

