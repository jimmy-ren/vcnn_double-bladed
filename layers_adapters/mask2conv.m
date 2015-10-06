function mask2conv()
    global config mem;
    curr_layer_idx = config.misc.current_layer;
    in = mem.mask_li{curr_layer_idx};
    for m = 1:config.batch_size
        for n = 1:config.chs
            mem.mask_inputs{curr_layer_idx}((n-1)*config.kernel_size(1, 1)*config.kernel_size(1, 2)+1:n*config.kernel_size(1, 1)*config.kernel_size(1, 2), (m-1)*(size(mem.mask_inputs{curr_layer_idx}, 2)/config.batch_size)+1:m*size(mem.mask_inputs{curr_layer_idx}, 2)/config.batch_size) = ...
                    config.IM2COL(in(:,:,n,m), [config.kernel_size(1, 1), config.kernel_size(1, 2)]);
        end
    end
    % only for SR now
    gen_mask_patch_cat_idx_for_super_res(mem.mask_inputs{1});
end

