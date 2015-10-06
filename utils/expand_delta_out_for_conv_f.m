function expand_delta_out_for_conv_f()
    global config mem;
    current_layer = config.misc.current_layer;
    mem.deltas{current_layer} = bsxfun(@times, mem.deltas{current_layer}, mem.one_over_add_counts);
    last_kernal = config.kernel_size(size(config.kernel_size, 1), :);
    last_layer_size = config.feature_map_sizes{length(config.feature_map_sizes)};
    expanded_delta = config.NEW_MEM(zeros(last_kernal(1) * last_kernal(2) * config.chs, last_layer_size(1)*last_layer_size(2), config.batch_size));
    h_step = last_kernal(1) * last_kernal(2);
    for b = 1:config.batch_size
        h_start = 0;
        for c = 1:config.chs
            expanded_delta(h_start+1:h_start+h_step, :, b) = config.IM2COL(mem.deltas{current_layer}(:, :, c, b), ...
                                                                           [last_kernal(1), last_kernal(2)]);
            h_start = h_start + h_step;
        end
    end
    %mem.deltas{current_layer} = expanded_delta;
    mem.deltas{current_layer} = reshape(expanded_delta, size(mem.activations{length(mem.activations)}));
end

