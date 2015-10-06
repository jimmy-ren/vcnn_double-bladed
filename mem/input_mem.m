function input_mem()
    global config mem;
    mem.layer_inputs{1} = config.NEW_MEM(zeros(config.kernel_size(1, 1)*config.kernel_size(1, 2)*config.chs, ...
                    (config.input_size(1)-config.kernel_size(1, 1)+1)*(config.input_size(2)-config.kernel_size(1, 2)+1)*config.batch_size));
	mem.activations{1} = config.NEW_MEM(zeros(config.feature_map_sizes{1}(3), config.feature_map_sizes{1}(1)*config.feature_map_sizes{1}(2)));
end

