function scale_output()
    global config mem;
    % this scaling may not be necessary in the future
    mem.activations{length(mem.activations)} = mem.activations{length(mem.activations)} .* config.NEW_MEM(config.misc.data_sd) + config.NEW_MEM(config.misc.data_mean);
end



