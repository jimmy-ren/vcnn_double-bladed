function y = scale_input(x)
    global config;
    % this scaling may not be necessary in the future
    y = (x - config.NEW_MEM(config.misc.data_mean)) ./ config.NEW_MEM(config.misc.data_sd);
end




