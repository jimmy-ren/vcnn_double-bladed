function conv_layer_idx = get_conv_layer_idx_from_layer_idx(layer_idx)
    global config;
    c = 0;
    for m = 1:layer_idx
        if strfind(config.forward_pass_scheme{m}, 'conv')
            c = c + 1;
        end
    end
    conv_layer_idx = c;
end

