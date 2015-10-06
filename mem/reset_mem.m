function reset_mem()
    global mem config;
    mem.layer_inputs = {};
    mem.activations = {};
    mem.orig_activation_size = {};
    mem.mask_inputs = {};
    mem.mask_activations = {};
    mem.mask_intermediate = {};
    mem.deltas = {};    
    mem.conv2conv = {};
    mem.pool2conv = {};
    mem.gen_out_matrix = {};
    mem.convBpool = {};
    mem.convBconv = {};
    mem.pooling_matrix ={};
    mem.output = 0;
    mem.one_over_add_counts = 0;
    mem.grads = {};
    if strcmp(config.forward_pass_scheme{1}, 'conv_v_sr')
        % for SR only
        mem.grads{1} = {};
    end
end

