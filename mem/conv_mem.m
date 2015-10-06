function conv_mem(layer_idx)
    global mem;
    mem.layer_inputs{layer_idx} = 0;
    mem.activations{layer_idx} = 0;    
end
