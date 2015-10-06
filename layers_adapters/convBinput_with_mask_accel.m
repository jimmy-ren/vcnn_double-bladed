function convBinput_with_mask_accel()
    global config mem;
    curr_layer_idx = config.misc.current_layer;
    
    for m = 1:config.misc.mask_type
        I = mem.layer_inputs{curr_layer_idx+1}(:,mem.mask_idx{m});
        M = mem.mask_inputs{curr_layer_idx+1}(:,mem.mask_idx{m});
        delta = mem.deltas{curr_layer_idx+1}(:,mem.mask_idx{m});
        WM = config.weights{curr_layer_idx+1} * M;
        WI = config.weights{curr_layer_idx+1} * I;
        if(m == 1)
            mem.grads{curr_layer_idx+1} = bsxfun(@rdivide, WM .* delta * I' - WI .* delta * M', WM(:,1).^2);
        else
            mem.grads{curr_layer_idx+1} = mem.grads{curr_layer_idx+1} + bsxfun(@rdivide, WM .* delta * I' - WI .* delta * M', WM(:,1).^2);
        end
    end
    WM = config.weights{curr_layer_idx+1} * mem.mask_inputs{curr_layer_idx+1};
    WM = permute(WM, [1 3 2]);
    mem.grads{curr_layer_idx+1+config.layer_num} = sum(mem.deltas{curr_layer_idx+1}./permute(WM, [1 3 2]), 2);
    
    %{
    WM = config.weights{curr_layer_idx+1} * mem.mask_inputs{curr_layer_idx+1};
    WI = config.weights{curr_layer_idx+1} * mem.layer_inputs{curr_layer_idx+1};
    
    WM = permute(WM, [1 3 2]);
    WI = permute(WI, [1 3 2]);
    I = permute(mem.layer_inputs{curr_layer_idx+1}, [3 1 2]);
    M = permute(mem.mask_inputs{curr_layer_idx+1}, [3 1 2]);    

    mem.mask_intermediate{curr_layer_idx+1} = bsxfun(mem.mask_fun, WM, I) - bsxfun(mem.mask_fun, WI, M);
    mem.mask_intermediate{curr_layer_idx+1} = bsxfun(@times, mem.mask_intermediate{curr_layer_idx+1}, permute(mem.deltas{curr_layer_idx+1}, [1 3 2])./(WM.^2));
    
    mem.grads{curr_layer_idx+1} = sum(mem.mask_intermediate{curr_layer_idx+1}, 3);
    mem.grads{curr_layer_idx+1+config.layer_num} = sum(mem.deltas{curr_layer_idx+1}./permute(WM, [1 3 2]), 2);
    
    mem.mask_intermediate{curr_layer_idx+1} = 0;
    %}
end


