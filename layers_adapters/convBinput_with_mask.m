function convBinput_with_mask()
    global config mem;
    curr_layer_idx = config.misc.current_layer;
    eps = 0.000;
    
    WM = config.weights{curr_layer_idx+1} * mem.mask_inputs{curr_layer_idx+1} + eps;
    WI = config.weights{curr_layer_idx+1} * mem.layer_inputs{curr_layer_idx+1};
    
    WM = permute(WM, [1 3 2]);
    WI = permute(WI, [1 3 2]);
    I = permute(mem.layer_inputs{curr_layer_idx+1}, [3 1 2]);
    M = permute(mem.mask_inputs{curr_layer_idx+1}, [3 1 2]);
    
    mem.mask_intermediate{curr_layer_idx+1} = bsxfun(mem.mask_fun, WM, I) - bsxfun(mem.mask_fun, WI, M);
    mem.mask_intermediate{curr_layer_idx+1} = bsxfun(@times, mem.mask_intermediate{curr_layer_idx+1}, permute(mem.deltas{curr_layer_idx+1}, [1 3 2])./(WM.^2));

    %{
    for m = 1:size(mem.mask_intermediate{curr_layer_idx+1}, 3)
        mem.mask_intermediate{curr_layer_idx+1}(:,:,m) = ...                
                WM(:,m) * mem.layer_inputs{curr_layer_idx+1}(:,m)' - WI(:,m) * mem.mask_inputs{curr_layer_idx+1}(:,m)';
                %bsxfun(@rdivide, bsxfun(@times, WM(:,m) * mem.layer_inputs{curr_layer_idx+1}(:,m)' - WI(:,m) * mem.mask_inputs{curr_layer_idx+1}(:,m)', mem.deltas{curr_layer_idx+1}(:,m)), WM(:,m).^2);
    end
    mem.mask_intermediate{curr_layer_idx+1} = bsxfun(@times, mem.mask_intermediate{curr_layer_idx+1}, permute(mem.deltas{curr_layer_idx+1}./(WM.^2), [1 3 2]));    
    %}
    mem.grads{curr_layer_idx+1} = sum(mem.mask_intermediate{curr_layer_idx+1}, 3);
    mem.grads{curr_layer_idx+1+config.layer_num} = sum(mem.deltas{curr_layer_idx+1}./(permute(WM, [1 3 2])+eps), 2);
    
    mem.mask_intermediate{curr_layer_idx+1} = 0;    
end


