function gen_mask_patch_cat_idx_for_super_res(patchified_mask)
    persistent mask_idx;
    if(isempty(mask_idx))
        global mem;
        [v, idx] = max(patchified_mask);
        idx_all = unique(idx);
        mask_idx = {};
        for m = 1:length(idx_all)
            mask_idx{m} = find(idx ==idx_all(m));
        end
        mem.mask_idx = mask_idx;
    end
end


