function conv2out_mem()
    global config mem;
    idx = reshape(1:config.output_size(1) * config.output_size(2), config.output_size(1), config.output_size(2));
    idx = im2col(idx, config.kernel_size(size(config.kernel_size, 1), :));
    %mem.gen_out_matrix = config.NEW_MEM(idx);
    mem.output = config.NEW_MEM(zeros(config.output_size(1), config.output_size(2), config.output_size(3), config.batch_size));
    
    counts = ones(size(idx));
    counts = reshape(accumarray(idx(:), counts(:)), config.output_size(1), config.output_size(2));
    mem.one_over_add_counts = config.NEW_MEM(1 ./ counts);
    mem.gen_out_matrix = config.NEW_MEM(zeros(size(idx, 1)*config.chs, size(idx, 2)));
    for m = 1:config.chs
        mem.gen_out_matrix((m-1)*size(idx, 1)+1:m*size(idx, 1), :) = idx + ((m-1)*max(max(idx)));
    end
    
    mem.gen_out_matrix = mem.gen_out_matrix(:);
    if config.batch_size > 1
        h = size(mem.gen_out_matrix, 1);
        mem.gen_out_matrix = repmat(mem.gen_out_matrix, [config.batch_size 2]);
        for m = 1:config.batch_size
            mem.gen_out_matrix((m-1)*h+1:m*h, 2) = m;
        end
    end    
end
