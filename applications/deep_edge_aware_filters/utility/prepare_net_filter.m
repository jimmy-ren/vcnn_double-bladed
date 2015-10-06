function prepare_net_filter(input_x, input_y, w_path)
    global config mem;
    if(input_x < input_y)
        smaller_edge = input_x;
    else
        smaller_edge = input_y;
    end

    max_patch_size = 270;

    if(smaller_edge <= max_patch_size)
        p_size = smaller_edge;
    else
        p_size = max_patch_size;
    end
    
    load(w_path);
    persistent init_called;
    if(isempty(init_called) || ~strcmp(init_called, w_path))
        fprintf('initializing a new model...\n');
        config = model;
        config.batch_size = 2;
        size_differ = [config.input_size(1)-config.output_size(1) config.input_size(2)-config.output_size(2)];
        config.input_size = [p_size p_size];
        config.output_size = [p_size-size_differ(1) p_size-size_differ(2) config.chs];
        config.size_differ = size_differ;
        init(1);
        load(w_path);
        config.SCALE_INPUT = model.SCALE_INPUT;
        config.SCALE_OUTPUT = model.SCALE_OUTPUT;
        init_called = w_path;
    end
    
    config.weights = model.weights;

    overlap_pixels = 30;
    problem_pixels = 0;
    
    start_rows = [1];
    start_cols = [1];    
    while(1)
        next_start_row = start_rows(length(start_rows)) + p_size - overlap_pixels;
        if(next_start_row + p_size - 1 > input_x)
            next_start_row = input_x - p_size + 1;
            if(next_start_row ~= start_rows(length(start_rows)))
                start_rows = [start_rows next_start_row];
            end
            break;
        else
            start_rows = [start_rows next_start_row];
        end
    end
    
    while(1)
        next_start_col = start_cols(length(start_cols)) + p_size - overlap_pixels;
        if(next_start_col + p_size - 1 > input_y)
            next_start_col = input_y - p_size + 1;
            if(next_start_col ~= start_cols(length(start_cols)))
                start_cols = [start_cols next_start_col];
            end
            break;
        else
            start_cols = [start_cols next_start_col];
        end
    end
    config.MEM.p_size = p_size;
    config.MEM.start_rows = start_rows;
    config.MEM.start_cols = start_cols;
    mem.fake_output_for_test = config.NEW_MEM(zeros(config.output_size(1), config.output_size(2), config.output_size(3), config.batch_size));
end
