function output = apply_net_filter(input_v, input_h)
    global config mem;    
    input_size = size(input_v);
    %output = config.NEW_MEM(zeros(size(input_v, 1), size(input_v, 2), config.chs, config.batch_size));
    output = zeros(size(input_v, 1), size(input_v, 2), config.chs, config.batch_size);
    p_size = config.MEM.p_size;
    size_differ = config.size_differ;
    %count = config.NEW_MEM(zeros(size(input_v, 1), size(input_v, 2)));
    count = zeros(size(input_v, 1), size(input_v, 2));
    for v = 1 : length(config.MEM.start_rows)
        for h = 1 : length(config.MEM.start_cols)
            v_start = config.MEM.start_rows(v);
            v_end = v_start + p_size - 1;
            h_start = config.MEM.start_cols(h);
            h_end = h_start + p_size - 1;
            if(v_end > input_size(1))
                v_end = input_size(1);
                v_start = v_end - p_size + 1;
            end
            if(h_end > input_size(2))
                h_end = input_size(2);
                h_start = h_end - p_size + 1;
            end
            input_piece = cat(4, input_v(v_start:v_end, h_start:h_end,:), permute(input_h(v_start:v_end, h_start:h_end,:), [2 1 3]));

            op_test_pipe(input_piece, mem.fake_output_for_test);
            %output_piece = mem.output;
            output_piece = gather(mem.output);
            if(size_differ(1) ~= size_differ(2))
                output_piece = output_piece(size_differ(2)/2+1:size(output_piece,1)-size_differ(2)/2, size_differ(1)/2+1:size(output_piece,2)-size_differ(1)/2,:,:);
            end
            output_piece(:,:,:,2) = permute(output_piece(:,:,:,2), [2 1 3]);

            output(v_start+(max(size_differ)/2):v_end-(max(size_differ)/2), h_start+(max(size_differ)/2):h_end-(max(size_differ)/2),:,:) = ...
                    output(v_start+(max(size_differ)/2):v_end-(max(size_differ)/2), h_start+(max(size_differ)/2):h_end-(max(size_differ)/2),:,:) + output_piece;
            count(v_start+(max(size_differ)/2):v_end-(max(size_differ)/2), h_start+(max(size_differ)/2):h_end-(max(size_differ)/2)) = ...
                    count(v_start+(max(size_differ)/2):v_end-(max(size_differ)/2), h_start+(max(size_differ)/2):h_end-(max(size_differ)/2)) + 1;
        end
    end
    count = max(count, 1);
    output = bsxfun(@rdivide, output, count);
    %output = gather(output);
    
    %output = output_piece;
    %output = padarray(output, [1 1]);
end


