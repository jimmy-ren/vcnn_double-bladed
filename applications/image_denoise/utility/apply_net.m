function output = apply_net(input)
    global config mem;
    fake_output_for_test = config.NEW_MEM(zeros(config.output_size(1), config.output_size(2), config.output_size(3), config.batch_size));
    input_size = size(input);
    output = config.NEW_MEM((zeros(size(input, 1), size(input, 2), config.chs)));
    p_size = config.MEM.p_size;
    size_differ = config.size_differ;
    count = zeros(size(input, 1), size(input, 2));
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
            input_piece = input(v_start:v_end, h_start:h_end,:);

            op_test_pipe(input_piece, fake_output_for_test);
            output_piece = mem.output;

            output(v_start+(size_differ(1)/2):v_end-(size_differ(1)/2), h_start+(size_differ(2)/2):h_end-(size_differ(2)/2),:) = ...
                    output(v_start+(size_differ(1)/2):v_end-(size_differ(1)/2), h_start+(size_differ(2)/2):h_end-(size_differ(2)/2),:) + output_piece;
            count(v_start+(size_differ(1)/2):v_end-(size_differ(1)/2), h_start+(size_differ(2)/2):h_end-(size_differ(2)/2)) = ...
                    count(v_start+(size_differ(1)/2):v_end-(size_differ(1)/2), h_start+(size_differ(2)/2):h_end-(size_differ(2)/2)) + 1;
        end
    end
    count = max(count, 1);
    output = bsxfun(@rdivide, output, count);
end


