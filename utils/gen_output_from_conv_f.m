function gen_output_from_conv_f()
    global config mem;
    %h_step = config.kernel_size(size(config.kernel_size, 1), 1) * config.kernel_size(size(config.kernel_size, 1), 2);
    %out_act = reshape(mem.activations{length(mem.activations)*config.is_training+1*(~config.is_training)}, size(mem.activations{length(mem.activations)*config.is_training+1*(~config.is_training)}, 1), [], config.batch_size);    
    
    %{
    for b = 1:config.batch_size
        %h_start = 0;
        %for c = 1:config.chs
        %    tmp = out_act(h_start+1:h_start+h_step, :, b);
        %    mem.output(:, :, c, b) = reshape(accumarray(mem.gen_out_matrix(:), tmp(:)), size(mem.output, 1), size(mem.output, 2));
        %    h_start = h_start + h_step;
        %end
        mem.output(:, :, :, b) = reshape(accumarray(mem.gen_out_matrix(:), reshape(out_act(:, :, b), size(out_act,1)*size(out_act,2), 1)), size(mem.output,1), size(mem.output,2), size(mem.output,3));
    end
    %}
    config.SCALE_OUTPUT();  % this scaling may not be necessary in the future
    mem.output = bsxfun(@times, reshape(accumarray(mem.gen_out_matrix, mem.activations{length(mem.activations)}(:)), size(mem.output)), mem.one_over_add_counts);
    %mem.output = bsxfun(@times, mem.output, mem.one_over_add_counts);
end


