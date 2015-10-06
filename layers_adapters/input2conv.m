function input2conv(in)
    global config mem;
    for m = 1:config.batch_size
        for n = 1:config.chs
            mem.layer_inputs{1}((n-1)*config.kernel_size(1, 1)*config.kernel_size(1, 2)+1:n*config.kernel_size(1, 1)*config.kernel_size(1, 2), (m-1)*(size(mem.layer_inputs{1}, 2)/config.batch_size)+1:m*size(mem.layer_inputs{1}, 2)/config.batch_size) = ...
                    config.SCALE_INPUT(config.IM2COL(in(:,:,n,m), [config.kernel_size(1, 1), config.kernel_size(1, 2)]));
        end
    end
end
