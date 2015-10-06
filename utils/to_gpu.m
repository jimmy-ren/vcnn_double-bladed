function xx = to_gpu(x)
    xx = gpuArray(single(x));
end