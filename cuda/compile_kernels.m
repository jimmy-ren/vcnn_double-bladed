function compile_kernels()
    [dirname, ~, ~] = fileparts(mfilename('fullpath'));

    compile_(dirname, 'im2col.cu');

end


function compile_(dirname, cu_fn)
    system(sprintf('(cd %s && nvcc --ptx %s)', dirname, cu_fn));
end
