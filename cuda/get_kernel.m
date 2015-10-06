function ker = get_kernel(ker_name, cu_filename)

global data;

switch data.gpu.float_t
case 'double',
    ker_name = [ker_name '_d'];
case 'single',
    ker_name = [ker_name '_f'];
end

if ~isfield(data.gpu.kernels, ker_name)
    cu_filename = fullfile('cuda', cu_filename);
    ptx_filename = regexprep(cu_filename, '\.cu$', '.ptx');
    ker = struct();
    ker.name = ker_name;
    ker.ker = parallel.gpu.CUDAKernel(ptx_filename, cu_filename, ker_name);
    ker.block_size = ker.ker.ThreadBlockSize;
    ker.grid_size = ker.ker.GridSize;
    data.gpu.kernels.(ker_name) = ker;
end

ker = data.gpu.kernels.(ker_name);
