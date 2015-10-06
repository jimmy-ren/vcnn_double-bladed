function ker = set_grid(ker, nthreads)

global data;

block_size = min(nthreads, 512);
grid_size = min([ceil(nthreads / block_size), 1], data.gpu.max_grid_size);
if any(ker.block_size ~= block_size) || any(ker.grid_size ~= grid_size)
    %gpu_sync();
    ker.block_size = block_size;
    ker.grid_size = grid_size;
    ker.ker.ThreadBlockSize = block_size;
    ker.ker.GridSize = grid_size;
    data.gpu.kernels.(ker.name) = ker;
end
