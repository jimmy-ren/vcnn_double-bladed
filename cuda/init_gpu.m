function init_gpu(device_index)

global config data;

config.gpu.enable = 1;%getvalue(config, 'gpu.enable', 0);

if ~config.gpu.enable && nargin == 0
    return
end

reinit = 0;

if nargin == 0
    device_index = getvalue(data, 'gpu.device_index', 1);
end

data.gpu.device_index = device_index;

dev = getvalue(data, 'gpu.dev', []);
if isempty(dev) || dev.Index ~= device_index
    data.gpu.dev = [];

    gpuDevice(device_index);

    dev = gpuDevice();
    data.gpu.dev = dev;
    data.gpu.max_grid_size = dev.MaxGridSize;
    if numel(data.gpu.max_grid_size) == 3
        data.gpu.max_grid_size = [data.gpu.max_grid_size(1), 1];
    end
    data.gpu.float_t = 'double';
    data.gpu.kernels = struct();

    compile_kernels();

    reinit = 1;
end

precision = 'single';%getvalue(config.gpu, 'precision', 'double');
if ~strcmp(precision, data.gpu.float_t)
    data.gpu.float_t = precision;
    reinit = 1;
end

if reinit
    data.gpu.gen = randi(65535);  % gen id for persistent data validation
end
