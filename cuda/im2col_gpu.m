function patches = im2col_gpu(X, psize)

global config data;

ker = get_kernel('im2col', 'im2col.cu');

im_ni = size(X, 1);
im_nj = size(X, 2);
nimgs = numel(X) / (im_ni * im_nj);

p_ni = psize(1);
p_nj = psize(2);
npatches = nimgs * (im_ni - p_ni + 1) * (im_nj - p_nj + 1);

ker = set_grid(ker, npatches);
patches = gpuArray.zeros(p_ni * p_nj, npatches, data.gpu.float_t);
patches = feval(ker.ker, X, patches, ...
                im_ni, im_nj, nimgs, ...
                p_ni, p_nj, npatches);
