function X = scol2im_gpu(patches, psize, im_ni, im_nj, method)

global config data;

ker = get_kernel('scol2im', 'im2col.cu');

nimgs = 1;

assert(strcmp(method, 'sum'));

p_ni = psize(1);
p_nj = psize(2);
[p_npix, npatches] = size(patches);
assert(p_npix == p_ni * p_nj);

X = gpuArray.zeros(im_ni, im_nj, data.gpu.float_t);
npixels = im_ni * im_nj;

ker = set_grid(ker, npixels);
X = feval(ker.ker, X, patches, ...
          im_ni, im_nj, nimgs, ...
          p_ni, p_nj, npatches);

