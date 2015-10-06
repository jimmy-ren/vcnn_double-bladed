#define IDX3(X, n1, n2, n3, i1, i2, i3) (X[(i1)*((n2)*(n3)) + (i2)*(n3) + (i3)])

template<class T>
__device__ void im2col_ker(const T *im, T *patches,
                           int im_ni, int im_nj, int nimgs,
                           int p_ni, int p_nj, int npatches)
{
	int total_threads = gridDim.x * blockDim.x;
    int patch = blockIdx.x * blockDim.x + threadIdx.x;
    int patches_per_img = npatches / nimgs;

	for (; patch < npatches; patch += total_threads) {
		int im_k = patch / patches_per_img;  /* image index */
		int im_j0 = patch / (im_ni - p_ni + 1);  /* patch topleft j in image */
		int im_i0 = patch % (im_ni - p_ni + 1);  /* patch topleft i in image */

		for (int pj = 0; pj < p_nj; ++pj) {
			for (int pi = 0; pi < p_ni; ++pi) {
				IDX3(patches, npatches, p_nj, p_ni,
							  patch, pj, pi)
						= IDX3(im, nimgs, im_nj, im_ni,
								   im_k, im_j0 + pj, im_i0 + pi);
			}
		}
	}
}


template<class T>
__device__ void scol2im_ker(T *im, const T *patches,
                            int im_ni, int im_nj, int nimgs,
                            int p_ni, int p_nj, int npatches)
{
	int total_threads = gridDim.x * blockDim.x;
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
	int valid_nj = im_nj - p_nj + 1;
	int valid_ni = im_ni - p_ni + 1;
    int npixels = nimgs * im_nj * im_ni;
    int patches_per_img = npatches / nimgs;

	for (; pixel < npixels; pixel += total_threads) {
		T x = 0;

		int im_k = pixel / (im_ni * im_nj);  /* image index */
		int im_j = pixel / im_ni;  /* pixel in image */
		int im_i = pixel % im_ni;

		for (int pj = 0; pj < p_nj; ++pj) {
			for (int pi = 0; pi < p_ni; ++pi) {
				int im_pj = im_j - pj;  /* topleft of patch in image */
				int im_pi = im_i - pi;  /* topleft of patch in image */
				if (im_pi < 0 || im_pj < 0 || 
					im_pj >= valid_nj || im_pi >= valid_ni)
						continue;

				int patch = im_k * patches_per_img + im_pj * valid_ni + im_pi;
				x += IDX3(patches, npatches, p_nj, p_ni,
								   patch, pj, pi);
			}
		}

		IDX3(im, nimgs, im_nj, im_ni,
				 im_k, im_j, im_i) = x;
	}
}

__global__ void im2col_d(const double *im, double *patches,
                         int im_ni, int im_nj, int nimgs,
                         int p_ni, int p_nj, int npatches)
{
    im2col_ker<double>(im, patches,
                       im_ni, im_nj, nimgs,
                       p_ni, p_nj, npatches);
}

__global__ void scol2im_d(double *im, const double *patches,
                          int im_ni, int im_nj, int nimgs,
                          int p_ni, int p_nj, int npatches)
{
    scol2im_ker<double>(im, patches,
                        im_ni, im_nj, nimgs,
                        p_ni, p_nj, npatches);
}

__global__ void im2col_f(const float *im, float *patches,
                         int im_ni, int im_nj, int nimgs,
                         int p_ni, int p_nj, int npatches)
{
    im2col_ker<float>(im, patches,
                      im_ni, im_nj, nimgs,
                      p_ni, p_nj, npatches);
}

__global__ void scol2im_f(float *im, const float *patches,
                          int im_ni, int im_nj, int nimgs,
                          int p_ni, int p_nj, int npatches)
{
    scol2im_ker<float>(im, patches,
                       im_ni, im_nj, nimgs,
                       p_ni, p_nj, npatches);
}

