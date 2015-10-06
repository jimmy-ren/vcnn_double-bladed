addpath applications/deep_edge_aware_filters/utility/GT_filters/
addpath applications/deep_edge_aware_filters/utility/GT_filters/L0smoothing/
addpath data/

clear;
patch_dim = 64;
num_patches = 10000;
listing = dir('data/deepeaf/BSDS500/*.jpg');
for m = 1 : 101
    fprintf('Extracting patch batch: %d / %d\n', m, 101);
    % extract random patches
    samples = zeros(patch_dim, patch_dim, 3, num_patches);
    labels = zeros(size(samples));
    for i = 1 : num_patches / 8
        if (mod(i,100) == 0)
            fprintf('Extracting patch: %d / %d\n', i*8, num_patches);
        end
        
        r_idx = random('unid', size(listing, 1));
        I = imread(strcat('data/deepeaf/BSDS500/', listing(r_idx).name));
        orig_img_size = size(I);
        r = random('unid', orig_img_size(1) - patch_dim + 1);
        c = random('unid', orig_img_size(2) - patch_dim + 1);
        
        patch = I(r:r+patch_dim-1, c:c+patch_dim-1, :);
        patchHoriFlipped = flipdim(patch, 2);
        idx_list = (i-1)*8+1:(i-1)*8+8;
        for idx = 1:4
            patch_rotated = im2double(imrotate(patch, (idx-1)*90));
            patch_filtered = GT_filter(patch_rotated);
            [vin, vout] = EdgeExtract(im2double(patch_rotated), im2double(patch_filtered));
            samples(:,:,:,idx_list(idx)) = vin;
            labels(:,:,:,idx_list(idx)) = vout;            
            
            patch_rotated = im2double(imrotate(patchHoriFlipped, (idx-1)*90));
            patch_filtered = GT_filter(patch_rotated);  
            [vin, vout] = EdgeExtract(im2double(patch_rotated), im2double(patch_filtered));            
            samples(:,:,:,idx_list(idx+4)) = vin;
            labels(:,:,:,idx_list(idx+4)) = vout;            
        end
    end
    samples = single(samples);
    labels = single(labels);
    % save it
    filename = strcat('data/deepeaf/L0/train/patches_', num2str(m));
    save(filename, '-v7.3', 'samples', 'labels');
end

