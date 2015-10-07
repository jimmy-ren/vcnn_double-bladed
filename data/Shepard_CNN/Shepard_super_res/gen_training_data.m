clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% indicate the super-resolution factor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
down_factor = 3;    % 2, 3 or 4
image_path = 'data/deepeaf/BSDS500/';
data_chs = 1;   % 1 or 3
training_mat_path = 'data/Shepard_CNN/Shepard_super_res/x3/train_1ch/';
val_mat_path = 'data/Shepard_CNN/Shepard_super_res/x3/val_1ch/';

patch_dim = 48;
num_patches = 20000;
listing = dir(strcat(image_path, '*.jpg'));
mat_num = 51;
for m = 1 : mat_num
    fprintf('Extracting patch batch: %d / %d\n', m, mat_num);
    % extract random patches
    images = zeros(patch_dim, patch_dim, data_chs, num_patches);
    labels = zeros(size(images));
    for i = 1 : num_patches
        if (mod(i,100) == 0)
            fprintf('Extracting patch: %d / %d\n', i, num_patches);
        end
        
        r_idx = random('unid', size(listing, 1));
        I = im2double(imread(strcat(image_path, listing(r_idx).name)));
        orig_img_size = size(I);
        r = random('unid', orig_img_size(1) - patch_dim + 1);
        c = random('unid', orig_img_size(2) - patch_dim + 1);
        
        patch = I(r:r+patch_dim-1, c:c+patch_dim-1, :);
        
        if(data_chs == 1)
            patch = rgb2ycbcr(patch);
            patch = patch(:,:,1);
        end        
        
        patch_down = imresize(patch, [patch_dim/down_factor, patch_dim/down_factor], 'bicubic');
        patch_up = imresize(patch_down, [patch_dim, patch_dim], 'nearest');
        images(:,:,:,i) = patch_up;
        labels(:,:,:,i) = patch;
        %imshow([patch_up patch]); pause;
    end
    % save it
    images = single(images);
    labels = single(labels);
    
    if(m >= mat_num)
        % save validation data
        filename = strcat(val_mat_path, 'val_1');
        images_t = images;
        labels_t = labels;
        save(filename, '-v7.3', 'images_t', 'labels_t');
    else
        % save training data
        filename = strcat(training_mat_path, 'patches_', num2str(m));
        save(filename, '-v7.3', 'images', 'labels');
    end
end



