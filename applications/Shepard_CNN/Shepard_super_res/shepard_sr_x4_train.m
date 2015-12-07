%--------------------------------------------------------------------------------------------------------
% The system is created based on the principles described in the following paper
% Jimmy SJ. Ren, Li Xu, Qiong Yan, Wenxiu Sun, "Shepard Convolutional Neural Networks", 
% Advances in Neural Information Processing Systems (NIPS 2015)
% email: jimmy.sj.ren@gmail.com
%--------------------------------------------------------------------------------------------------------
addpath applications/Shepard_CNN/Shepard_super_res/
addpath utils/
addpath cuda/
addpath mem/
addpath layers/
addpath layers_adapters/
addpath optimization/
addpath pipeline/
addpath data/

clearvars -global config;
clearvars -global mem;
clear gen_mask_patch_cat_idx_for_super_res;
clear;
global config mem;
shepard_sr_x4_configure();
init(0);

load('data/Shepard_CNN/Shepard_super_res/x4/val_1ch/val_1');

%images_t = reshape(images_t, size(images_t,1), size(images_t,2), 1, size(images_t,3));
%labels_t = reshape(labels_t, size(labels_t,1), size(labels_t,2), 1, size(labels_t,3));

perm = randperm(size(images_t, 4));
images_t = images_t(:,:,:,perm);
labels_t = labels_t(:,:,:,perm);
test_samples = config.NEW_MEM(images_t(:,:,:,1:1000));
test_labels = config.NEW_MEM(labels_t(:,:,:,1:1000));

mask = config.NEW_MEM([1 0 0 0;0 0 0 0;0 0 0 0;0 0 0 0]);
mask = repmat(mask, config.input_size(1)/4, config.input_size(2)/4, config.chs);
%mask = config.NEW_MEM([1 0; 0 0]);
%mask = repmat(mask, config.input_size(1)/2, config.input_size(2)/2, config.chs);
mask = repmat(mask, 1,1,1,config.batch_size);

count = 0;
cost_avg = 0;
epoc = 0;
points_seen = 0;
display_points = 5000;
save_points = 50000;
max_grad = 1;
fprintf('%s\n', datestr(now, 'dd-mm-yyyy HH:MM:SS FFF'));
for pass = 1:10
    for p = 1:50
        load(strcat('data/Shepard_CNN/Shepard_super_res/x4/train_1ch/patches_', num2str(p), '.mat'));
        
        %images = reshape(images, size(images,1), size(images,2), 1, size(images,3));
        %labels = reshape(labels, size(labels,1), size(labels,2), 1, size(labels,3));
        
        perm = randperm(20000);
        images = images(:,:,:,perm);
        labels = labels(:,:,:,perm);
        train_imgs = config.NEW_MEM(images);
        train_labels = config.NEW_MEM(labels);
        
        for i = 1:size(train_labels, 4) / config.batch_size            
            points_seen = points_seen + config.batch_size;
            in = train_imgs(:,:,:,(i-1)*config.batch_size+1:i*config.batch_size);
            %in = train_labels(:,:,:,(i-1)*config.batch_size+1:i*config.batch_size);
            out = train_labels(:,:,:,(i-1)*config.batch_size+1:i*config.batch_size);
            out = out((size(in, 1) - config.output_size(1)) / 2 + 1:(size(in, 1) - config.output_size(1)) / 2 + config.output_size(1), ...
                      (size(in, 2) - config.output_size(2)) / 2 + 1:(size(in, 2) - config.output_size(2)) / 2 + config.output_size(2), :, :);
            
                  
            % make the mask list
            mask_li = {};
            mask_li{1} = mask;
            % operate the training pipeline
            op_train_pipe_with_mask(in.*mask, mask_li, out);
            % update the weights
            config.UPDATE_WEIGHTS();
            
            if(cost_avg == 0)
                cost_avg = config.cost;
            else
                cost_avg = (cost_avg + config.cost) / 2;
            end

            % display point
            if(mod(points_seen, display_points) == 0)
                count = count + 1;
                fprintf('%d ', count);
            end
            % save point
            if(mod(points_seen, save_points) == 0)
                fprintf('\n%s', datestr(now, 'dd-mm-yyyy HH:MM:SS FFF'));
                epoc = epoc + 1;
                test_cost = 0;
                for t = 1:size(test_samples, 4) / config.batch_size
                    t_label = test_labels(:,:,:,(t-1)*config.batch_size+1:t*config.batch_size);
                    t_label = config.NEW_MEM(t_label((size(in, 1) - config.output_size(1)) / 2 + 1:(size(in, 1) - config.output_size(1)) / 2 + config.output_size(1), ...
                                            (size(in, 2) - config.output_size(2)) / 2 + 1:(size(in, 2) - config.output_size(2)) / 2 + config.output_size(2), :));
                    
                    op_test_pipe_with_mask(test_samples(:,:,:,(t-1)*config.batch_size+1:t*config.batch_size).*mask, mask_li, t_label);
                    
                    test_out = gather(mem.output);
                    test_cost = test_cost + config.cost;
                end
                test_cost = test_cost / (size(test_samples, 4) / config.batch_size);
                fprintf('\nepoc %d, training avg cost: %f, test avg cost: %f\n', epoc, cost_avg, test_cost);                
                
                save_weights(strcat('applications/Shepard_CNN/Shepard_super_res/results/shepard_layer_x4/epoc', num2str(epoc), '.mat'));
                
                cost_avg = 0;
            end
        end
    end
end

