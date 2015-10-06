%--------------------------------------------------------------------------------------------------------
% The system is created based on the principles described in the following paper
% Jimmy SJ. Ren and Li Xu, "On Vectorization of Deep Convolutional Neural Networks for Vision Tasks", 
% The 29th AAAI Conference on Artificial Intelligence (AAAI-15). Austin, Texas, USA, January 25-30, 2015
% email: jimmy.sj.ren@gmail.com
%--------------------------------------------------------------------------------------------------------
addpath applications/image_denoise/
addpath applications/image_denoise/utility/
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
clear;
global config mem;
denoise_configure();
init(0);

load('data/denoise/val/val_1');
perm = randperm(size(test_samples, 4));
test_samples = test_samples(:,:,:,perm);
test_labels = test_labels(:,:,:,perm);
test_samples = config.NEW_MEM(test_samples(:,:,:,1:1000));
test_labels = config.NEW_MEM(test_labels(:,:,:,1:1000));

count = 0;
cost_avg = 0;
epoc = 0;
points_seen = 0;
display_points = 1000;
save_points = 10000;
max_grad = 1;
fprintf('%s\n', datestr(now, 'dd-mm-yyyy HH:MM:SS FFF'));
for pass = 1:10
    for p = 1:11
        load(strcat('data/denoise/train/patches_', num2str(p), '.mat'));
        perm = randperm(10000);
        samples = samples(:,:,:,perm);
        labels = labels(:,:,:,perm);
        train_imgs = config.NEW_MEM(samples);
        train_labels = config.NEW_MEM(labels);
        
        for i = 1:size(train_labels, 4) / config.batch_size            
            points_seen = points_seen + config.batch_size;
            in = train_imgs(:,:,:,(i-1)*config.batch_size+1:i*config.batch_size);
            out = train_labels(:,:,:,(i-1)*config.batch_size+1:i*config.batch_size);
            out = out((size(in, 1) - config.output_size(1)) / 2 + 1:(size(in, 1) - config.output_size(1)) / 2 + config.output_size(1), ...
                      (size(in, 2) - config.output_size(2)) / 2 + 1:(size(in, 2) - config.output_size(2)) / 2 + config.output_size(2), :, :);
            
            % operate the training pipeline
            op_train_pipe(in, out);
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
                    
                    op_test_pipe(test_samples(:,:,:,(t-1)*config.batch_size+1:t*config.batch_size), t_label);
                    test_out = gather(mem.output);
                    test_cost = test_cost + config.cost;
                end
                test_cost = test_cost / size(test_samples, 4);
                fprintf('\nepoc %d, training avg cost: %f, test avg cost: %f\n', epoc, cost_avg, test_cost);

                save_weights(strcat('applications/image_denoise/results/epoc', num2str(epoc), '.mat'));
                cost_avg = 0;
            end
        end
    end
end