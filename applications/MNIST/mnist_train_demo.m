%--------------------------------------------------------------------------------------------------------
% The system is created based on the principles described in the following paper
% Jimmy SJ. Ren and Li Xu, "On Vectorization of Deep Convolutional Neural Networks for Vision Tasks", 
% The 29th AAAI Conference on Artificial Intelligence (AAAI-15). Austin, Texas, USA, January 25-30, 2015
% email: jimmy.sj.ren@gmail.com
%--------------------------------------------------------------------------------------------------------
addpath applications/
addpath utils/
addpath cuda/
addpath mem/
addpath layers/
addpath layers_adapters/
addpath pipeline/
addpath optimization/
addpath data/MNIST/

clearvars -global config;
clearvars -global mem;
clear;
global config mem;
mnist_configure();
%mnist_configure_deep();
init(0);

images_orig = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
idx = find(labels==0);
labels(idx) = 10;
%display_network(images_orig(:,1:100)); % Show the first 100 images
%disp(labels(1:30));
labelData = zeros(10, size(images_orig, 2));
for i = 1 : length(labels)
    labelData(labels(i), i) = 1;
end
% load test images and labels
testImages = loadMNISTImages('t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
idx = find(testLabels==0);
testLabels(idx) = 10;

% pad training set
images = reshape(images_orig, 28, 28, size(images_orig, 2));
images_padded = padarray(images, [2, 2], 0);
fprintf('Training image padding done!\n');

% pad test set
testImagesSquare = reshape(testImages, 28, 28, size(testImages, 2));
testImages_padded = padarray(testImagesSquare, [2, 2], 0);
fprintf('Test image padding done!\n');

test_imgs = config.NEW_MEM(testImages_padded);
test_imgs = reshape(test_imgs, size(test_imgs, 1), size(test_imgs, 2), 1, size(test_imgs, 3));
test_labels = config.NEW_MEM(testLabels);

count = 0;
cost_avg = 0;
epoc = 0;
points_seen = 0;
display_points = 1000;
save_points = 10000;
fprintf('%s\n', datestr(now, 'dd-mm-yyyy HH:MM:SS FFF'));
% go through the data set 'pass' times
for pass = 1:5
    % divide data into 6 portions, each with 10k samples
    for p = 1:6
        size_to_load = 10000;
        train_imgs = config.NEW_MEM(images_padded(:,:,(p-1)*size_to_load+1:p*size_to_load));
        train_imgs = reshape(train_imgs, size(train_imgs, 1), size(train_imgs, 2), 1, size(train_imgs, 3));
        train_labels = config.NEW_MEM(labelData(:, (p-1)*size_to_load+1:p*size_to_load));
        perm = randperm(size(train_imgs, 4));
        train_imgs = train_imgs(:,:,:,perm);
        train_labels = train_labels(:,perm);

        for i = 1:size_to_load / config.batch_size
            eta = config.learning_rate / (1 + points_seen*config.decay);
            points_seen = points_seen + config.batch_size;
            in = train_imgs(:,:,:,(i-1)*config.batch_size+1:i*config.batch_size);
            out = train_labels(:,(i-1)*config.batch_size+1:i*config.batch_size);
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
                fake_output_for_test = config.NEW_MEM(zeros(size(mem.GT_output)));
                epoc = epoc + 1;
                val_cost = 0;
                val_size = 1000;
                val_size = val_size + (config.batch_size - mod(val_size, config.batch_size));
                outputs = zeros(10, val_size);
                for m = 1:val_size / config.batch_size
                    val_in = test_imgs(:,:,:,(m-1)*config.batch_size+1:m*config.batch_size);
                    op_test_pipe(val_in, fake_output_for_test);
                    outputs(:, (m-1)*config.batch_size+1:m*config.batch_size) = gather(mem.output);
                end
                [max_val, estimated_labels] = max(outputs);
                estimated_labels = estimated_labels';
                correct_count = length(find(estimated_labels == test_labels(1:val_size)));
                acc = correct_count / val_size;
                err = 1 - acc;
                val_cost = val_cost / val_size;

                fprintf('\nTesting err, subcats\n');
                l = test_labels(1:val_size);
                for m = 1:10
                    ll = l(l == m);
                    ee = estimated_labels(l == m);
                    correct_count = length(find(ee == ll));
                    acc2 = correct_count / length(ll);
                    err2 = 1 - acc2;
                    fprintf('%.4f, ', err2);
                end

                for m = 1:val_size / config.batch_size
                    val_in = train_imgs(:,:,:,(m-1)*config.batch_size+1:m*config.batch_size);                
                    op_test_pipe(val_in, fake_output_for_test);
                    outputs(:, (m-1)*config.batch_size+1:m*config.batch_size) = gather(mem.output);
                end
                [max_val, estimated_labels] = max(outputs);
                [max_val, true_labels] = max(train_labels);
                correct_count = length(find(estimated_labels == true_labels(1:val_size)));
                train_acc = correct_count / val_size;
                train_err = 1 - train_acc;

                fprintf('\nTraining err, subcats\n');
                l = true_labels(1:val_size);
                for m = 1:10
                    ll = l(l == m);
                    ee = estimated_labels(l == m);
                    correct_count = length(find(ee == ll));
                    acc2 = correct_count / length(ll);
                    err2 = 1 - acc2;
                    fprintf('%.4f, ', err2);
                end

                save_weights(strcat('applications/MNIST/results/epoc', num2str(epoc), '.mat'));
                fprintf('\nepoc %d, training avg cost: %f, training error: %f, test err: %f\n', epoc, cost_avg, train_err, err);                
                cost_avg = 0;
            end
        end
    end
    
    outputs = zeros(10, size(test_imgs, 3));
    for m = 1:size(test_imgs, 4) / config.batch_size
        val_in = test_imgs(:,:,:,(m-1)*config.batch_size+1:m*config.batch_size);
        op_test_pipe(val_in, fake_output_for_test);
        outputs(:, (m-1)*config.batch_size+1:m*config.batch_size) = gather(mem.output);
    end
    [max_val, estimated_labels] = max(outputs);
    estimated_labels = estimated_labels';
    correct_count = length(find(estimated_labels == test_labels));
    acc = correct_count / size(test_imgs, 4);
    err = 1 - acc;

    fprintf('\n      epoc %d, overall test accuracy: %f\n\n', epoc, acc);
end

