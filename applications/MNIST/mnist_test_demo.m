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
config.batch_size = 200;
init(1);
load('w_mnist.mat');
config.weights = model.weights;

% load test images and labels
testImages = loadMNISTImages('t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
idx = find(testLabels==0);
testLabels(idx) = 10;

display_samples(testImages(:,1:100)); % Show the first 100 samples

% pad test set
testImagesSquare = reshape(testImages, 28, 28, size(testImages, 2));
testImages_padded = zeros(32, 32, size(testImagesSquare, 3));
for i = 1 : size(testImagesSquare, 3)
    testImages_padded(:,:,i) = padarray(testImagesSquare(:,:,i), [2, 2], 0);
end
fprintf('Test image padding done!\n');

% put the data on the device (e.g. GPU)
test_imgs = config.NEW_MEM(testImages_padded);
test_imgs = reshape(test_imgs, size(test_imgs, 1), size(test_imgs, 2), 1, size(test_imgs, 3));
test_labels = config.NEW_MEM(testLabels);

fprintf('Testing...\n');
fake_output_for_test = config.NEW_MEM(zeros(config.output_size(3), config.batch_size));
outputs = zeros(10, size(test_imgs, 4));
for m = 1:size(test_imgs, 4) / config.batch_size
    val_in = test_imgs(:,:,:,(m-1)*config.batch_size+1:m*config.batch_size);
    op_test_pipe(val_in, fake_output_for_test);
    outputs(:, (m-1)*config.batch_size+1:m*config.batch_size) = gather(mem.output);
end
[max_val, estimated_labels] = max(outputs);
estimated_labels = estimated_labels';
correct_count = length(find(estimated_labels == test_labels));
acc = correct_count / size(test_imgs, 4);
fprintf('Overall test accuracy: %f%%\n', acc * 100);



