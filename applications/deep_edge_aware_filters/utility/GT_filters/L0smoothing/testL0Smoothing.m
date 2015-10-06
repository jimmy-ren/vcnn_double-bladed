
I = im2double(imread('applications/deep_edge_aware_filters/images/1.png'));
tic;
res = L0Smoothing(I, 0.02);
toc;

figure,imshow(res);

