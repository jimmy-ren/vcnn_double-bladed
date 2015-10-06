function [PSNR, MSE] = cal_psnr(I,K)
    [M,N,D] = size(I);
    Diff = double(I)-double(K);
    MSE = sum(Diff(:).^2)/numel(I);
    PSNR=10*log10(255^2/MSE);
end