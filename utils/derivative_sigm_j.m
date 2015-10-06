function deri = derivative_sigm_j(a)
    % derivative of tanh(x) is 1-tanh(x).^2
    %tmp = gones(size(a));
    %tmp = gpuArray(ones(size(a)));
    %deri = (tmp - (a / 1.7159).^2) * 1.7159 * 0.6667;
    %deri = tmp - (a.^2);
    %deri = gsingle(not(not(max(0, a))));
    deri = gpuArray(not(not(max(0, a))));
end


