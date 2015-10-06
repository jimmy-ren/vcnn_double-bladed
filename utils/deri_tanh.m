function deri = deri_tanh(a)
    % derivative of tanh(x) is 1-tanh(x).^2
    global config;
    tmp = config.NEW_MEM(ones(size(a)));
    deri = tmp - (a.^2);
end

