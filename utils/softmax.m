function y = softmax(x)
    M = bsxfun(@minus, x, max(x, [], 1));
    numerator = exp(M);
    denominator = sum(numerator) + 0.00001;    
    y = bsxfun(@rdivide, numerator, denominator) + 0.00001;    
end


