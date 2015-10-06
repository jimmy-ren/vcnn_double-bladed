function cost = L2_norm(out, labels, batch_size)
    cost = sum((labels(:) - out(:)).^2) / 2;
    cost = cost / batch_size;
end

