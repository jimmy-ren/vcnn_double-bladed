function cost = cross_entropy(out, labels, batch_size)
    cost = -sum(labels .* log(out));
    cost = sum(cost) / batch_size;
end

