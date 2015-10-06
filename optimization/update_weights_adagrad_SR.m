function update_weights_adagrad_SR()
    global config mem;
    for m = 1:config.misc.mask_type
        config.his_grad{1}{m} = config.his_grad{1}{m} + mem.grads{1}{m} .* mem.grads{1}{m};
        config.weights{1}{m} = config.weights{1}{m} - config.learning_rate * (mem.grads{1}{m} ./ (config.fudge_factor + sqrt(config.his_grad{1}{m})));
    end
    for m = 2:length(config.weights)
        config.his_grad{m} = config.his_grad{m} + mem.grads{m} .* mem.grads{m};
        config.weights{m} = config.weights{m} - config.learning_rate * (mem.grads{m} ./ (config.fudge_factor + sqrt(config.his_grad{m})));
    end
end




