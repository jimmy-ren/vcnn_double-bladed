function update_weights_adagrad()
    global config mem;
    for m = 1:length(config.weights)
        config.his_grad{m} = config.his_grad{m} + mem.grads{m} .* mem.grads{m};
        config.weights{m} = config.weights{m} - config.learning_rate * (mem.grads{m} ./ (config.fudge_factor + sqrt(config.his_grad{m})));
    end
end

