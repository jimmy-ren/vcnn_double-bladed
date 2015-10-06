function numgrad = computeNumericalGradient_SR(image, label, weight_idx)
    global config mem;
    estimatedGrad = mem.grads;
    epsilon = config.NEW_MEM(0.01);
    % Initialize numgrad with zeros
    numgrad = zeros(size(config.weights{1}{weight_idx}));
    
    for x = 1:size(config.weights{1}{weight_idx}, 1)
        for y = 1:size(config.weights{1}{weight_idx}, 2)
            config.weights{1}{weight_idx}(x, y) = config.weights{1}{weight_idx}(x, y) + epsilon;
            op_train_pipe(image, label);
            cost1 = config.cost;
            config.weights{1}{weight_idx}(x, y) = config.weights{1}{weight_idx}(x, y)  - (2*epsilon);
            op_train_pipe(image, label);
            cost2 = config.cost;
            config.weights{1}{weight_idx}(x, y) = config.weights{1}{weight_idx}(x, y) + epsilon;
            
            % compute the numerical 
            temp = (cost1 - cost2) / (2 * epsilon);
            numgrad(x, y) = gather(temp);

            diff = norm(temp-estimatedGrad{1}{weight_idx}(x, y));
            threshold = 10^-10;
            %if(diff > threshold)
                fprintf('numerical: %i, ', temp);
                fprintf('estimated: %i. \n', estimatedGrad{1}{weight_idx}(x, y));
                fprintf('loop (%d, %d)\n', x, y);
                fprintf('diff too large! diff: %i.\n', diff);
            %end
        end
    end
end



