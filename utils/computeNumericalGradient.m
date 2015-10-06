function numgrad = computeNumericalGradient(image, label, weight_idx)
    global config mem;
    estimatedGrad = mem.grads;
    epsilon = config.NEW_MEM(0.01);
    % Initialize numgrad with zeros
    numgrad = zeros(size(config.weights{weight_idx}));
    %{
    % % % % % % % % % % % % % % % %
    N = size(config.weights.C1, 1) * size(config.weights.C1, 2);
    try % Initialization
       ppm = ParforProgressStarter2('test', N, 0.1);
    catch me % make sure "ParforProgressStarter2" didn't get moved to a different directory
       if strcmp(me.message, 'Undefined function or method ''ParforProgressStarter2'' for input arguments of type ''char''.')
           error('ParforProgressStarter2 not in path.');
       else
           % this should NEVER EVER happen.
           msg{1} = 'Unknown error while initializing "ParforProgressStarter2":';
           msg{2} = me.message;
           print_error_red(msg);
           % backup solution so that we can still continue.
           ppm.increment = nan(1, nbr_files);
       end
    end
    % % % % % % % % % % % % % % % %
    %}
    for x = 1:size(config.weights{weight_idx}, 1)
        for y = 1:size(config.weights{weight_idx}, 2)
            config.weights{weight_idx}(x, y) = config.weights{weight_idx}(x, y) + epsilon;
            op_train_pipe(image, label);
            cost1 = config.cost;
            config.weights{weight_idx}(x, y) = config.weights{weight_idx}(x, y)  - (2*epsilon);
            op_train_pipe(image, label);
            cost2 = config.cost;
            config.weights{weight_idx}(x, y) = config.weights{weight_idx}(x, y) + epsilon;
            
            % compute the numerical 
            temp = (cost1 - cost2) / (2 * epsilon);
            numgrad(x, y) = gather(temp);

            diff = norm(temp-estimatedGrad{weight_idx}(x, y));
            threshold = 10^-10;
            %if(diff > threshold)
                fprintf('numerical: %i, ', temp);
                fprintf('estimated: %i. \n', estimatedGrad{weight_idx}(x, y));
                fprintf('loop (%d, %d)\n', x, y);
                fprintf('diff too large! diff: %i.\n', diff);
            %end
            %ppm.increment((x-1)*size(config.weights.W2, 1)+y);
        end
    end
    
    try % use try / catch here, since delete(struct) will raise an error.
       delete(ppm);
    catch me %#ok<NASGU>
    end
end



