function [Ws, DoFs] = group_lasso_basic(X, Y, lambdas)
    [n, p] = size(X);
    q = size(Y, 2);  % Number of tasks
    W_init = zeros(p, q);
    m = length(lambdas);  % Number of lambda values
    
    Ws = zeros(p, q, m);  % 3D array to store W for each lambda
    DoFs = zeros(1, m);  % Array to store DoF for each lambda
        
    for i = 1:m
        lambda = lambdas(i);
        
        % Define the objective function and gradient
        obj_func = @(W) 0.5/n * sum(sum((Y - X * W).^2)) + lambda * sum(sqrt(sum(W.^2)));
        grad_func = @(W) -1/n * X' * (Y - X * W);
        
        % Optimize with gradient descent
        W = W_init;
        learning_rate = 0.01;  % Learning rate for gradient descent
        max_iter = 1000;  % Maximum number of iterations
        tol = 1e-6;  % Convergence tolerance
        obj_val_old = obj_func(W);
        for t = 1:max_iter
            W = W - learning_rate * grad_func(W);
            obj_val_new = obj_func(W);
            if abs(obj_val_new - obj_val_old) < tol
                break;
            end
            obj_val_old = obj_func(W);
        end
        
        Ws(:, :, i) = W;  % Store the W matrix for this lambda
        DoFs(i) = sum(sqrt(sum(W.^2, 2)) > tol);  % Compute DoF for this lambda
        W_init = W;  % Update W_init for the next iteration
    end
end
