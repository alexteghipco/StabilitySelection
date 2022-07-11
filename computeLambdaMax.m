function [lambdaMax,nullMSE,X,y,muX,sigmaX,muY]=computeLambdaMax(X,y,weights,alpha,withConstant,standardize)
    constantPredictors = (range(X)==0);
    [X,y,muX,sigmaX,muY] = standardizeXY(X, y, weights, withConstant, standardize, constantPredictors);

    observationWeights = ~isempty(weights);
    N = size(X,1);

    % Calculate max lambda that permits non-zero coefficients
    if ~observationWeights
        dotp = abs(X' * y);
        lambdaMax = max(dotp) / (N*alpha);
    else
        wX0 = X .* weights';
        dotp = abs(sum(wX0 .* y));
        lambdaMax = max(dotp) / alpha;
    end
    
    if ~observationWeights
        nullMSE = mean(y.^2);
    else
        % This works because weights are normalized and Y0 is already
        % weight-centered.
        nullMSE = weights * (y.^2);
    end
    
end %-computeLambdaMax

function [X0,Y0,muX,sigmaX,muY] = standardizeXY(X, Y, weights, withConstant, standardize, constantPredictors)
    
    observationWeights = ~isempty(weights);
    
    if withConstant
        if standardize
            if observationWeights
                % withConstant, standardize, observationWeights
                muX=weights*X;
                muY=weights*Y;
                sigmaX=sqrt(weights*((X-muX).^2));
                sigmaX(constantPredictors) = 1;
            else % ~observationWeights
                % withConstant, standardize, ~observationWeights
                muX=mean(X);
                muY=mean(Y);
                sigmaX=std(X,1);
                sigmaX(constantPredictors) = 1;
            end
        else % ~standardize
            if observationWeights
                % withConstant, ~standardize, observationWeights
                muX=weights*X;
                muY=weights*Y;
                sigmaX=1;
            else % ~observationWeights
                % withConstant, ~standardize, ~observationWeights
                muX=mean(X);
                muY=mean(Y);
                sigmaX=1;
            end
        end
    else % ~withConstant
        muX=zeros(1,size(X,2));
        muY=0;
        sigmaX=1;
    end
    X0 = (X-muX)./sigmaX;
    Y0 = Y-muY;
    
end %-standardizeXY