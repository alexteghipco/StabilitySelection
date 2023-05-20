function [X,w,bin,fitin] = alasso(X,y,Xtest,ytest,alph,stnd,lam,gam,regp,parallel)
%regp = 'largest'; % could also be smallest, largest, or middle
if isempty(alph)
    alph = 0.001; % was 0.01
end
if isempty(gam)
    gam = 1;
end
if isempty(stnd)
    stnd = true;
end
if isempty(lam)
    [~,~,lam] = defLam(X,y,alph,stnd,[],0,[],'log',5000,'smaller');
end

[bin,fitin] = lasso(X,y,'Lambda',lam,'Alpha',alph,'Standardize',stnd);

% find regularization penalty that shrinks the least #
% of features to zero
id = find(fitin.DF ~= max(fitin.DF));
if isempty(Xtest) && isempty(ytest)
    if strcmpi(regp,'smallest') 
        id = find(fitin.DF == mv,1,'first');
    elseif strcmpi(regp,'largest')
        id = find(fitin.DF == mv,1,'last');
    elseif strcmpi(regp,'middle') 
        id1 = find(fitin.DF == mv,1,'first');
        id2 = find(fitin.DF == mv,1,'last');
        id = round((id1+id2)/2);
    end
else
    if ~parallel
        for jj = 1:length(id)
            yh = Xtest * bin(:,id(jj)) + fitin.Intercept(id(jj));
            mse(jj) = mean((ytest-yh).^2);
        end
    else
        parfor jj = 1:length(id)
            yh = Xtest * bin(:,id(jj)) + fitin.Intercept(id(jj));
            mse(jj) = mean((ytest-yh).^2);
        end
    end
    [~,mi] = min(mse);
    id = id(mi);
end

w = bin(:,id);
% Exclude zero weights
w(w == 0) = [];
X(:, w == 0) = [];

w = 1./abs(w).^(gam);
id = find(w == inf);
w(id) = 1e-50;

% Apply weights to columns
X = X .* w';
