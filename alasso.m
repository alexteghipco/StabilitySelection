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
    if strcmpi(regp,'smallest') % there could be a range of values that satisfy this criterion so we can get the largest lamda that does this, the smallest lambda that does this, or the middle lambda in the set.
        id = find(fitin.DF == mv,1,'first'); % smallest regularization penalty that shrinks least number of features to zero
    elseif strcmpi(regp,'largest')
        id = find(fitin.DF == mv,1,'last'); % largest regularization penalty that shrinks least number of features to zero
    elseif strcmpi(regp,'middle') % middle regularization penalty that shrinks least number of features to zero
        id1 = find(fitin.DF == mv,1,'first');
        id2 = find(fitin.DF == mv,1,'last');
        id = round((id1+id2)/2);
    end
else
    if ~parallel
        for jj = 1:length(id)
            yh = Xtest * bin(:,id(jj)) + fitin.Intercept(id(jj));
            mse(jj) = mean((y(tmpid)-yh).^2);
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
w = 1./abs(w).^(gam); % gam *can* vary but see original adaptive EN paper
id = find(w == inf);
w(id) = 1e-50;
X = X.*w';