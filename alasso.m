function [X,w] = alasso(X,y,alph,stnd,lam,gam)
if isempty(alph)
    alph = 0.00001;
end
if isempty(gam)
    gam = 1;
end
if isempty(stnd)
    stnd = true;
end
if ~isempty(lam)
    [bin,fitin] = lasso(X,y,'Lambda',lam,'Alpha',alph,'Standardize',stnd);
else
    [bin,fitin] = lasso(X,y,'Alpha',alph,'Standardize',stnd);
end

% find greatest regularization penalty that shrinks the least #
% of features to zero
mv = max(fitin.DF);
id = find(fitin.DF == mv,1,'last');
w = bin(:,id);
w = 1./abs(w).^(gam); % gam *can* vary but see original adaptive EN paper
id = find(w == inf);
w(id) = 1e-50;
X = X.*w';