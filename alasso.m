function [X,w] = alasso(X,y,alph,stnd,lam,gam)
regp = 'largest'; % could also be smallest, largest, or middle
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

% find regularization penalty that shrinks the least #
% of features to zero
mv = max(fitin.DF);
if strcmpi(regp,'smallest') % there could be a range of values that satisfy this criterion so we can get the largest lamda that does this, the smallest lambda that does this, or the middle lambda in the set.
    id = find(fitin.DF == mv,1,'first'); % smallest regularization penalty that shrinks least number of features to zero
elseif strcmpi(regp,'largest')
    id = find(fitin.DF == mv,1,'last'); % largest regularization penalty that shrinks least number of features to zero
elseif strcmpi(regp,'middle') % middle regularization penalty that shrinks least number of features to zero
    id1 = find(fitin.DF == mv,1,'first');
    id2 = find(fitin.DF == mv,1,'last');
    id = round((id1+id2)/2);
end

w = bin(:,id);
w = 1./abs(w).^(gam); % gam *can* vary but see original adaptive EN paper
id = find(w == inf);
w(id) = 1e-50;
X = X.*w';