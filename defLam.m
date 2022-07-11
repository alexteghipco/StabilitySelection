function [lmxo,lmno,lms] = defLam(X,y,alpha,stnd,lmx,lmn,lamRatio,lst,ln)
if isempty(stnd)
    stnd = true;
end
if isempty(lamRatio)
    lamRatio = 1e-4;
end
if isempty(lst)
    lst = 'log';
end
if isempty(ln)
    ln = 100;
end

for i = 1:length(alpha)
    if isempty(lmx) % compute max lambda if not specified by user...
        [lmxo(i,1),nullMSE,~,~,~,~,~] = computeLambdaMax(X,y,[],alpha(i),true,stnd);
    else
        lmxo(i,1) = lmx;
    end
    if isempty(lmn) % compute min lambda if not specified by user...
        lmno(i,1) = lmxo(i,1) * lamRatio;
    else
        lmno(i,1) = lmn;
    end
end
lms = [];
if strcmpi(lst,'linear')
    lms = linspace(min(lmno),max(lmxo),ln);
elseif strcmpi(lst,'log')
    lms = exp(linspace(log(min(lmno)),log(max(lmxo)),ln));
end