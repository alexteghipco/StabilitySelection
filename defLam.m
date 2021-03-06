function [lmxo,lmno,lms] = defLam(X,y,alpha,stnd,lmx,lmn,lamRatio,lst,ln,logPref)
%logPref = 'smaller';
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

if ~isempty(X) && ~isempty(y) % if X and y are not empty we assume the passed in lmn and lmx are just going to be used to create a lambda series...
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
else
    lmno = lmn;
    lmxo = lmx;
end

% if min(lmno) == 0
%     if ~strcmpi(lst,'linear')
%         warning('You have 0 lambda value so we will change our spacing from log to linear...')
%         lst = 'linear';
%     end
% end

lms = [];
if strcmpi(lst,'linear')
    lms = linspace(min(lmno),max(lmxo),ln);
elseif strcmpi(lst,'log')
    if strcmpi(logPref,'smaller')
        if lmno ~= 0
            lms = exp(linspace(log(min(lmno)),log(max(lmxo)),ln));
        else
           lms = exp(linspace(log(1e-100),log(max(lmxo)),ln));
        end
    elseif strcmpi(logPref,'larger')
        lms = exp(linspace(log(min(lmxo)),log(max(lmno)),ln));
        lms = fliplr(lms);
    end
end