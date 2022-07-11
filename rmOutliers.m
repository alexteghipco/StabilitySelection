function [X,oid,mdl,pcao] = rmOutliers(X,varargin)

% load in defaults
options = struct('parallel',true,'rep',1,'repThresh',0.8,'prepro','none','pcaP',0.05,'pcaPermN',5000,'stnd',true,'nonconsec',true,'propOutliers',0.1,'outlierMethod','median','outlierDir','row','outlierThresh',3);
optionNames = fieldnames(options);

% now parse the user-specified arguments and overwrite the defaults
vleft = varargin(1:end);
for pair = reshape(vleft,2,[]) %pair is {propName;propValue}
    inpName = pair{1}; % make case insensitive by using lower() here but this can be buggy
    if any(strcmpi(inpName,optionNames)) % check if arg pair matches default
        def = options.(inpName); % default argument
        if ~isempty(pair{2}) % if passed in argument isn't empty, then write that in as the option
            options.(inpName) = pair{2};
        else
            options.(inpName) = def; % otherwise use the default values for the option
        end
    else
        error('%s is not a valid argument',inpName)
    end
end

% check compatibility
v = version('-release');
if str2double(v(1:end-1)) < 2017 && (strcmpi(options.outlierMethod,'ogk') || strcmpi(options.outlierMethod,'olivehawkins') || strcmpi(options.outlierMethod,'fmcd'))
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2017...please install a more recent version of matlab (2017+) to use robustcov for outlier detection'])
end
if str2double(v(1:end-1)) < 2022 && strcmpi(options.outlierMethod,'iforest')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2022...please install a more recent version of matlab (2022+) to use iforest for outlier detection'])
end
if str2double(v(1:end-1)) < 2015 && strcmpi(options.outlierMethod,'ocsvm')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2015...please install a more recent version of matlab (2015+) to use ocsvm for outlier detection'])
end

if strcmpi(options.outlierDir,'column')
    tmpX = X';
else
    tmpX = X;
end

if strcmpi(options.prepro,'pca')
    [pTvaf,pExp,tvafPerm,tvaf,vaf,vafs,vafp,vafpPerm,vafPerm,explainedPermAll,coeff,score,latent,tsquared,explained,mu] = permutedPCA(X,options.pcaPermN,'whole model');
    id = find(pExp < options.pcaP);
    if ~isempty(id)
        if options.nonconsec
            tmpl = [diff(id')~=1,true];
            tmplid = find(tmpl == 1);
            id = 1:tmplid(1);
        end
    else
        warning('No components survived correction during PCA...using first component only')
        id = 1;
    end
    tmpX = score(:,id);
    pcao.pTvaf = pTvaf;
    pcao.pExp = pExp;
    pcao.tvafPerm = tvafPerm;
    pcao.tvaf = tvaf;
    pcao.vaf = vaf;
    pcao.vafs = vafs;
    pcao.vafp = vafp;
    pcao.vafpPerm = vafpPerm;
    pcao.vafPerm = vafPerm;
    pcao.explainedPermAll = explainedPermAll;
    pcao.coeff = coeff;
    pcao.score = score;
    pcao.latent = latent;
    pcao.tsquared = tsquared;
    pcao.explained = explained;
    pcao.keptComponentIds = id;
else
    pcao = [];
end

%mdl = [];
if options.parallel
    parfor i = 1:options.rep
        [mdl{i},oidtmp{i}] = getOutls(tmpX,options);
    end
else
    for i = 1:options.rep
        [mdl{i},oidtmp{i}] = getOutls(tmpX,options);
    end
end

ss = zeros(size(X,1),1);
for i = 1:options.rep
    ss(oidtmp{i}) = ss(oidtmp{i})+1;
end

oid = find(ss >= options.repThresh);

if strcmpi(options.outlierDir,'column')
    X(:,oid) = [];
    X = X';
else
    X(oid,:) = [];
end

function [mdl,oid] = getOutls(tmpX,options)
mdl = [];
if strcmpi(options.outlierMethod,'median') || strcmpi(options.outlierMethod,'mean') || strcmpi(options.outlierMethod,'quartiles') || strcmpi(options.outlierMethod,'grubbs') || strcmpi(options.outlierMethod,'gesd')
    TF = isoutlier(tmpX,options.outlierMethod,'ThresholdFactor',options.outlierThresh);
    tmpl = sum(TF,2);
elseif strcmpi(options.outlierMethod,'ogk') 
    [mdl.sig, mdl.mu, mdl.mah, tmpl, mdl.s] = robustcov(tmpX,'Method',options.outlierMethod,'OutlierFraction',options.propOutliers);
elseif strcmpi(options.outlierMethod,'olivehawkins') || strcmpi(options.outlierMethod,'fmcd')
    if options.propOutliers > 0.5
        options.propOutliers = 0.5;
        warning('Outlier proportion/fraction can be a maximum of 0.5 for fmcd and olivehawkins...fixing propOutliers to 0.5')
    end
    [mdl.sig, mdl.mu, mdl.mah, tmpl, mdl.s] = robustcov(tmpX,'Method',options.outlierMethod);
elseif strcmpi(options.outlierMethod,'iforest')
    [mdl,tmpl] = iforest(tmpX,ContaminationFraction=options.propOutliers);
elseif strcmpi(options.outlierMethod,'ocsvm')
    mdl = fitcsvm(tmpX,ones(size(tmpX,1),1), ...
        OutlierFraction=options.propOutliers, ...
        KernelScale="auto",Standardize=options.stnd);
    [~,s_OCSVM] = resubPredict(mdl);
    tmpl = s_OCSVM < 0;
end
oid = find(tmpl == 1);