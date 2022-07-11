function [fk,fsc,fscmx,maxVars,alpha,lam, scores, oid, ctr, mdl] = stabSel(X,y,varargin)
% Perform stability selection to identify a stable set of features with
% which you should build your *regression* model (update for classifcation
% in the works). Choose one of 13 feature selection algorithms, which will
% be used in the stability selection framework. You may also integrate
% outlier detection and removal (one of 4 methods) either prior to, or
% during, the subsampling procedure that stability selection relies on.
% There are many options for you to tinker with if you would like, but you
% you can leave these out of your call and stabSel will ensure the
% selection of reasonable options.
%
% Call: [fk,fsc,fscmx,maxVars,alpha,lam] = stabSel(X, y);
%
%% What do I need to provide stabSel:  -----------------------------------
% X: an n x p matrix of predictors
% y: an n x 1 vector of responses
%
% Optional arguments can be supplied as well, like so: 
% stabSel(X,y,'maxVars',25,'stnd',false)
%
%% What does stability selection do?  ------------------------------------
% Subsamples of your data are taken, and a feature selection algorithm is
% run on each subsample. The features that are selected over subsamples are
% counted up and a threshold is applied to identify a 'stable set' of
% features. The critical parameters in stability selection are: i) the
% number of features you force your feature selection algorithm to select
% within each subsample, ii) the proportion of subsamples that the feature
% must be selected within in order to enter the stable set, and iii) the
% proportion of the data that you are subsampling. If we hold parameters
% (ii) and (iii) fixed, we can compute the # of features our selection
% algorithm should choose to maintain a FDR or a FWER < 0.05. This is done
% for you by default in stabSel. Note also, however, that in cases where
% data samples are low, you can increase your subsampling proportion. For
% method and details, see: Meinshausen, N., & Bühlmann, P. (2010).
% Stability selection. Journal of the Royal Statistical Society: Series B
% (Statistical Methodology), 72(4), 417-473.
%
% Note that if you are using stabSel to select features for a model,
% consider ensuring that the data used for feature selection and model
% validation and/or tuning is independent. This will give more accurate
% prformance estimates for the model.
%
%% General optional arguments ---------------------------------------------
%
% 'maxVars' : # of variables the selection algorithm is forced to choose on each
% subsample. If empty or left out, this will be calculated automatically to
% ensure FDR or FWER correction (p < 0.05). Default: empty (i.e., []).
%
% 'corr' : 'FDR' (default) or 'FWER' to ensure the type of correction at p
% < 0.05. Note, stabSel will warn you if you undermine the conditions
% required to guarantee correction. Default: 'FDR'.
%
% 'rep' : number of subsamples to draw (default: 200). Default: 200.
%
% 'prop' : proportion of data to subsample (i.e., 0.7 means
% 70% of the data will be used in each subsample; default: 0.5 but you may
% increase this number if dataset is small). Default: 0.5.
%
% 'propN' : If you would like to specify the number of samples to use in
% each subsample (rather than the proportion of the data), set 'prop' to be
% greater than or equal to 1 and propN to true. Default: false.
%
% 'adjProp' : if you are doing outlier detection, by default, the
% proportion of data used in each subsample is cacluated *after* outlier
% removal. But, you can set adjProp to true in order to calculate the
% proportion before outlier removal and ensure that this number of samples
% is taken *after* outlier removal. Default: false.
%
% 'thresh' : threshold for proportion of subsamples a feature must appear
% in to enter the stable set. The default is 0.9 which is used for the
% FDR/FWER computation so if you change this, it may break FDR/FWER AND
% maxVars calculation. However, there are cases where this is reasonable.
% See Meinshausen & Bulhmann (2010). If threshold is > 1 we assume you are
% telling stabSel how many features to select. Default: 0.9.
%
% 'selAlgo' : sets the algorithm for selecting features. This may be 'EN'
% for an elastic net, 'lasso' for lasso regression, 'ridge for ridge
% regression, 'LR' for correlation, 'robustLR' for robust linear regression
% (not as sensitive to outliers but takes MUCH longer to run), 'NCA' for
% neighborhood components analysis, 'mrmr' for minimum redundance maximum
% relevance algorithm which relies on mutual information, 'ftest' for an
% ftest, 'releiff' for the relief algorithm that relies on nearest neighbor
% search, 'GPR' for a gaussian process model, and 'RF' for a random forest.
% Note, that an additional argument can be used to adjust 'EN' or 'lasso'
% to be adaptive (in both cases, coefficients from ridge regression are
% used to weight your data before applying lasso or an elastic net. For
% more information, see: Zou, H., & Zhang, H. H. (2009). On the adaptive
% elastic-net with a diverging number of parameters. Annals of statistics,
% 37(4), 1733.). Note, some of these options require more recent versions
% of matlab, but stabSel will check this for you. Default: 'EN'.
%
% 'stnd' : standardize data. This only applies when selAlgo is 'EN',
% 'lasso', 'ridge', or 'GPR'. It also determines if standardization will be used
% with some outlier detection methods (i.e., robustcov and ocsvm).
%
% 'parallel' : set to true to attempt parallel computing where possible.
% Applies to scripts stabSel draws on as well. Default: false.
%
% 'verbose' : set to true to get feedback in the command window about what
% stabSel is doing. Default: false.
%
% ------------ Optional arguments for > 1 selection algorithm -------------
% -------------------- (elastic net, lasso, ridge, nca, gpr) -------------------
%
% 'lmn' : set the min. lambda value for elastic net, lasso, ridge, and
% neighborhood components analysis. Leave blank to automatically compute
% lambda values for elastic net, lasso and ridge using the max l trick (see
% below). If automatically computed, 'lamRatio' will be used to determine
% the lmn (see below). For NCA, default lmn will be 0.0001 but this value
% will be adjusted based on the standard deviation of your predictors.
% Default: [].
%
% 'lmx' : max lambda value to use in sequence. This will override the max l
% trick for elastic net, lasso and ridge (see lamENInside section below for
% more detail). Default: [].
%
% 'ln' : number of lambda values in sequence (i.e., between user-supplied
% lmx and lmn OR automatically determined lmn and lmx). This applies to
% elastic net, lasso, ridge, NCA, and GPR.
%
% 'lst' : determines how values will be spaced between lmn and lmx to
% produce lambda sequence. Set to 'linear' or 'log'. Default: 'linear'.
% Consider using log for en, lasso, ridge (argument also applies to GPR and
% NCA).
%
% ------------ Optional arguments for elastic net, lasso, ridge -----------
%
% 'adaptive' : sets EN or lasso to be adaptive. Lasso lacks oracle
% properties, may have 'inaccurate' weights, or inconsisttently select
% variables (e.g., noise variables which is especially true when n >> p;
% see Zou & Zhang, 2009). Adaptive lasso has oracle properties but lacks
% the ability to select *all* of the correlated variables that are
% predictive. The L2 regualrization in EN fixes this, but EN lacks oracle
% properties as well so ideally we would use adaptive EN. In practice,
% adaptive EN does not always work flawlessly. This is in part because
% adaptive EN introduces a new parameter to tune: gamma. It is unclear how
% to best tune this parameter within stabSel so this is currently being
% worked out. However, some a prior set gamma value may work quite well for
% your data. Default: false.
%
% 'adaptOutside' : determines if you are weighting data (as in adaptive EN,
% lasso, ridge) BEFORE (set to false) or DURING (set to true) subsampling.
% Default: false.
%
% 'gam' : gamma value for adapting EN or lasso. This effectively controls
% the degree to which your data is weighted by a first-run of ridge
% regression. Default: 1.
%
% 'alpha' : alpha values for determining weight of L1 vs L2 optimization in
% elastic net. This can be an n x 1 vector. Default: [0.1:0.1:1].
%
% 'lamEN' : lambda values to be used for elastic net, lasso, or ridge
% regression. By default, lambdas are computed automatically using the 'max
% lambda' trick. We can perform simple matrix multiplication using your
% input data to detrmine the max lambda that will produce exactly 1
% non-zero coefficient. Note, this typically works better when your data is
% standardized. Lambda values will be automatically generated this way for
% all alpha values, then the min and max lambda computed over this set of
% sequences is used to generate a single lambda sequence that is applied.
% Default: [].
%
% 'lamENInside' : determines if automatically defined lambda values will be
% computed BEFORE subsampling or DURING subsampling. Set to true to compute
% them DURING subsampling. That means each subsample will have a different
% set of lambdas. There are cases where this may make sense to do (i.e., if
% one sequence of lambdas does not fit your data well; stabSel will warn
% you when the lambdas are a VERY poor fit but not otherwise so inspect the
% results of stabSel yourself to determine this). Default: false.
%
% 'lamAEN' : if you pass in lambdas through lamEN, these will apply to the
% first-run of ridge regression that is used to weight your data. But if
% you are doing adaptive EN or lasso, you need lambda values for *AFTER*
% this weighting. You can supply them here or leave blank to compute them
% using the max l trick. Default: [].
%
% 'lamRatio' : to get a sequence of lambda values, we use the max l trick
% to get the max lambda, then use this ratio to get the smallest lambda.
% Default: 1e-4. Default: [].
%
% --------------------- Optional arguments for nca ------------------------
%
% 'lamNCA', : lambda values to use for NCA. Default values will be set
% between 0.0001 to 100 before weighting by standard deviation of response
% variable. Default: [].
%
% ------------ Optional arguments for linear regression -------------------
%
% 'LRtype', : Determines how linear regression and/or correlation is used
% to select features. We can either use the pvalue if set to 'pval' or the
% top maxVars features if set to 'filter'. Default: 'pval'.
%
% 'lrp', : p-value to use for selecting features (only applies if LRType is
% pval. Default: 0.05.
%
% ------------ Optional arguments for relieff -----------------------------
%
% 'rK' : determines k in knn (# of neighbors). Default: [1:14
% 15:10:size(X,2)/2].
%
% 'rE' : sigma, or distance scaling factor. Consider changing this to just
% 50 if stabSel is taking too long on your data. Default: [10:15:200].
%
% ------------ Optional arguments for GPR ---------------------------------
%
% 'lamGPR' : determines regularization standard deviation. Default:
% 1e-2*std(y). Note, GPR will use: ardsquaredexponential kernel function,
% lbfgs optimizer, fully independent conditional approximation for
% prediction, and subset of regressors approximation as the fit method.
% Note, you currently cannot set lmn and lmx for lamGPR.
%
% ------------------- Optional arguments for RF ---------------------------
%
% 'lamRF' : this determines the number of learning cycles to use for the
% random forest. Can be an n x 1 sequence. More options for RF are
% available but not currently implemented due to computational costs
% required to execute. To make them available, uncomment line 481 and
% comment out line 480. You can then pass in a learning rate using optional
% argument 'lr'. Max. # splits using optional argument 'mns' and min leaf
% size using optional argument 'mls'. In these 3 cases, you should pass in
% a sequence of values that is n x 1 in size. Default values for lamRF are
% 10 linearly spaced values between 10 and 500. Default values for min leaf
% size are round(linspace(1,max(2,floor(size(X,2)/2)),15)). Default values
% for max # splits are round(linspace(1,size(X,2)-1,15)). Default values
% for learning rate are linspace(1e-3,1,10). Default: [].
%
% ---------- Optional arguments for outlier detection/removal -------------
%
% 'outlier' : determines where outlier detection removal will be performed
% on matrix X. Set to 'outside' to perform outlier detection before
% subsampling and 'inside' to perform outlier detection inside the
% subsampling loop. Set to 'none to avoid detection/removing outliers.
% Default: 'none'.
%
% 'outlierPrepro' : currently can only be set to 'none' or 'pca'. In the
% latter case, PCA is performed first, before outlier detection. This is
% useful because some methods of outlier detection do not work with lots of
% features. Default: none.
%
% 'outlierPreproP' : if doing PCA, a permutation analysis will be performed
% to determine the number of components to keep. This sets the p-value for
% the components that will be retained. Default: 0.05.
%
% 'outlierPreproN' : # of permutations over which a p-value will be
% computed if doing PCA. Default: 5000.
%
% 'outlierPreproNonconsec' : if doing PCA, in some cases, significant
% components may not be consectuive (i.e., a component that explains very
% little variance may be significant). To keep only the first consecutive
% set of components that explain significant variance (organized by
% descending % of variance explained) set this to true. Default: true.
%
% 'outlierDir' : specify whether you want to find outliers in the rows or
% columns using 'rows' or 'columns'. If you set this to columns we assume
% you want to find outlier features. Default: 'rows'.
%
% 'outlierMethod' : specify method for outlier detection. Set to 'median',
% 'mean', 'quartiles', 'grubbs', 'gesd' to use isoutlier.m, which will
% perofrm outlier detection using some scaled MAD from the median, some
% std devs from the mean, some IQ ranges above/below upper lower quartiles,
% using the grubb's test, or using a generalized extreme studentized
% deviate test for outliers. Set this to 'fmcd, 'ogk', or 'olivehawkins' to
% use different methods for estimating robust covariance to detect outliers
% (robustcov.m). Use 'iforest' to generate an isolation forest to identify
% outliers. Use 'ocsvm' to use one class svm to identify outliers. Default:
% median.
% 
% 'outlierReps' : some outlier detection methods are stochastic and
% repeating them can be helpful. This determines the number of times that
% they will be repeated. Default: 1 (i.e., no repeats).
%
% 'repThresh' : a threshold that determines how often outliers need to be
% identified across repeats to be kept as outliers. Default: 0.8 (i.e., 80%
% of the time).
%
% 'propOutliers' : some outlier detection methods force you to select a
% proportion of the data you expect to be an outlier. This is the case for
% iforest, ocsvm, fmcd, and olivehawkins. Default: 0.1.
%
% 'outlierThresh' : some outlier detection methods require a threshold
% factor that varies in meaning depending on the method used but determines
% the # of outliers identified. For 'median' it refers to the number of
% scaled MAD. For 'mean' it refers to the number of std devs from the mean.
% For 'grubbs' and 'gesd' it ranges from 0 - 1 and determines the detection
% threshold. For 'quartiles' it refers to the number of interquartile
% ranges. Default: 3 (i.e., good for 'median', the default for
% outlierMethod).
%
%% Outputs ----------------------------------------------------------------
% fk : stable set of features (i.e., features kept) fsc : empirical
% probabilities across regularization parameters that were used.
%
% fsmx : max empirical probability across regularization parameters.  Note, in
% stability selection we take the max proportion of times that a feature
% was selected ACROSS all regularization parameters.
%
% maxVars : # of variables/features selected in each subsample.
%
% alpha : alpha values used. Only applies to elastic net, lasso, ridge.
%
% lam : lambda values used. Only applies to elastic net, lasso, ridge, nca,
% GPR, RF.
%
% 'scores' : returns weighting of features for linear
% regression/correlation, relieff, mrmr, nca. 
%
% 'oid' : rows of X that are determined to be outliers.
%
% 'ctr' : indices of samples that were used on each subsample. 
%
% 'mdl' : this is a model that was trained to identify outliers. If
% outliers were detected inside the subsampling scheme, there is a model
% for each subsample. You can get a consensus of the predictions of these
% models on new data.
%
%% Internal notes----------------------------------------------------------
% This was originally designed around using the elastic net. Need to
% streamline arguments and outputs (e.g., scores can be returned as weights
% from other models, options should be renamed, need to select better
% defaults to test for RF, etc). 
% Alex Teghipco // alex.teghipco@sc.edu // 

% load in defaults
options = struct('maxVars',[],'propN',false,'adjProp',true,'alpha',[0.1:0.1:1],...
    'lamEN',[],'lamENInside',false,'lamAEN',[],'lamRatio',1e-4,'adaptOutside',...
    false,'lmn',[],'lmx',[],'ln',1000,'prop',0.5,'rep',200,'stnd',true,'corr',...
    'fdr','thresh',0.9,'parallel',false,'adaptive',false,...
    'outlier','none','outlierPrepro','none','outlierPreproNonconsec',true,...
    'propOutliers',0.1,'outlierPreproP',0.05,'outlierPreproN',5000,...
    'outlierReps',1,'outlierRepThresh',1,'outlierMethod','median',...
    'outlierDir','row','outlierThresh',3,'selAlgo','en','lst','linear',...
    'gam',1,'lrp',0.05,'rK',[1:14 15:10:size(X,2)/2],'rE',[10:15:200],...
    'lamNCA',[],'lamGPR',[],'lamRF',[],'lr',[],'mls',[],'mns',[],'verbose',true,...
    'LRtype','pval');
optionNames = fieldnames(options);
rng shuffle

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
if str2double(v(1:end-1)) < 2022 && strcmpi(options.selAlgo,'mrmr')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2022...please install a more recent version of matlab (2022+) to use mrmr for feature selection'])
end
if str2double(v(1:end-1)) < 2017 && strcmpi(options.selAlgo,'nca')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2017...please install a more recent version of matlab (2017+) to use NCA for feature selection'])
end
if str2double(v(1:end-1)) < 2011 && strcmpi(options.selAlgo,'relieff')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2011...please install a more recent version of matlab (2011+) to use relieff for feature selection'])
end
if str2double(v(1:end-1)) < 2012 && strcmpi(options.selAlgo,'en')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2012...please install a more recent version of matlab (2012+) to use elastic net, lasso, or ridge for feature selection'])
end

% fix lasso or ridge as inputs
if strcmpi(options.selAlgo,'lasso')
    options.alpha = 1;
    options.selAlgo = 'en';
end
if strcmpi(options.selAlgo,'ridge')
    options.alpha = 0.00001;
    options.selAlgo = 'en';
end

% proportion may reflect N of subsample instead of an N to compute
if options.propN && options.prop >= 1
    tmpl = options.prop;
    options.prop = options.prop/size(X,1);
    if options.verbose
       disp('Changing subsampling N to proportion...')
       disp(['To achieve N of ' num2str(tmp) ' in subsample, proportion must be: ' num2str(options.prop)]);
    end
end

% convert correction arg into number we'll use to perform actual
% correction
if ischar(options.corr)
    if strcmpi(options.corr,'FDR')
        options.corr = 1;
    end
    if strcmpi(options.corr,'FWER')
        options.corr = 0.5;
    end
end

% warn about correction
if options.verbose && (~isempty(options.maxVars) || options.thresh ~= 0.9 || options.prop ~= 0.5)
    disp('Feature selection is not guaranteed unless threshold for stable set is 0.9, subsampling proportion is 0.5, and maxVars are set to be automatically calculated (but note increasing subsampling proportion *may* be okay; see documentation)')
end

% estimate # variables we SHOULD force EN to select...
if isempty(options.maxVars)
    options.maxVars = round(sqrt(0.8*options.corr*size(X,2)));
    if options.verbose
        disp(['The number of variables you should force your selection algorithm to choose is: ' num2str(options.maxVars)])
    end
end

% check for redundant lambda args...
if ~isempty(options.lamEN) && options.lamENInside
    if options.verbose
        warning('You cannot pass in lambda values AND try to define lambda values inside the subsamples. Assuming turning on lamENInside was a mistake...setting this option to false.')
    end
end

% warn about features as outliers
if ~strcmpi(options.outlier,'none') && strcmpi(options.outlierDir,'columns')
    warning('Treating features as outliers has not been thoroughly debugged...proceed with caution')
end

% check to see if you want to do outlier removal now...
oid = [];
mdl = [];
if strcmpi(options.outlier,'outside')
    [X,oid,mdl,~] = rmOutliers(X,'prepro',options.outlierPrepro,'nonconsec',options.outlierPreproNonconsec,'outlierMethod',options.outlierMethod,'outlierDir',options.outlierDir,'outlierThresh',options.outlierThresh,'pcaP',options.outlierPreproP,'pcaPermN',options.outlierPreproN,'propOutliers',options.propOutliers,'rep',options.outlierReps,'repThresh',options.outlierRepThresh);
    % check if you want to adjust subsampling prop. to account for #
    % outliers removed...prop will be adjusted to keep training set same
    % size as if the prop selected was applied to all data...
    y(oid) = [];
    if options.verbose
        if strcmpi(options.outlierDir,'rows')
            disp(['Removed ' num2str(length(oid)) ' row (samples) outliers from data (' num2str((length(oid)/size(X,2))*100) '%) before subsampling'])
        elseif strcmpi(options.outlierDir,'collumns')
            disp(['Removed ' num2str(length(oid)) ' col (features) outliers from data (' num2str((length(oid)/size(X,2))*100) '%) before subsampling'])
        end
    end
    if options.adjProp
        if strcmpi(options.outlierDir,'rows') % no adjustment if outliers are features
            in = fix((size(X,1) + length(oid))*options.prop);
            options.prop = in/size(X,1);
            if options.verbose
                disp(['Adjusting subsampling proportion to ' num2str(options.prop) ' in order to retain ' num2str(in) ' samples in each subsample'])
            end
            if options.prop > 1
                options.prop = 1;
                if options.verbose
                    warning('Adjusting subsampling proportion after outlier removal results in subsampling sets that comprise the entire dataset')
                end
            end
        end
    end
end

% define lambdas for EN (use max l trick unless max/min specified by user)
if isempty(options.lamEN) && ~options.lamENInside
    if ~options.adaptive % if not adaptive
        if options.verbose
            disp('Computing lambda values for EN (lambda values are computed from the whole dataset and the same series will be passed in to each subsample)')
        end
        [~,~,options.lamEN] = defLam(X,y,options.alpha,options.stnd,options.lmx,options.lmn,options.lamRatio,options.lst,options.ln);
    else % if adaptive we need to run first pass EN on whole data, get weights and get lambdas from those weights
        if options.verbose
            disp(['Computing lambda values for adaptive EN (lambda values are computed from the whole dataset weighted by an adaptive EN and the same series will be passed in to each subsample)'])
        end
        [Xtmp,~] = alasso(X,y,[],options.stnd,options.lamAEN,options.gam);
        [~,~,options.lamEN] = defLam(Xtmp,y,options.alpha,options.stnd,options.lmx,options.lmn,options.lamRatio,options.lst,options.ln);
        if options.adaptOutside
            X = Xtmp;
            options.adaptive = false;
        end
    end
end

% define lambdas for NCA
if isempty(options.lamNCA) && strcmpi(options.selAlgo,'nca')
    if isempty(options.lmn)
        options.lmn = 0.001;
        if options.verbose
            disp(['The min lambda for NCA is set to a default of: ' num2str(options.lmn)])
        end
    end
    if isempty(options.lamNCA)
        options.lmx = 100;
        if options.verbose
            disp(['The max lambda for NCA is set to a default of: ' num2str(options.lmx)])
        end
    end
    if strcmpi(options.lst,'linear')
        options.lamNCA = linspace(options.lmn,options.lmx,options.ln)*std(y)/length(y);
    elseif strcmpi(options.lst,'log')
        options.lamNCA = exp(linspace(log(options.lmn),log(options.lmx),options.ln))*std(y)/length(y);
    end
end   

% define lambdas for GPR        
if isempty(options.lamGPR)
    if strcmpi(options.lst,'linear')
        options.lamGPR = linspace(1e-2*std(y)/3,1e-2*std(y)*3,options.ln);
        if options.verbose
            disp(['Lambdas are on linear scale...(we make ' num2str(options.ln) ' of them)'])
        end
    else
        options.lamGPR = exp(linspace(log(1e-2*std(y)/3),log(1e-2*std(y)*3),options.ln));
        if options.verbose
            disp(['Lambdas are on log scale...(we make ' num2str(options.ln) ' of them)'])
        end
    end
end

% define lots of vars for RF -- note we currently only use lamRF (number of
% learning cycles) because doing this much tuning is expensive...RF IS
% CURRENTLY EXPERIMENTAL
if strcmpi(options.selAlgo,'RF')
    warning('RF parameters are currently hard coded...its difficult to find reasonable set of parameters that are quick to evaluate (currently only evaluate learning cycles between 10 and 500 based on passed ln argument)')
    if isempty(options.lamRF)
        options.lamRF = linspace(10,500,options.ln);
    end
    if isempty(options.lr)
        options.lr = linspace(1e-3,1,10);
    end
    if isempty(options.mls)
        options.mls = round(linspace(1,max(2,floor(size(X,2)/2)),15));
    end
    if isempty(options.mns)
        options.mns = round(linspace(1,size(X,2)-1,15));
    end
end

% initialize outputs...occasionally these are undefined depending on user
% args
scores = []; 
lam = [];

% preallocate subsample indices...this is in case you want to get
% lamba min/max using the subsamples (i.e., we get one series of l but
% tailored to the data)
if options.verbose
    disp('Preallocating subsample indices')
end
for j = 1:options.rep
    if options.prop < 1
        c3 = cvpartition(size(X,1),'Holdout',options.prop);
        ctr{j} = find(test(c3)==1);
    elseif options.prop == 1
        ctr{j} = 1:length(y);
        if options.verbose && j == 1
            warning('Your subsamples comprise the entire dataset')
        end
    end
    if options.lamENInside
        if ~options.adaptive
            if options.verbose
                if j == 1
                    disp('Computing lambda values for EN (lambda values are computed separately for each subsample)')
                end
            end
            [~,~,tmpl(:,j)] = defLam(X(ctr{j},:),y(ctr{j}),options.alpha,options.stnd,options.lmx,options.lmn,options.lamRatio,options.lst,options.ln);
        else
            [Xtmp,~] = alasso(X(ctr{j},:),y(ctr{j}),[],options.stnd,options.lamAEN,options.gam);          
            [~,~,tmpl(:,j)] = defLam(Xtmp,y(ctr{j}),options.alpha,false,options.lmx,options.lmn,options.lamRatio,options.lst,options.ln);
            if options.verbose
                if j == 1
                    disp(['Computing lambda values for adaptive EN (lambda values are computed separately for each subsample after weighting dataset by an EN and we define one series of lambda values based on min/max lambda across all subsamples)'])
                end
            end
        end
    end
end
if options.lamENInside
    [~,~,options.lamEN] = defLam([],[],options.alpha,options.stnd,max(tmpl(:)),min(tmpl(:)),options.lamRatio,options.lst,options.ln);
end

% now we initialize a matrix for counting selected features across
% subsamples
if strcmpi(options.selAlgo,'EN')
    fsc = zeros(size(X,2),length(options.lamEN),length(options.alpha)); % features x lam x alpha
elseif strcmpi(options.selAlgo,'LR')
    fsc = zeros(size(X,2),1); % features
elseif strcmpi(options.selAlgo,'nca')
    fsc = zeros(size(X,2),length(options.lamNCA)); % features x lam
elseif strcmpi(options.selAlgo,'mrmr')
    fsc = zeros(size(X,2),1); % features
elseif strcmpi(options.selAlgo,'ftest')
    fsc = zeros(size(X,2),1); % features
elseif strcmpi(options.selAlgo,'relieff')
    fsc = zeros(size(X,2),length(options.rK),length(options.rE)); % features x knn x sigma
elseif strcmpi(options.selAlgo,'gpr')
    fsc = zeros(size(X,2),length(options.lamGPR)); % features x lam
elseif strcmpi(options.selAlgo,'rf')
    %fsc = zeros(size(X,2),length(options.lamRF),length(options.mls),length(options.mns)); % features x lamRF x mls x mns
    fsc = zeros(size(X,2),length(options.lamRF)); % features x lamRF x mls x mns
end
if options.verbose
    disp(['Allocated matrix for counting feature selection across algorithm parameter. Size is: ' num2str(size(fsc))])
end

% And finally we can start selecting features within each subsample
for i = 1:options.rep
    if options.verbose
        if i == 1
            disp(['Starting feature selection. Working on subsample ' num2str(i) ' of ' num2str(options.rep)])
        else
            disp(['Working on subsample ' num2str(i) ' of ' num2str(options.rep)])
        end
    end
    stnd = options.stnd; % sometimes we will have to overwrite standardization so copy in user selected option for first step...
    
    % get subsample indices...
    Xtmp = X(ctr{i},:);
    Ytmp = y(ctr{i});
    
    % remove outliers defined as either rows (samples) or cols
    % (features)
    %oid = 1:size(Xtmp,2);
    % remove outliers defined as either rows (samples) or cols
    % (features)
    if strcmpi(options.outlier,'inside')
        [Xtmp,oid{i},mdl{i},~] = rmOutliers(Xtmp,'prepro',options.outlierPrepro,'nonconsec',options.outlierPreproNonconsec,'outlierMethod',options.outlierMethod,'outlierDir',options.outlierDir,'outlierThresh',options.outlierThresh,'pcaP',options.outlierPreproP,'pcaPermN',options.outlierPreproN,'propOutliers',options.propOutliers,'rep',options.outlierReps,'repThresh',options.outlierRepThresh);
        if strcmpi(options.outlierDir,'rows')
            Ytmp(oid{i}) = [];
        end
        % check if you want to adjust subsampling prop. to account for #
        % outliers removed...prop will be adjusted to keep training set same
        % size as if the prop selected was applied to all data...
        if options.verbose
            if strcmpi(options.outlierDir,'rows')
                disp(['Removed ' num2str(length(oid{i})) ' row (samples) outliers from data (' num2str((length(oid{i})/size(Xtmp,2))*100) '%) before subsampling'])
            elseif strcmpi(options.outlierDir,'collumns')
                disp(['Removed ' num2str(length(oid{i})) ' col (features) outliers from data (' num2str((length(oid{i})/size(Xtmp,2))*100) '%) before subsampling'])
            end
        end
    end
    
    % do standardization if necessary
    if options.stnd && (~strcmpi(options.selAlgo,'en') && ~strcmpi(options.selAlgo,'lasso') && ~strcmpi(options.selAlgo,'ridge') && ~strcmpi(options.selAlgo,'rf') && ~strcmpi(options.selAlgo,'gpr'))
        Xtmp = bsxfun(@rdivide,bsxfun(@minus,Xtmp,mean(Xtmp,2)),std(Xtmp,0,2));
    end
    
    % start EN...
    if strcmpi(options.selAlgo,'EN')
       % copy in lambda values precalculated...
       lam = options.lamEN;
       
       % run EN to get weights for adaptive lasso
       if options.adaptive
          [Xtmp,~] = alasso(Xtmp,Ytmp,[],stnd,options.lamAEN,options.gam);          
          stnd = false; % fix stnd for lasso OF weights
       end
       
       % now do lasso for each alpha (within each we pass in our lams)
       for kk = 1:length(options.alpha)
           if options.parallel
               [lsB,lsFit] = lasso(Xtmp,Ytmp,'Lambda',lam,'Alpha',options.alpha(kk),'Standardize',stnd,'DFMax',options.maxVars,'Options',statset('UseParallel',true));
           else
               [lsB,lsFit] = lasso(Xtmp,Ytmp,'Lambda',lam,'Alpha',options.alpha(kk),'Standardize',stnd,'DFMax',options.maxVars);
           end
           % we need to adjust lam in cases where lasso output excludes lam
           % (i.e., because of dfmax)
           if ~isempty(lsFit.Lambda)
               adj = find(options.lamEN == lsFit.Lambda(1)) -1;
               if options.verbose
                   disp(['Percentage of lambdas that did not fit current run: ' num2str(adj/options.ln)])
               end
           else
               adj = 0;
               warning('No lambdas returned any elastic net coefficients in this run. It is difficult to select a series of lambda values that fit all of your data. Try defining lambda series *outside* the subsampling procedure (if you are not doing so already).')
           end
           
           % now count selected feats
           if strcmpi(options.outlier,'none') || (~strcmpi(options.outlier,'none') && strcmpi(options.outlierDir,'rows'))
               for jj = 1:size(lsB,2)
                   if ~isempty(lsB)
                       id = find(lsB(:,jj)~=0);
                       fsc(id,jj+adj,kk) = fsc(id,jj+adj,kk)+1;
                   end
               end
           else
               s = 1:size(X,2);
               s(oid{i}) = [];
               for jj = 1:size(lsB,2)
                   if ~isempty(lsB)
                       id = find(lsB(:,jj)~=0);
                       fsc(s(id),jj+adj,kk) = fsc(s(id),jj+adj,kk)+1;
                   end
               end
           end
       end
    end
    
    % start linear regression...
    if strcmpi(options.selAlgo,'LR')
        [rV,pV] = corr(Xtmp,Ytmp,'rows','pairwise');
        if strcmpi(options.LRtype,'pval')
            id = find(pV <= options.lrp);
        elseif strcmpi(options.LRtype,'filter')
            [~,idtmp] = sort(rV,'descend');
            id = idtmp(1:options.maxVars);
        end
        fsc(id) = fsc(id)+1;
    end

    if strcmpi(options.selAlgo,'robustLR')
        if strcmpi(options.LRtype,'pval')
            id = [];
            if ~options.parallel
                for kk = 1:size(Xtmp,2)
                    lm = fitlm(Xtmp(:,kk),Ytmp,'RobustOpts','on');
                    if lm.Coefficients.pValue(2) <= options.lrp
                        id = [id kk];
                    end
                end
            else
                parfor kk = 1:size(Xtmp,2)
                    lm = fitlm(Xtmp(:,kk),Ytmp,'RobustOpts','on');
                    if lm.Coefficients.pValue(2) <= options.lrp
                        id = [id kk];
                    end
                end
            end
        elseif strcmpi(options.LRtype,'filter')
            if ~options.parallel
                for kk = 1:size(Xtmp,2)
                    lm = fitlm(Xtmp(:,kk),Ytmp,'RobustOpts','on');
                    scores(kk,1) = lm.Coefficients.Estimate(2);
                end
            else
                parfor kk = 1:size(Xtmp,2)
                    lm = fitlm(Xtmp(:,kk),Ytmp,'RobustOpts','on');
                    scores(kk,1) = lm.Coefficients.Estimate(2);
                end
            end
            [~,idtmp] = sort(scores,'descend');
            id = idtmp(1:options.maxVars);
        end
        fsc(id) = fsc(id)+1;
    end
    
    % start mrmr...
    if strcmpi(options.selAlgo,'mrmr')
        [id,scores{i}] = fsrmrmr(Xtmp,Ytmp);
        fsc(id(1:options.maxVars)) = fsc(id(1:options.maxVars))+1;
    end
    
    % start ftest...
    if strcmpi(options.selAlgo,'ftest')
        [id,scores{i}] = fsrftest(Xtmp,Ytmp);
        fsc(id(1:options.maxVars)) = fsc(id(1:options.maxVars))+1;
    end
    
    % start relieff...
    if strcmpi(options.selAlgo,'relieff')
        id = cell(length(options.rK),length(options.rE));
        for jj = 1:length(options.rK)
            %disp(num2str(jj))
            if ~options.parallel
                for kk = 1:length(options.rE)
                    [id{jj,kk},scores{jj,kk}] = relieff(Xtmp,Ytmp,options.rK(jj),'sigma',options.rE(kk),'method','regression');
                    %fsc(id(1:options.maxVars),jj,kk) = fsc(id(1:options.maxVars),jj,kk)+1;
                end
            else
                scores = cell(length(options.rK),length(options.rE));
                parfor kk = 1:length(options.rE)
                    [id{jj,kk},scores{jj,kk}] = relieff(Xtmp,Ytmp,options.rK(jj),'sigma',options.rE(kk),'method','regression');
                    %fsc(id(1:options.maxVars),jj,kk) = fsc(id(1:options.maxVars),jj,kk)+1;
                end
            end
        end
        for jj = 1:length(options.rK)
            for kk = 1:length(options.rE)
                fsc(id{jj,kk}(1:options.maxVars),jj,kk) = fsc(id{jj,kk}(1:options.maxVars),jj,kk)+1;
            end
        end
    end
    
    % start nca...
    if strcmpi(options.selAlgo,'nca')
        lam = options.lamNCA;
        if ~options.parallel
            for jj = 1:length(lam)
                nca = fsrnca(Xtmp,Ytmp,'FitMethod','exact', ...
                    'Solver','minibatch-lbfgs','Lambda',lam(jj), ...
                    'GradientTolerance',1e-4,'IterationLimit',200,'Standardize',stnd);
                [~,id{jj}] = sort(nca.FeatureWeights,'descend');
                scores{jj} = nca.FeatureWeights;
            end
        else
            id = cell(length(lam));
            scores = cell(length(lam));
            parfor jj = 1:length(lam)
                nca = fsrnca(Xtmp,Ytmp,'FitMethod','exact', ...
                    'Solver','minibatch-lbfgs','Lambda',lam(jj), ...
                    'GradientTolerance',1e-4,'IterationLimit',200,'Standardize',stnd);
                [~,id{jj}] = sort(nca.FeatureWeights,'descend');
                scores{jj} = nca.FeatureWeights;
            end
        end
        for jj = 1:length(lam)
            fsc(id{jj}(1:options.maxVars),jj) = fsc(id{jj}(1:options.maxVars),jj)+1;
        end       
    end
    
    % start gpr...
    if strcmpi(options.selAlgo,'gpr')
        lam = options.lamGPR;
        for jj = 1:length(lam)
            mdl = fitrgp(Xtmp,Ytmp,'KernelFunction','ardsquaredexponential', ...
                'Optimizer','lbfgs','FitMethod','sr','PredictMethod','fic','Standardize',stnd,'Regularization',lam(jj));
            sl = mdl.KernelInformation.KernelParameters(1:end-1);
            weights = exp(-sl); % Predictor weights
            scores{jj} = weights/sum(weights); % Normalized predictor weights
            [~,id{jj}] = sort(scores{jj},'descend');
        end
        for jj = 1:length(lam)
            fsc(id{jj}(1:options.maxVars),jj) = fsc(id{jj}(1:options.maxVars),jj)+1;
        end
    end
    
    % start RF...
    if strcmpi(options.selAlgo,'rf')
        lam = options.lamRF;
        for jj = 1:length(lam)
%             for kk = 1:length(options.mls)
%                 for ll = 1:length(options.mns)
%                     t = templateTree('NumVariablesToSample','all','MaxNumSplits',options.mns(ll),'MinLeafSize',options.mls(kk),...
%                         'PredictorSelection','interaction-curvature','Surrogate','on');
%                     mdl = fitrensemble(Xtmp,Ytmp,'Method','Bag','NumLearningCycles',lam(jj),'Learners',t);
%                     scores{jj,kk,ll} = oobPermutedPredictorImportance(mdl);
%                     [~,id{jj,kk,ll}] = sort(scores{jj,kk,ll},'descend');
%                     fsc(oid(id{jj,kk,ll}(1:options.maxVars)),jj,kk,ll) = fsc(oid(id{jj,kk,ll}(1:options.maxVars)),jj,kk,ll)+1;
%                 end
%             end
                    t = templateTree('NumVariablesToSample','all',...
                        'PredictorSelection','interaction-curvature','Surrogate','on','MaxNumSplits',3,'MinLeafSize',6);
                    mdl = fitrensemble(Xtmp,Ytmp,'Method','Bag','NumLearningCycles',lam(jj),'Learners',t);
                    scores{jj} = oobPermutedPredictorImportance(mdl);
                    [~,id{jj}] = sort(scores{jj},'descend');
                    fsc(id{jj}(1:options.maxVars),jj) = fsc(id{jj}(1:options.maxVars),jj)+1;
        end
    end
end

% select stable set...
fsc = fsc./options.rep;
fscmx = squeeze(max(fsc,[],[2:length(size(fsc))]));
if options.thresh > 1 % if threshold is not a proportion we assume you want a fixed set of selected features = threshold
    [~,fk] = maxk(fscmx,round(options.thresh));
else
    fk = find(fscmx > options.thresh);
end
maxVars = options.maxVars;
alpha = options.alpha;