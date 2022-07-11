function [fk,fsc,fscmx,maxVars,alpha,lam, scores, oid, ctr, mdl] = stabSel2(X,y,varargin)
% Perform stability selection using an elastic net. 
%
% Call: [fk,fsc,fscmx,maxVars,alpha,lam] = stabSel(X, y);
%
%% What do I need to provide stabSel: 
% X: an n x p matrix of predictors
% y: an n x 1 vector of responses
%
% Optional arguments can be supplied as well, like so: 
% stabSel(X,y,'maxVars',25)
%
%% Options ----------------------------------------------------------------
% 'maxVars' : # of variables elastic net is forced to choose on each
% subsample. If empty or left out, this will be calculated automatically to
% ensure FDR or FWER correction (p < 0.05).
%
% 'alpha' : alpha values for elastic net. 0 = ridge, 1 = lasso. Thus, you
% can do stability selection with ridge regression by setting alpha to a
% single number close to zero (e.g., 0.0001) or stability selection with
% lasso by setting alpha to 1.
%
% 'lam' : lambda values to include. 1000 values will be automatically
% generated (limits based on max l that produces a solution with 1 feature;
% limits defined across all alpha values because we take max 'stability'
% across parameters so l should be same or comparable across alpha values).
%
% 'prop' : proportion of data to subsample (i.e., 0.7 means
% 70% of the data will be used in each subsample; default: 0.5 but you may
% increase this number if dataset is small)
%
% 'rep' : number of subsamples (default: 200)
%
% 'stnd' : standardize (default: true). If true, will still be turned off
% after computing weights IF you are using adaptive EN/lasso/ridge.
%
% 'corr' : 'FDR' (default) or 'FWER'
%
% 'thresh' : threshold for retaining feature in stable set (i.e.,
% proportion of subsamples in which feature has to be selected). The
% default is 0.9 which is used for the FDR/FWER computation so if you
% change this, it breaks FDR/FWER AND maxVars. 
%
% 'parallel' : set to false (default) to avoid parallel computing 
%
% 'adaptive' : set to true to perform adaptive EN/lasso/ridge (which of 3
% determined by your als values). This simply initializes your preferred
% algorithm with weights from ridge regression.
%
%% Outputs ----------------------------------------------------------------
% fk : features kept (i.e., stable set of features)
% fsc : empirical probabilities for pairs of regularization parameters
% fsmx : max empirical probability across regularization parameters
% maxVars : # of variables/features selected in each subsample
% alpha : alpha values used 
% lam : lambda values used

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
if str2double(v(1:end-1)) < 2022 && strcmpi(options.selAlgo,'fsrmrmr')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2022...please install a more recent version of matlab (2022+) to use fsrmrmr for feature selection'])
end
if str2double(v(1:end-1)) < 2017 && strcmpi(options.selAlgo,'nca')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2017...please install a more recent version of matlab (2017+) to use NCA for feature selection'])
end
if str2double(v(1:end-1)) < 2011 && strcmpi(options.selAlgo,'relieff')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2011...please install a more recent version of matlab (2011+) to use fsrmrmr for feature selection'])
end
if str2double(v(1:end-1)) < 2012 && strcmpi(options.selAlgo,'en')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2012...please install a more recent version of matlab (2012+) to use elastic net, lasso, or ridge for feature selection'])
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

% check for warning...
if ~isempty(options.lamEN) && options.lamENInside
    if options.verbose
        warning('You cannot pass in lambda values AND try to define lambda values inside the subsamples. Assuming turning on lamENInside was a mistake...setting this option to false.')
    end
end

% check to see if you want to do outlier removal now...
oid = [];
mdl = [];
if strcmpi(options.outlier,'outside')
    [X,oid,mdl,~] = rmOutliers(X,'prepro',options.outlierPrepro,'nonconsec',options.outlierPreproNonconsec,'outlierMethod',options.outlierMethod,'outlierDir',options.outlierDir,'outlierThresh',options.outlierThresh,'pcaP',options.outlierPreproP,'pcaPermN',options.outlierPreproN,'propOutliers',options.propOutliers,'rep',options.outlierReps,'repThresh',options.outlierRepThresh);
    % check if you want to adjust subsampling prop. to account for #
    % outliers removed...prop will be adjusted to keep training set same
    % size as if the prop selected was applied to all data...
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
        options.lamRF = linspace(10,500,10);
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
            [~,~,tmpl(:,j)] = defLam(Xtmp,y(ctr{j}),options.alpha,options.stnd,options.lmx,options.lmn,options.lamRatio,options.lst,options.ln);
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
           for jj = 1:size(lsB,2)
               if ~isempty(lsB)
                   id = find(lsB(:,jj)~=0);
                   fsc(id,jj+adj,kk) = fsc(id,jj+adj,kk)+1;
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
    [~,fk] = maxk(fscmx,options.thresh);
else
    fk = find(fscmx > options.thresh);
end
maxVars = options.maxVars;
alpha = options.alpha;