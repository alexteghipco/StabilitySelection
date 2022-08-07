function [fk,fsc,fscmx,maxVars,alpha,lam,scores,oid,ctr,mdl,ep,empMaxVars,thresh,numFalsePos] = stabSel(X,y,varargin)
%
% Identify a stable set of features in your data using the framework of
% stability selection.
%
% Call: [fk,fsc,fscmx,maxVars,alpha,lam,scores,oid,ctr,mdl,ep,empMaxVars] = stabSel(X, y);
%
%% ------------------------------------------------------------------------
% What do I need to provide stabSel:  -------------------------------------
% -------------------------------------------------------------------------
% X: an n x p matrix of predictors
% y: an n x 1 vector of responses
%
% Optional arguments can be supplied as well, like so: 
% % stabSel(X,y,'maxVars',25,'stnd',false)
%
%% ------------------------------------------------------------------------
% What does stabSel do? ---------------------------------------------------
% -------------------------------------------------------------------------
% Use one of 13 feature selection methods (see 'selAlgo'; default is an
% elastic net) within the framework of stability selection to identify a
% "stable" set of features (see output 'fk') in input matrix X. The stable
% set of features contains those features that have a higher probability of
% being selected (see fscmx for empirical probabilities for each feature)
% by your feature selection method across many perturbed versions of your
% dataset (see 'samType' for resampling options). The feature selection
% method selects the top N features in the dataset that are predictive of
% your response variable, so the stable set of features can be thought of
% as the columns of X that more consistently predict y.
%
% The two most critical parameters in stability selection are the N
% variables that a feature selection method selects on average (see
% 'maxVars'), and the probability threshold that determines whether a
% feature enters the stable set (see 'thresh'). Given these two parameters
% (and assuming a default-set proportion of the data for resampling), we
% can compute the number of false positives in the stable set (see
% citations for equation), and by extension, the FWER p-value. This also
% means that we can set the number of false positives that we would like
% before starting stability selection (see'numFalsePos'; or we can  set
% fwer, see 'fwer') and automatically determine the number of variables our
% feature selection should be forced to select (maxVars), or the threshold
% we should use for forming the stable set (thresh). It is easy to set
% 'thresh' prior to running stabSel, but it can be tricky to set 'maxVars'.
% That's because the number of features selected by some methods is
% determined in an opaque way by a parameter that we have little insight
% into. This means the range we use for this parameter will determine
% maxVars. However, stabSel still allows you to set a specific 'maxVars'
% ahead of time in such cases. That's because for some methods where this
% occurs, we can simply look at a model's weights and use those as a
% filter, while in other cases, we can force the selection method to select
% maxVars *or fewer* features. This means that although we can control FWER
% by specifying maxVars or thresh, in some cases, when we specify maxVars
% rather than then thresh, we will end up with a lower effective FWER
% (i.e., because the actual average number of features selected by the
% selection method will be less than maxVars).
% 
% If you do not set 'maxVars' or 'thresh', stabSel will either use a
% heuristic to select the number of variables the selection method should
% be choosing on average (e.g., when using a filter where we *have* to know
% how many features to select), or will use a series of regularization
% parameters indescriminantly (i.e., without forcing selection of maxVars
% or fewer features) and then choose an appropriate 'thresh'. In both
% cases, by default, stabSel will try to ensure <1 false positive in the
% stable set.
% 
% StabSel also supports outlier detection and removal (one of 4
% methods) either prior to, or during, the resampling procedure. By
% removing outliers within the resampling procedure, a more generalizable
% method of outlier removal is implemented.
% 
% There are many options for you to tinker with in stabSel, but you you can
% leave these out of your call and stabSel will use very reasonable
% defaults. The exhaustive list of options are detailed in the following
% section, which also tells you what stabSel internally sets each option to
% if you don't refer to it/change it in your call.
%
% A synthesized summary of main default options so you don't have to read
% the documentation: by default, stabSel will use an elastic net to select
% your features across 50 complementary bootstraps of your data (this
% should be appropriate, see Shah & Samworth citaton). A series of
% regularization parameters will be used in lieu of a fixed number of
% selected variables. The series will be defined by the highest parameter
% value that is estimated to select a single feature across all resampled
% datasets.
% 
% The average number of variables selected across parameters will then be
% used to determine a  probability threshold for the stable set of features
% that ensures fewer than 1 false positive. Outlier removal will not be
% performed.
%
% Final note: Consider using an adaptive elastic net (not default because
% it requires more manual intervention; see documentation). If you are
% using stabSel to select features *for* a model, consider ensuring that
% the data used for feature selection and model validation and/or tuning is
% independent. This will give more accurate prformance estimates for the
% model. Otherwise, in some projects it will not be unreasonable for
% feature selection to be an analysis that is independent of model building
% (e.g., when a more parsimonious model is irrelevant).
%
%% ------------------------------------------------------------------------
% ----------------------------- Outputs -----------------------------------
% -------------------------------------------------------------------------
% fk : indices of features that form the stable set (i.e., features kept) 
% 
% fsc : empirical probabilities for regularization parameters that were
% used (you probably don't care about this)
%
% fscmx : max empirical probability across regularization parameters for
% each feature. Note, in stability selection we take the max proportion of
% times that a feature was selected ACROSS all regularization parameters.
% You can inspect this to get a different stable set using some other
% probability threshold without re-running stabSel (but then ignore any
% output about false positives or fwer)
%
% maxVars : # of variables/features specified to be selected in each
% resampled dataset.
%
% alpha : alpha values used. Only applies to elastic net, lasso, ridge.
% These will map onto a dimension of fsc (different for different
% selection methods)
%
% lam : lambda values used. Only applies to elastic net, lasso, ridge, nca,
% GPR, RF. These will map onto a dimension of fsc (different for different
% selection methods)
%
% 'scores' : returns weighting of features for linear
% regression/correlation, relieff, mrmr, nca.
%
% 'oid' : rows of X that are determined to be outliers.
%
% 'ctr' : indices of samples that were used to make each resampled dataset.
%
% 'mdl' : this is a model that was trained to identify outliers (can be
% used to predict outliers in new data). If outliers were detected inside
% the subsampling scheme, there is a model for each subsample. You can get
% a consensus of the predictions of these models on new data.
%
% 'ep' : effective FWER p-value based on the average # of features selected
% across resampled datasets and the threshold you selected.
%
% 'empMaxVars' : empirical maxVars. If your selection algorithm is a
% filter, then this will equal maxVars. However, algorithms that have
% regularization parameters that shrink features to zero will select a
% number of features that are not equal to maxVars because it is impossible
% to tell exactly which parameters will give exactly the number of features
% specified. In some cases the number of features taken from each resampled
% dataset will be fixed to *not more than*  maxVars. Or, maxVars can be
% ignored and a threshold selected based on empirical maxVars to maintain a
% predetermined FWER.
%
%% ------------------------------------------------------------------------
% General optional arguments (apply to all selection methods) -------------
% -------------------------------------------------------------------------
% 'samType' : determines the resampling scheme. Set to 'subsample' to take
% subsamples of your data as per the original stability selection paper
% (see section above). Set to 'bootstrap' to take bootstraps of your data.
% It's possible bootstrapping injects more noise into the data, which is
% beneficial to stability selection. But note, subsampling w/50% of your
% data is close to the equivalent of bootstrapping. If bootstrapping,
% consider getting complementary pairs (see next option). Default:
% 'bootstrap'.
%
% 'compPars' : determines if subsampling or bootstrapping follows the
% complementary pairs principle where each subsequent resample has no
% overlap with the previous one. Note, this makes less sense to do with
% subsampling and works best when 'prop' is ~0.5. Can be true or false.
% Default: true.
%
% 'rep' : number of bootstraps/subsamples to draw. Increase this to 300+ if
% 'compPars' is false. Default: 50.
%
% 'prop' : proportion of data to resample (i.e., 0.7 means 70% of the data
% will be used in each subsample or that a bootstrap will be of a size that
% is 70% of the data; it makes sense to increase this number above default
% if the dataset is very small but this may break FWER/false positive
% calculations). If 'prop' is 1 we use all of the data. Default: 0.5.
%
% 'propN' : if you would like to specify the number of samples to use in
% each resample (rather than the proportion of the data), set 'prop' to be
% greater than or equal to 1 (i.e., the number of samples you would like
% each subsample to contain) and propN to true. Default: false.
%
% 'adjProp' : if you are doing outlier detection, by default, the
% proportion of data used in each resample is cacluated *after* outlier
% removal. But, you can set adjProp to true in order to apply the
% proportion *before* outlier removal and ensure that this is the number of
% samples that is taken *after* outlier removal. Default: false.
%
% 'numFalsePos' : threshold for the number of false positives you would
% like to ensure in the stable set (<). You should provide either this OR
% fwer. This will have no effect if both 'maxVars' and 'thresh' are passed
% in. Default: 1.
%
% 'fwer' : FWER p-value threshold for stable set. You should provide either
% this OR fwer. This will have no effect if both 'maxVars' and 'thresh' are
% passed in. Default: [].
%
% 'maxVars' : average # of variables the selection algorithm should choose
% on each subsample. Consider passing this into stabSel as blank if your
% selection algorithm has a regularization parameter that controls the
% number of features selected (lasso, ridge, elastic net, nca). If
% 'maxVars' is not empty, but 'thresh' is, then stabSel will determine a
% 'thresh' that ensures FWER or number of false positives provided.
% Default: [].
%
% 'thresh' : threshold for proportion of subsamples a feature must appear
% in to enter the stable set. Consider passing this into stabSel as blank
% if your selection algorithm has a regularization parameter that controls
% the number of features selected. This way, a threshold will be computed
% based on the fwer or number of false positives you have specified. If
% 'thresh' is not empty, but 'maxVars' is, then stabSel will determine a
% 'maxVars' that ensures FWER or number of false positives provided. If
% threshold is > 1 we assume you are telling stabSel how many features to
% select and fwer/false positives will no longer apply. Default: 0.9.
% Default: [].
%
% 'selAlgo' : sets the algorithm for selecting features. This may be 'EN'
% for an elastic net (see lasso.m for more detail), 'lasso' for lasso
% regression (see lasso.m for more detail), 'ridge for ridge regression
% (see lasso.m for more detail), 'corr' for correlation (see corr.m for
% more detail), 'robustLR' for robust linear regression (not as sensitive
% to outliers but takes MUCH longer to run; see fitlm.m for more detail),
% 'NCA' for neighborhood components analysis (see fsrnca.m for more
% detail), 'ftest' for an ftest (see fsrftest.m for more detail), 'mrmr'
% for minimum redundance maximum relevance algorithm that relies on mutual
% information (see fsmrmr.m for more detail), 'releiff' for the
% ReliefF/RReliefF algorithm that relies on nearest neighbor search (see
% relieff.m for more detail), 'GPR' for a gaussian process model (see
% fitrgp.m for more detail), and 'RF' for a random forest (see
% fitrensemble.m). Note, that an additional argument can be used to adjust
% 'EN' or 'lasso' to be adaptive (in both cases, coefficients from ridge
% regression are used to weight your data before applying lasso or an
% elastic net. For more information, see: Zou, H., & Zhang, H. H. (2009).
% On the adaptive elastic-net with a diverging number of parameters. Annals
% of statistics, 37(4), 1733.). Note, some of these options require more
% recent versions of matlab, but stabSel will check this for you. Default:
% 'EN'.
%
% 'stnd' : standardize X (z-score cols) if set to true, otherwise set to
% false. When selAlgo is 'EN', 'lasso', 'ridge', or 'GPR' stnd is passed
% into the respective matlab commands. RF does not require standardization
% and can hurt performance. For all other options, standardization will be
% performed inside the subsampling scheme and *after* outlier detection.
% Stnd also applies to some outlier detection methods (ocsvm).
%
% 'parallel' : set to true to attempt parallel computing where possible.
% Applies to scripts stabSel draws on as well. Default: false.
%
% 'verbose' : set to true to get feedback in the command window about what
% stabSel is doing. Default: false.
%
%% ------------------------------------------------------------------------
% ------- Optional arguments that apply to > 1 selection algorithm --------
% -------------- (elastic net, lasso, ridge, nca, gpr, rf) ----------------
% -------------------------------------------------------------------------
% 'lam' : user-specified lambda (regularization parameter) sequence to be
% used for elastic net, lasso, ridge, GPR, or NCA. When this is empty
% (i.e., by default), lambdas are computed automatically between lmx and
% lmn (see options lmn, lmx, ln below for details). In the case of EN,
% lasso, and ridge, this is done separately for each alpha value, so each
% alpha value will have its own unique lambda sequence. This is necessary
% because lambda values for ridge and lasso will typically have very
% different scales based on whatever lambda is necessary to create the
% sparsest model possible. If adaptive is set to true, the values you give
% 'lam' will apply to the elastic net or lasso being used to select
% features and NOT to the ridge regression used to initially weight the
% data (see lamAEN below for where to supply those if you would like). Note,
% Default lambda values for NCA will be set between 0.0001 to 100 before
% weighting by standard deviation of the response variable y. For GPR,
% lambda values refer to the regularization standard deviation values. By
% default, these will be set between the standard deviation of your
% response variable y multiplied by 0.01 and then divided/multiplied by 3.
% Weighting will be omitted if you supply your own values for NCA or GPR.
% Default:
%
% 'lamInside' : determines if automatically defined lambda values will be
% computed BEFORE subsampling, ACROSS subsamples or DURING subsampling
% (applies to EN/lasso/ridge and GPR). Set to 'during' to compute lambda
% values unique to each subsample. This only works for lasso, en, and ridge
% regression. It is not recommended unless stabSel returns warnings about
% being unable to select good lambda values to fit your subsamples
% (specifically when lamInside is set to either of the other two options).
% Having the same lambda values across subsamples is arguably preferable
% for interpretation because we take the max 'stability' of a feature
% across regularization parameters. Set this option to 'before' in order to
% compute one set of lambda values using your entire input data X and y.
% Set this option to 'across' to first define unique lambda sequences for
% each subsample (as would be done if you set this to 'during'), then
% compute a single lambda sequence using the smallest and largest lambda
% values across subsamples. If using EN, all of these options will be
% executed independently for each alpha value. Default: false.
%
% 'lmx' : max lambda value to use in lambda sequence for elastic net, lasso
%, ridge, NCA, and GPR. For elastic net, lasso and ridge, leaving this
% blank will use the 'max l trick' to compute the max lambda value with
% which the model will select at least a single feature (by taking the max
% value of the dot product of your X and y variables, then dividing it by
% the number of samples in your data multiplied by the weight of the L1 to
% L2 norm being used). For NCA the default is 100 adjusted by the std(y)
% and for GPR the default is 1e-2*std(y)*3. For NCA only, if you pass in a
% value, it will still be weighted by std(y). If you would like to specify
% exact min/max lambdas for NCA, use 'lam'. Default: [].
%
% 'lmn' : min lambda value to use in lambda sequence for elastic net, lasso
%, ridge, NCA, and GPR. For elastic net, lasso and ridge, leaving this
% blank will multiply lmx by 'lamRatio' to get lmn. For NCA, the default is
% .001 adjusted by the std(y) and for GPR the default is 1e-2*std(y)/3. For
% NCA only, if you pass in a value, it will still be weighted by std(y).
% If you would like to specify exact min/max lambdas for NCA, use 'lam'.
% Default: [].
%
% 'ln' : number of lambda values to include in an automatically generated
% lambda sequence. This applies to elastic net, lasso, ridge, NCA, GPR, and
% RF (note RF lambda values are passed in currently as 'lamRF'; see below).
% Default: 100.
%
% 'lst' : determines how values will be spaced between lmn and lmx to
% automatically produce a lambda sequence. Set to 'linear' or 'log'.
% Consider using log for en, lasso, ridge (but argument also applies to GPR
% and NCA). This recommendation is based on the fact that in most cases,
% lmx will be quite high, causing most models to be sparse. Alternatively,
% increase 'ln' dramatically (to maybe 10,000; also increase 'lamRatio') but
% this will not be very efficient. Default: 'linear'.
%
% 'logDirPref' : if 'lst' is set to 'log', this will determine which end of
% the lambda sequence you would like to sample more. Set this to 'smaller'
% to sample more lambda values closer to lmn. Set this to 'larger' to
% sample more lambda values closer to lmx (i.e., sparser models). Default:
% 'smaller'. 
%
% 'lamOutlier' : only applies when lamInside is set to 'across'. Sometimes
% a lambda sequence for one subsample will be drastically different than
% the lambda sequence for another subsample. If you want to define a single
% lambda sequence across all subsamples, it is helpful to remove outlier
% lmn and lmx values as we take the lowest lmn and highest lmx to define
% this single sequence. If set to true, we will remove lmx values above
% the 95th percentile (as defined across subsamples). Default: true.
%
% 'filter' : It is possible to use some selection algorithms as a filter
% (robust linear regression, correlation, NCA). Set to true to use as
% filter. If set to true, maxVars number of variables will be selected by
% sorting the scores/weights of the selection algorithm. If false, see
% option below. Default: false.
%
% 'filterThresh', : threshold used to retain features if filter is set to
% false. Corresponds to a p-value to use for selecting features for
% correlation and robust linear regression. Corresponds to weight threshold
% for NCA (but should be 0 in most cases). Default: [].
%
%% ------------------------------------------------------------------------
% ------------ Optional arguments for elastic net, lasso, ridge -----------
% -------------------------------------------------------------------------
% stabSel was developed with lasso regression in mind so you may notice
% more options for it at the moment (see original stability selection paper
% referenced above).
%
% 'adaptive' : sets EN or lasso to be adaptive. Lasso lacks oracle
% properties, may have 'inaccurate' weights, or inconsisttently select
% variables (e.g., noise variables which is especially true when n >> p;
% see Zou & Zhang, 2009). Adaptive lasso has oracle properties but lacks
% the ability to select *all* of the correlated variables that are
% predictive. The L2 regualrization in EN fixes this, but EN lacks oracle
% properties as well so ideally we would use adaptive EN. In practice,
% adaptive EN does not always work flawlessly. This is in part because
% adaptive EN introduces a new parameter to tune: gamma, which controls the
% degree to which the ridge regression is used to weight the data. It is
% unclear how to best tune this parameter within stabSel so this is
% currently being worked out (best approach will probably be nesting so
% that we subsample the subsamples and get out of bag error). However, some
% a prior set gamma value may work quite well for your data. Note, adaptive
% EN can significantly increase computational time depending on your data.
% Finaly, note, a single regualrization parameter must be selected for the
% ridge regression used to weight data. If adaptive EN is performed inside
% the resampling scheme (see next option), we choose the parameter that
% minimizes out of sample error. Default: false.
%
% 'adaptOutside' : only applies if adaptive is true. Determines if you are
% weighting data (as in adaptive EN, lasso, ridge) BEFORE (set to true) or
% DURING (set to false) subsampling. Weighting data DURING subsampling
% probably makes the most sense, otherwise there is some data leakage.
% Default: false.
%
% 'ridgeRegSelect' : only applies if 'adaptive' is true and 'adaptOutside'
% is true. Ridge regression used for weighting data is run on a sequence of
% automatically generated lambda values and we take the result that shrinks
% the least number of features to zero (usually the model that contains all
% features). In most cases multiple lambda values meet this criterion. In
% such cases, you can decide to take the largest lambda (set this to
% 'largest'), the smallest lambda (set this to 'smallest') or the average
% lambda value (set this to 'middle'). Default: 'largest' (i.e., greatest
% penalty that shrinks the least features to zero).
%
% 'gam' : only applies if adaptive is true. Gamma value for controlling
% extent to which data is weighted by ridge regression. Default: 1.
%
% 'alpha' : alpha values for determining weight of L1 to L2 optimization in
% elastic net. This can be an n x 1 vector. 1 is equal to lasso, values
% close to but not zero are equal to ridge. Default: [0.1:0.1:1].
%
% 'lamAEN' : Lamda values to be used in ridge regression if adaptive is
% true. If false, 'max l trick' will be used. Default: [].
%
% 'lamRatio' : to get a sequence of lambda values, we use the 'max l trick'
% to get the max lambda, then use this ratio to get the smallest lambda.
% Default: 1e-4.
%
% 'fixMax' : can be set to true to force elastic net, lasso, ridge,
% to return a maximum of 'maxVars' features irrespective of regularization
% parameter lambda. This is set to true if using any of these selection
% algorithms and passing in maxVars to ensure we do not select more than
% the number of variables you want. Note also that if fixMax is set to true
% and maxVars is empty, then maxVars will be automatically computed based
% on FWER/false positives (if you use any of the above selection
% algorithms). Default: false.
%
%% ------------------------------------------------------------------------
% ------------------ Optional arguments for relieff -----------------------
% -------------------------------------------------------------------------
% 'rK' : determines k in knn (# of neighbors). Default: [1:14
% 15:10:size(X,2)/2].
%
% 'rE' : sigma, or distance scaling factor. Consider changing this to just
% 50 if stabSel is taking too long on your data. Default: [10:15:200].
%
%% ------------------------------------------------------------------------
% ---------------------- Optional arguments for GPR -----------------------
% -------------------------------------------------------------------------
% None but as a note, GPR will use: ardsquaredexponential kernel
% function, lbfgs optimizer, fully independent conditional approximation
% for prediction, and subset of regressors approximation for the fit
% method.
%
%% ------------------------------------------------------------------------
% ---------------------- Optional arguments for RF ------------------------
% -------------------------------------------------------------------------
% 'lamRF' : this determines the number of learning cycles to use for the
% random forest. Can be an n x 1 sequence. More options for RF are
% available but not currently implemented due to computational costs
% required to execute. To make them available, uncomment relevant code. You
% can then pass in a learning rate using optional argument 'lr'. Max. #
% splits using optional argument 'mns' and min leaf size using optional
% argument 'mls'. In these 3 cases, you should pass in a sequence of values
% that is n x 1 in size. Default values for lamRF are 10 linearly spaced
% values between 10 and 500. Default values for min leaf size are
% round(linspace(1,max(2,floor(size(X,2)/2)),15)). Default values for max #
% splits are round(linspace(1,size(X,2)-1,15)). Default values for learning
% rate are linspace(1e-3,1,10). Default: [].
%
%% ------------------------------------------------------------------------
% ---------- Optional arguments for outlier detection/removal -------------
% -------------------------------------------------------------------------
% 'outlier' : determines where outlier detection removal will be performed
% on matrix X. Set to 'outside' to perform outlier detection before
% subsampling and 'inside' to perform outlier detection inside the
% subsampling loop. Set to 'none to avoid detection/removing outliers.
% Default: 'none'.
%
% 'outlierPrepro' : currently can only be set to 'none' or 'pca'. In the
% latter case, PCA is performed first, before outlier detection. This is
% useful because some methods of outlier detection do not work with lots of
% features (i.e., robustcov). Default: none.
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
% prform outlier detection using some scaled MAD from the median, some
% std devs from the mean, some IQ ranges above/below upper lower quartiles,
% using the grubb's test, or using a generalized extreme studentized
% deviate test for outliers. Set this to 'fmcd, 'ogk', or 'olivehawkins' to
% use different methods for estimating robust covariance to detect outliers
% (robustcov.m). Use 'iforest' to generate an isolation forest to identify
% outliers. Use 'ocsvm' to use one class svm to identify outliers. Default:
% median.
% 
% 'outlierReps' : some outlier detection methods are stochastic and
% repeating them can be helpful (ocsvm and iforest). This determines the
% number of times that they will be repeated. Default: 1 (i.e., no
% repeats).
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
%% ------------------------------------------------------------------------
% --------------------- Optional arguments for nca ------------------------
% -------------------------------------------------------------------------
%
%% ------------------------------------------------------------------------
% ------ Optional arguments for corr and robust linear regression ---------
% -------------------------------------------------------------------------
%
%% ------------------------------------------------------------------------
% Citations  --------------------------------------------------------------
% -------------------------------------------------------------------------
% For original method and details of stability selection, see: Meinshausen,
% N., & Bühlmann, P. (2010). Stability selection. Journal of the Royal
% Statistical Society: Series B (Statistical Methodology), 72(4), 417-473.
%
% For complementary pairs, see: Shah, R. D., & Samworth, R. J. (2013).
% Variable selection with error control: another look at stability
% selection. Journal of the Royal Statistical Society: Series B
% (Statistical Methodology), 75(1), 55-80.

%% ------------------------------------------------------------------------
% Internal notes ----------------------------------------------------------
% -------------------------------------------------------------------------
% 1) make lasso return weights for lambda that was chosen as scores
% 2) verbose warnings, displays are not consistent across algorithms
% 3) need more verbose options and feedback
% 4) fix RF defaults and just warn about computational time (maybe based on
% # of features)
% 5) documentation details lots of recommendations where settings  *should*
% be adapted based on other settings being changed away from defaults.
% Check if user changed these and update them yourself if they are
% obviously terrible, otherwise just throw a warning. 
% 6) NCA could have lambdas defined inside sampling scheme...
%
% Alex Teghipco // alex.teghipco@sc.edu // July 13, 2022

% load in defaults
options = struct('maxVars',[],'propN',false,'adjProp',true,'alpha',[0.1:0.1:1],...
    'lam',[],'lamInside','during','lamAEN',[],'lamRatio',1e-4,'adaptOutside',...
    false,'lmn',[],'lmx',[],'ln',1000,'prop',0.5,'rep',50,'stnd',true,'numFalsePos',...
    [],'fwer',[],'thresh',0.9,'parallel',false,'adaptive',false,...
    'outlier','none','outlierPrepro','none','outlierPreproNonconsec',true,...
    'propOutliers',0.1,'outlierPreproP',0.05,'outlierPreproN',5000,...
    'outlierReps',1,'outlierRepThresh',1,'outlierMethod','median',...
    'outlierDir','row','outlierThresh',3,'selAlgo','en','lst','linear',...
    'gam',1,'lrp',0.05,'rK',[1:14 15:10:size(X,2)/2],'rE',[10:15:200],...
    'lamRF',[],'lr',[],'mls',[],'mns',[],'verbose',false,'logDirPref','smaller',...
    'filter',false,'filterThresh',0.05,'samType','bootstrap','compPars',true,...
    'ridgeRegSelect','largest','lamOutlier',true,'fixMax',false);
optionNames = fieldnames(options);
rng shuffle

% now parse the user-specified arguments and overwrite the defaults
vleft = varargin(1:end);
for pair = reshape(vleft,2,[]) %pair is {propName;propValue}
    inpName = pair{1}; % make case insensitive by using lower() here but this can be buggy
    if any(strcmpi(inpName,optionNames)) % check if arg pair matches default
        def = options.(inpName); % default argument
        %if ~isempty(pair{2}) % if passed in argument isn't empty, then write that in as the option
            options.(inpName) = pair{2};
        %else
        %    options.(inpName) = def; % otherwise use the default values for the option
        %end
    else
        error('%s is not a valid argument',inpName)
    end
end

% check compatibility
v = version('-release');
if str2double(v(1:end-1)) < 2022 && strcmpi(options.selAlgo,'mrmr')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2022...please install a more recent version of matlab (2022+) to use mrmr for feature selection'])
end
if (str2double(v(1:end-1)) < 2016 || (str2double(v(1:end-1)) == 2016 && ~strcmpi(v(end),b)))  && strcmpi(options.selAlgo,'nca')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2016b...please install a more recent version of matlab (2016b+) to use NCA for feature selection'])
end
if (str2double(v(1:end-1)) < 2010 || (str2double(v(1:end-1)) == 2010 && ~strcmpi(v(end),b))) && strcmpi(options.selAlgo,'relieff')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2010b...please install a more recent version of matlab (2010b+) to use relieff for feature selection'])
end
if (str2double(v(1:end-1)) < 2011 || (str2double(v(1:end-1)) == 2011 && ~strcmpi(v(end),b))) && (strcmpi(options.selAlgo,'en') || strcmpi(options.selAlgo,'lasso') || strcmpi(options.selAlgo,'ridge'))
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2011b...please install a more recent version of matlab (2011b+) to use elastic net, lasso, or ridge for feature selection'])
end
if (str2double(v(1:end-1)) < 2015 || (str2double(v(1:end-1)) == 2015 && ~strcmpi(v(end),b))) && strcmpi(options.selAlgo,'gpr')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2015b...please install a more recent version of matlab (2015b+) to use GPR for feature selection'])
end
if (str2double(v(1:end-1)) < 2016 || (str2double(v(1:end-1)) == 2016 && ~strcmpi(v(end),b))) && strcmpi(options.selAlgo,'rf')
    error(['Uh oh, it looks like your version of matlab (' v ') is from before 2016b...please install a more recent version of matlab (2016b+) to use RF for feature selection'])
end
if str2double(v(1:end-1)) < 2008 % cvpartition is faster than manual subsampling but requires 2008
    manualSubs = true;
else
    manualSubs = false;
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

% fix numFalsePos
if isempty(options.numFalsePos) && isempty(options.fwer)
   options.numFalsePos = 1;
end

% fix size of lam
if size(options.lam,1) == 1 && size(options.lam,2) > 1
    options.lam = options.lam';
end

% proportion may reflect N of subsample instead of an N to compute--if so
% convert to proportion.
if options.propN && options.prop >= 1
    tmpl = options.prop;
    options.prop = options.prop/size(X,1);
    if options.verbose
       disp('Changing subsampling N to proportion...')
       disp(['To achieve N of ' num2str(tmp) ' in subsample, proportion must be: ' num2str(options.prop)]);
    end
end

% fix fixMax if necessary
if strcmpi(options.selAlgo,'en') && ~isempty(options.maxVars) && ~options.fixMax
    options.fixMax = true;
    warning('When passing in a maxVars with lasso, ridge, elastic net, or nca we have to set fixMax to true to ensure a max of maxVars is returned across regularization parameters. Setting it to true now.')
end
if (strcmpi(options.selAlgo,'nca') || strcmpi(options.selAlgo,'corr') || strcmpi(options.selAlgo,'robustLR')) && ~isempty(options.maxVars) && ~options.filter
    options.filter = true;
     warning('When passing in a maxVars with correlation or robust linear regression, we assume you want to use both as a filter');
end

% warning about threshold if it is known--cannot be below 0.5 if you
% want to do fwer (p or num false positives)
if ~isempty(options.thresh) && options.thresh < 0.5 && (~isempty(options.numFalsePos) || ~isempty(options.fwer))
    warning('You have asked to estimate some false positives in the stable set, but your threshold is below 0.5. We can only estimate false positives when thresh is > 0.5. Fixing thresh to 0.501.')
    options.thresh = 0.501;
end

% Check for missing fwer and/or numFalsePos
if (isempty(options.numFalsePos) && isempty(options.fwer)) && (isempty(options.thresh) || isempty(options.maxVars))
    options.numFalsePos = 1;
    warning('You did not set thresholds for number of false positives or FWER. Typically this is not a problem, but you also did not set BOTH thresh and maxVars. Setting number of false positives to 1 so that it is possible to get these variables.')
end

% get number of features (avg) our regularization parameters should produce
% if user-specified threshold does not exist
if isempty(options.thresh) && isempty(options.maxVars) && (options.fixMax && strcmpi(options.selAlgo,'en')) || (~options.filter && (strcmpi(options.selAlgo,'corr') || strcmpi(options.selAlgo,'robustLR') || strcmpi(options.selAlgo,'nca')))
    if ~isempty(options.numFalsePos)
        options.maxVars = round(sqrt(0.8*options.numFalsePos*size(X,2)));
        disp(['Estimated maxVars will be: ' num2str(options.maxVars) ' based on a threshold of less than ' num2str(options.numFalsePos) ' false positives'])
    else
        warning('Estimating maxVars that will produce <1 false positive. If there is a FWER threshold, it will be applied when estimating probability threshold for defining stable set..')
        options.maxVars = round(sqrt(0.8*size(X,2)));
    end
end

% check that there is not BOTH fwer and numFalsePos
if ~isempty(options.numFalsePos) && ~isempty(options.fwer)
    options.numFalsePos = [];
    warning('You cannot specify both FWER threshold and a threshold for number of false positives. Keeping FWER.')
end

% get number of features (avg) our regularization parameters should produce
% if user-specified threshold exists
if isempty(options.maxVars) && ~isempty(options.thresh)
   if ~isempty(options.fwer)
       tmpii = linspace(0.0001,size(X,2)/3,100000);
        disp(['Finding number of false positives for which FWER should be : ' num2str(options.fwer)])
        for i = 1:length(tmpii)
            n1 = sqrt(size(X,2)*(tmpii(i)/(1/((2*options.thresh)-1))));
            tmpi(i,1) = ((1/((2*options.thresh)-1))*((n1.^2)/size(X,2)))/n1;
        end
        id = find(tmpi < options.fwer,1,'last');
        if ~isempty(id)
            options.numFalsePos = tmpii(id);
        else
            options.numFalsePos = NaN;
            warning('It was not possible to find a number of false positives that would ensure selected fwer. numFalsePos will be set to 1 to estimate maxVars.')
        end
   end
   if ~isnan(options.numFalsePos)
       n1 = sqrt(size(X,2)*(options.numFalsePos/(1/((2*options.thresh)-1))));
       if n1 > 1
           options.maxVars = round(n1);
       else
           options.maxVars = round(n1+1)-1;
       end
   else
        options.maxVars = round(sqrt(0.8*size(X,2)));
   end
   if options.maxVars == 0
       error('The specified threshold does not allow you to have a number of maxVars (average # of features selected) that can achieve less than the number of false positives you have indicated. Increase the number of false positives (or fwer) you are comfortable with, or decrease the threshold.')
   else
       disp(['Average # of features selected by method (maxVars) should be : ' num2str(options.maxVars) '. This will ensure ' num2str(options.numFalsePos) ' or fewer false positives'])
   end
end

% warn about correction
if options.verbose && options.prop ~= 0.5
    warning('Having a sampling proportion other than 0.5 may break estimation of the number of false positives (incl. fwer option). This may be okay to do in some cases (see original stability selection paper).')
end

% estimate # variables we SHOULD force algorithm to select...
%if ~isempty(options.maxVars) && ~isempty(options.thresh) && 

% if isempty(options.maxVars) && ~isempty(options.thresh)
%     options.maxVars = round(sqrt(0.8*options.corr*size(X,2)));
%     if options.verbose
%         disp(['The number of variables you should force your selection algorithm to choose is: ' num2str(options.maxVars)])
%     end
% end
%(1/((2*options.thresh)-1))*((options.maxVars.^2)/size(X,2))

% check for redundant lambda args...
if (~isempty(options.lam) && (strcmpi(options.lamInside,'across') || strcmpi(options.lamInside,'inside'))) && (strcmpi(options.selAlgo,'en') || strcmpi(options.selAlgo,'gpr'))
    if options.verbose
        warning('You cannot pass in lambda values AND try to define lambda values inside or across the resamples. Assuming turning on lamInside was a mistake...setting this option to false.')
    end
    options.lamInside = false;
end
if (strcmpi(options.lamInside,'across') || strcmpi(options.lamInside,'inside')) && (~strcmpi(options.selAlgo,'en') && ~strcmpi(options.selAlgo,'gpr'))
    if options.verbose
        warning(['You cannot define lambda values inside or across the resamples with your chosen selection algorithm: ' options.selAlgo '. Assuming turning on lamInside was a mistake...setting this option to false.'])
    end
    options.lamInside = false;
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
if (isempty(options.lam) && strcmpi(options.lamInside,'before')) && strcmpi(options.selAlgo,'en')
    if ~options.adaptive % if not adaptive
        if options.verbose
            disp('Computing lambda values for EN (lambda values are computed from the whole dataset and the same series will be passed in to each subsample)')
        end
        for kk = 1:length(options.alpha)
            [~,~,options.lam(:,kk)] = defLam(X,y,options.alpha(kk),options.stnd,options.lmx,options.lmn,options.lamRatio,options.lst,options.ln,options.logDirPref);
        end
    else % if adaptive we need to run first pass EN on whole data, get weights and get lambdas from those weights
        if options.verbose
            disp(['Computing lambda values for adaptive EN (lambda values are computed from the whole dataset weighted by an adaptive EN and the same series will be passed in to each subsample)'])
        end
        [Xtmp,~,~,~] = alasso(X,y,[],[],[],options.stnd,options.lamAEN,options.gam,options.ridgeRegSelect,options.parallel);
        for kk = 1:length(options.alpha)
            [~,~,options.lam(:,kk)] = defLam(Xtmp,y,options.alpha(kk),options.stnd,options.lmx,options.lmn,options.lamRatio,options.lst,options.ln,options.logDirPref);
        end
        if options.adaptOutside
            X = Xtmp;
            options.adaptive = false;
        end
    end
    if options.verbose
        disp(['The min lambda for EN/lasso is set to : ' num2str(options.lmn)])
    end
    if options.verbose
        disp(['The min lambda for EN/lasso is set to : ' num2str(options.lmn)])
    end
end

% define lambdas for NCA
if isempty(options.lam) && strcmpi(options.selAlgo,'nca')
    if isempty(options.lmn)
        options.lmn = 0.001;
        if options.verbose
            disp(['The min lambda for NCA is set to a default of: ' num2str(options.lmn)])
        end
    end
    if isempty(options.lmx)
        options.lmx = 100;
        if options.verbose
            disp(['The max lambda for NCA is set to a default of: ' num2str(options.lmx)])
        end
    end
    if strcmpi(options.lst,'linear')
        options.lam = linspace(options.lmn,options.lmx,options.ln)*std(y)/length(y);
    elseif strcmpi(options.lst,'log')
        options.lam = exp(linspace(log(options.lmn),log(options.lmx),options.ln))*std(y)/length(y);
    end
end   

% define lambdas for GPR        
if (isempty(options.lam) && strcmpi(options.lamInside,'before')) && strcmpi(options.selAlgo,'gpr')
    if isempty(options.lmn)
        options.lmn = 1e-2*std(y)/3;
        if options.verbose
            disp(['The min lambda for GPR is set to a default of: ' num2str(options.lmn)])
        end
    end
    if isempty(options.lmx)
        options.lmx = 1e-2*std(y)*3;
        if options.verbose
            disp(['The max lambda for GPR is set to a default of: ' num2str(options.lmx)])
        end
    end
    if strcmpi(options.lst,'linear')
        options.lam = linspace(options.lmn,options.lmx,options.ln);
        if options.verbose
            disp(['Lambdas are on linear scale...(we make ' num2str(options.ln) ' of them)'])
        end
    else
        options.lam = exp(linspace(log(options.lmn),log(options.lmx),options.ln));
        if options.verbose
            disp(['Lambdas are on log scale...(we make ' num2str(options.ln) ' of them)'])
        end
    end
end

% define lots of vars for RF -- note we currently only use lamRF (number of
% learning cycles) because doing this much tuning is expensive...RF IS
% CURRENTLY EXPERIMENTAL
if isempty(options.lamRF) && strcmpi(options.selAlgo,'rf')
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
if strcmpi(options.selAlgo,'en')
    empMaxVars = repmat(NaN,length(options.alpha),1);
else
    empMaxVars = options.maxVars;
end

% preallocate subsample indices...this is in case you want to get
% lamba min/max using the subsamples (i.e., we get one series of l but
% tailored to the data)
if options.verbose
    disp('Preallocating subsample indices')
end
n = round(length(y)*options.prop);
if options.prop == 1
    if options.verbose && options.compPars
        warning('You cannot sample your entire dataset and ensure complementary pairs. Turning off complementary pairs.')
    end
    options.compPars = false;
end
for j = 1:options.rep
    if options.prop < 1
        if strcmpi(options.samType,'subsample')
            if manualSubs
                s = 1:length(y);
                if options.compPars && j ~= 1
                    s = setdiff(s,ctr{j-1});
                end
                if length(s) < n
                    error('Your sampling proportion is too high. Decrease it and/or turn off compPars (complimentary pairs; try decreasing first)')
                else
                    s = s(randperm(length(s)));
                    ctr{j} = s(1:n);
                end
            else
                c3 = cvpartition(size(X,1),'Holdout',options.prop);
                ctr{j} = find(test(c3)==1);
            end
        elseif strcmpi(options.samType,'bootstrap')
            s = 1:length(y);
            if options.compPars && j ~= 1
               s = setdiff(s,ctr{j-1});
            end
            ctr{j} = s(ceil(length(s)*rand(1,n)));
        end
    elseif options.prop == 1
        ctr{j} = 1:length(y);
        if options.verbose && j == 1
            warning('Your subsamples comprise the entire dataset')
        end
    end
    if strcmpi(options.lamInside,'across')
        if strcmpi(options.selAlgo,'en')
            if ~options.adaptive
                if options.verbose
                    if j == 1
                        disp('Computing lambda values for EN (lambda values are computed separately for each subsample)')
                    end
                end
                for kk = 1:length(options.alpha)
                    [~,~,tmpl] = defLam(X(ctr{j},:),y(ctr{j}),options.alpha(kk),options.stnd,options.lmx,options.lmn,options.lamRatio,options.lst,options.ln,options.logDirPref);
                    tmplmn(kk,j) = min(tmpl);
                    tmplmx(kk,j) = max(tmpl);
                end
            else
                tmpid = setdiff([1:length(y)],ctr{j});
                [Xtmp,~,bin,fitin] = alasso(X(ctr{j},:),y(ctr{j}),X(tmpid,:),y(tmpid),1e-300,options.stnd,options.lamAEN,options.gam,options.ridgeRegSelect,options.parallel);
                for kk = 1:length(options.alpha)
                    [~,~,tmpl] = defLam(Xtmp,y(ctr{j}),options.alpha(kk),false,options.lmx,options.lmn,options.lamRatio,options.lst,options.ln,options.logDirPref);
                    tmplmn(kk,j) = min(tmpl);
                    tmplmx(kk,j) = max(tmpl);
                end
                if options.verbose
                    if j == 1
                        disp(['Computing lambda values for adaptive EN (lambda values are computed separately for each subsample after weighting dataset by an EN and we define one series of lambda values based on min/max lambda across all subsamples)'])
                    end
                end
            end
        elseif strcmpi(options.selAlgo,'gpr')
            tmplmn(j) = 1e-2*std(y(ctr{j}))/3;
            tmplmx(j) = 1e-2*std(y(ctr{j}))*3;
        end
    end
end
if strcmpi(options.lamInside,'across') && strcmpi(options.selAlgo,'en') % this is because GPR lambdas will not vary across parameters but EN will
    %options.lam = tmpl;
    for kk = 1:length(options.alpha)
        if options.lamOutlier
            id = find(tmplmx(kk,:) < prctile(tmplmx(kk,:),95)==1);
        else
            id = 1:lenght(tmplmx(kk,:));
        end
        [~,~,options.lam(:,kk)] = defLam([],[],options.alpha(kk),options.stnd,max(tmplmx(kk,id)),min(tmplmn(kk,id)),options.lamRatio,options.lst,options.ln,options.logDirPref);
    end
elseif strcmpi(options.lamInside,'across') && strcmpi(options.selAlgo,'gpr')
    if strcmpi(options.lst,'linear')
        options.lam(:,j) = linspace(min(tmplmn),max(tmplmx),options.ln);
    elseif strcmpi(options.lst,'log')
        options.lam(:,j) = exp(linspace(log(min(tmplmn)),log(max(tmplmx)),options.ln));
    end
end

% now we initialize a matrix for counting selected features across
% subsamples
if strcmpi(options.selAlgo,'EN')
    fsc = zeros(size(X,2),options.ln,length(options.alpha)); % features x lam x alpha
elseif strcmpi(options.selAlgo,'robustLR') || strcmpi(options.selAlgo,'corr') || strcmpi(options.selAlgo,'mrmr') || strcmpi(options.selAlgo,'ftest')
    fsc = zeros(size(X,2),1); % features
elseif strcmpi(options.selAlgo,'nca') || strcmpi(options.selAlgo,'gpr')
    fsc = zeros(size(X,2),options.ln); % features x lam
elseif strcmpi(options.selAlgo,'relieff')
    fsc = zeros(size(X,2),length(options.rK),length(options.rE)); % features x knn x sigma
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
            disp(['Starting feature selection. Working on resample ' num2str(i) ' of ' num2str(options.rep)])
        else
            disp(['Working on subsample ' num2str(i) ' of ' num2str(options.rep)])
        end
    end
    stnd = options.stnd; % sometimes we will have to overwrite standardization so copy in user selected option for first step...
    
    % get subsample indices...
    Xtmp = X(ctr{i},:);
    Ytmp = y(ctr{i});
    
    % remove outliers defined as either rows (samples) or cols (features)
    si = 1:size(Xtmp,2);
    if strcmpi(options.outlier,'inside')
        [Xtmp,oid{i},mdl{i},~] = rmOutliers(Xtmp,'prepro',options.outlierPrepro,'nonconsec',options.outlierPreproNonconsec,'outlierMethod',options.outlierMethod,'outlierDir',options.outlierDir,'outlierThresh',options.outlierThresh,'pcaP',options.outlierPreproP,'pcaPermN',options.outlierPreproN,'propOutliers',options.propOutliers,'rep',options.outlierReps,'repThresh',options.outlierRepThresh);
        if strcmpi(options.outlierDir,'rows')
            Ytmp(oid{i}) = [];
        else
            si(oid{i}) = [];
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
       lam = options.lam;
       
       % run EN to get weights for adaptive lasso
       if options.adaptive
          tmpid = setdiff([1:length(y)],ctr{i});
          [Xtmp,~,~,~] = alasso(Xtmp,Ytmp,X(tmpid,:),y(tmpid),1e-300,stnd,options.lamAEN,options.gam,options.ridgeRegSelect,options.parallel);  
          stnd = false; % fix stnd for lasso OF weights
       end

       % now do lasso for each alpha (within each we pass in our lams)
       for kk = 1:length(options.alpha)
           if strcmpi(options.lamInside,'during')
               [~,~,lam(:,kk)] = defLam(Xtmp,Ytmp,options.alpha(kk),stnd,[],[],options.lamRatio,options.lst,options.ln,options.logDirPref);
           end
           if options.parallel
               if options.fixMax
                   [lsB,lsFit] = lasso(Xtmp,Ytmp,'Lambda',lam(:,kk),'Alpha',options.alpha(kk),'Standardize',stnd,'DFMax',options.maxVars,'Options',statset('UseParallel',true));
               else
                   [lsB,lsFit] = lasso(Xtmp,Ytmp,'Lambda',lam(:,kk),'Alpha',options.alpha(kk),'Standardize',stnd,'Options',statset('UseParallel',true));
               end
%                if sum(lsFit.DF == 0) == length(lam(:,kk)) && strcmpi(options.lamInside,'during') % add zero lambda if we are still not getting any variables...
%                    [~,~,lam(:,kk)] = defLam(Xtmp,Ytmp,options.alpha(kk),stnd,[],0,options.lamRatio,options.lst,options.ln,options.logDirPref);
%                    [lsB,lsFit] = lasso(Xtmp,Ytmp,'Lambda',lam(:,kk),'Alpha',options.alpha(kk),'Standardize',stnd,'DFMax',options.maxVars,'Options',statset('UseParallel',true));
%                end
           else
               if options.fixMax
                   [lsB,lsFit] = lasso(Xtmp,Ytmp,'Lambda',lam(:,kk),'Alpha',options.alpha(kk),'Standardize',stnd,'DFMax',options.maxVars);
               else
                   [lsB,lsFit] = lasso(Xtmp,Ytmp,'Lambda',lam(:,kk),'Alpha',options.alpha(kk),'Standardize',stnd);
               end 
% if sum(lsFit.DF == 0) == length(lam(:,kk)) && strcmpi(options.lamInside,'during')
%                    [~,~,lam(:,kk)] = defLam(Xtmp,Ytmp,options.alpha(kk),stnd,[],0,options.lamRatio,options.lst,options.ln,options.logDirPref);
%                    [lsB,lsFit] = lasso(Xtmp,Ytmp,'Lambda',lam(:,kk),'Alpha',options.alpha(kk),'Standardize',stnd,'DFMax',options.maxVars);
%                end
           end
           empMaxVars(kk) = mean([empMaxVars(kk) mean([lsFit.DF zeros(options.ln - length(lsFit.DF),1)'])],'omitnan');

           if options.verbose
               disp(['Smallest lambdas DF is: ' num2str(lsFit.DF(1))])
           end

           % we need to adjust lam in cases where lasso output excludes a
           % certain lam we passed in (i.e., l is too low).
           if ~isempty(lsFit.Lambda)
               [~, adj] = min(abs(lam(:,kk) - lsFit.Lambda(1))); adj = adj-1;
               if options.verbose
                   disp(['Percentage of lambdas that did not fit current run: ' num2str(adj/options.ln)])
               end

               for jj = 1:size(lsB,2)
                   %                    if lsFit.DF(jj) > options.maxVars
                   %                        [~,id] = maxk(lsB(:,jj),options.maxVars);
                   %                    else
                   %                        id = find(lsB(:,jj)~=0);
                   %                    end
                   id = find(lsB(:,jj)~=0);
                   if ~isempty(id)
                       fsc(si(id),jj+adj,kk) = fsc(si(id),jj+adj,kk)+1;
                   end
               end
           else
               %adj = 0;
               warning('No lambdas returned any elastic net coefficients in this run. It is difficult to select a series of lambda values that fit all of your data. Try defining lambda series *outside* the subsampling procedure (if you are not doing so already).')
           end
           
           % now count selected feats
%            if strcmpi(options.outlier,'none') || (strcmpi(options.outlier,'inside') && strcmpi(options.outlierDir,'rows'))
%               id = find(lsB(:,1)~=0);
%               if ~isempty(id)
%                   fsc(id,adj+1,kk) = fsc(id,adj+1,kk)+1;
%               end
%                %                for jj = 1:size(lsB,2)
% %                    if ~isempty(lsB)
% %                        id = find(lsB(:,jj)~=0);
% %                        fsc(id,jj+adj,kk) = fsc(id,jj+adj,kk)+1;
% %                    end
% %                end
%            else
%                s = 1:size(X,2);
%                s(oid{i}) = [];
%                for jj = 1:size(lsB,2)
%                    if ~isempty(lsB)
%                        id = find(lsB(:,jj)~=0);
%                        fsc(s(id),jj+adj,kk) = fsc(s(id),jj+adj,kk)+1;
%                    end
%                end
%            end
       end
    end
    
    % start linear regression...
    if strcmpi(options.selAlgo,'corr')
        [rV,pV] = corr(Xtmp,Ytmp,'rows','pairwise');
        if ~options.filter
            id = find(pV <= options.filterThresh);
        else
            [~,idtmp] = sort(abs(rV),'descend');
            id = idtmp(1:options.maxVars);
        end
        fsc(si(id)) = fsc(si(id))+1;
        if ~options.filter
            empMaxVars = mean([empMaxVars mean(length(id))],'omitnan');
        end
    end

    if strcmpi(options.selAlgo,'robustLR')
        if ~options.filter
            id = [];
            if ~options.parallel
                for kk = 1:size(Xtmp,2)
                    lm = fitlm(Xtmp(:,kk),Ytmp,'RobustOpts','on');
                    if lm.Coefficients.pValue(2) <= options.filterThresh
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
        else
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
        fsc(si(id)) = fsc(si(id))+1;
        if ~options.filter
            empMaxVars = mean([empMaxVars mean(length(id))],'omitnan');
        end
    end
    
    % start mrmr...
    if strcmpi(options.selAlgo,'mrmr')
        [id,scores{i}] = fsrmrmr(Xtmp,Ytmp);
        fsc(si(id(1:options.maxVars))) = fsc(si(id(1:options.maxVars)))+1;
    end
    
    % start ftest...
    if strcmpi(options.selAlgo,'ftest')
        [id,scores{i}] = fsrftest(Xtmp,Ytmp);
        fsc(si(id(1:options.maxVars))) = fsc(si(id(1:options.maxVars)))+1;
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
                fsc(si(id{jj,kk}(1:options.maxVars)),jj,kk) = fsc(si(id{jj,kk}(1:options.maxVars)),jj,kk)+1;
            end
        end
    end
    
    % start nca...
    if strcmpi(options.selAlgo,'nca')
        lam = options.lam;
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
        if options.filter
            for jj = 1:length(lam)
                fsc(s(id{jj}(1:options.maxVars))) = fsc(s(id{jj}(1:options.maxVars)))+1;
            end
        else
            for jj = 1:length(lam)
                idtmp = find(scores{jj} > options.filterThresh);
                anm(jj,1) = length(idtmp);
                fsc(si(idtmp)) = fsc(si(idtmp))+1;
            end
            empMaxVars = mean([empMaxVars mean(anm)],'omitnan');
        end
    end
    
    % start gpr...
    if strcmpi(options.selAlgo,'gpr')
        lam = options.lam;
        for jj = 1:size(lam,1)
            mdl = fitrgp(Xtmp,Ytmp,'KernelFunction','ardsquaredexponential', ...
                'Optimizer','lbfgs','FitMethod','sr','PredictMethod','fic','Standardize',stnd,'Regularization',lam(:,jj));
            sl = mdl.KernelInformation.KernelParameters(1:end-1);
            weights = exp(-sl); % Predictor weights
            scores{jj} = weights/sum(weights); % Normalized predictor weights
            [~,id{jj}] = sort(scores{jj},'descend');
        end
        for jj = 1:size(lam,1)
            fsc(si(id{jj}(1:options.maxVars))) = fsc(si(id{jj}(1:options.maxVars)))+1;
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
                    fsc(si(id{jj}(1:options.maxVars))) = fsc(si(id{jj}(1:options.maxVars)))+1;
        end
    end
end

% get max empirical probabilities...
fsc = fsc./options.rep;
fscmx = squeeze(max(fsc,[],[2:length(size(fsc))]));

% display effective maxVars
if strcmpi(options.selAlgo,'en')
    empMaxVars = mean(empMaxVars,'omitnan');
    disp(['Empirical maxVars (actual average # of selected features across parameters) was: ' num2str(empMaxVars)])
end

% get probability threshold for stable set if it was not passed in...
usrt = true;
if isempty(options.thresh)
    if ~isempty(options.fwer)
        tmpii = linspace(0.0001,size(X,2)/3,100000);
        disp(['Finding number of false positives that should give FWER of : ' num2str(options.fwer)])
        for i = 1:length(tmpii)
            n1 = ((((empMaxVars.^2)/size(X,2))/tmpii(i))+1)/2;
            tmpi(i,1) = (1/((2*n1)-1))*((empMaxVars.^2)/size(X,2))/empMaxVars;
        end
        id = find(tmpi < options.fwer,1,'last');
        if ~isempty(id)
            options.numFalsePos = tmpii(id);
        else
            options.numFalsePos = NaN;
             warning('It was not possible to find a number of false positives that would ensure selected fwer. Thresh will be set to 1.')
        end
        disp(['Found a good number of false positives : ' num2str(options.numFalsePos)])
    end
    if ~isnan(options.numFalsePos)
        options.thresh = ((((empMaxVars.^2)/size(X,2))/options.numFalsePos)+1)/2;
    else
        options.thresh = 1;
    end
    usrt = false;
    disp(['Probability threshold for features to enter stable set was calculated to be: ' num2str(options.thresh)])
end

if options.thresh > 1 && usrt % if threshold is not a proportion we assume you want a fixed set of selected features = threshold
    [~,fk] = maxk(fscmx,round(options.thresh));
    warning('Your threshold appears to be > 1 so we are assuming you want to select a fixed number of features as indicated by thresh. FWER no longer applies!!')
else
    fk = find(fscmx > options.thresh);
end

% show effective FWER
ep = ((1/((2*options.thresh)-1))*((empMaxVars.^2)/size(X,2)))/empMaxVars;
disp(['Effective FWER p-value is: ' num2str(ep) '. This may be slightly higher than your selected fwer due to rounding. Discrepancy may be higher when maxVars is lower.'])
disp(['Number of features that survived effective FWER p-value: ' num2str(length(fk))])

% output
maxVars = options.maxVars;
alpha = options.alpha;
thresh = options.thresh;
numFalsePos= options.numFalsePos;
