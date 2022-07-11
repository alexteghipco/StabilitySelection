function [pTvaf,pExp,tvafPerm,tvaf,vaf,vafs,vafp,vafpPerm,vafPerm,explainedPermAll,coeff,score,latent,tsquared,explained,mu] = permutedPCA(inData,numPerms,permType)
% [p,coeff,score,latent,tsquared,explained,mu] = permutedPCA(inData,numPerms,permType)
%
% permutedPCA computes the total variance accounted for by a PC model
% (i.e., including PCs 1...c) and tests the computed values against a null
% distribution built over the number of permutation specified in numPerms.
% 
% It is assumed that inData is an n x p matrix with n observations and p
% variables. 
%
% Output p contains the significance of each model (e.g., model that
% includes just component 1 is specified in p(1) and model that includes
% all possible components 1...n is specified in p(end). 
%
% Other outputs include the coefficients or loadings for each PC (coeff),
% the PCs themselves (score), and the % of total variance they explain
% (explained). For more information about outputs see MATLAB's pca
% function.
%
% About choosing the number of permutations: For weak effects, or for reporting
% p-values in publications, Buja and Eyuboglu (1992) suggest using 99 or
% 499 permutations, because permutation tests with a smaller number of
% permutations than 99 have too little power.
%
% For methods see: "STATISTICAL SIGNIFICANCE OF THE CONTRIBUTION OF
% VARIABLES TO THE PCA SOLUTION: AN ALTERNATIVE PERMUTATION STRATEGY" by
% LINTING et al (2011), published in psychometrika
rng('shuffle');
verbose = false;
parallel = true;
tvaf = [];
vaf = [];
vafp = [];
vafpPerm = [];
vafPerm = [];
pTvaf = [];
pExp = [];
vafs = [];
tvafPerm = [];
explainedPermAll = [];

%% 1. Get real data
disp(['Input data has ' num2str(size(inData,2)) ' variables']);
disp('Getting pca results for real data...');
[coeff,score,latent,tsquared,explained,mu] = pca(inData);
% TVAF (variance-accounted-for in the entire dataset) is equal to the
% sum of the eigenvalues of the first c components
switch permType
    case 'whole model'
        for c = 1:length(latent)
            tvaf(c,1) = sum(latent(1:c));
        end
    case 'variables'
        [vaf,vafp] = corr(inData,score);
end

%% 2. Establish null distribution
% To establish a null distribution, first, the correlational structure of
% the observed data is destroyed by randomly rearranging the values within
% each variable (independent of the other variables).
if ~parallel && verbose
    w = waitbar(0,'Starting permutations...');
end
switch permType
    case 'whole model'
        if ~parallel
            for permi = 1:numPerms
                if verbose
                    waitbar(permi/numPerms,w,['Working on permutation ' num2str(permi)]);
                end
                dataPerm = premvars(inData);
                [coeffPerm,scorePerm,latentPerm,tsquaredPerm,explainedPermAll(:,permi),muPerm] = pca(dataPerm,'Economy',false);
                [tvafPerm(:,permi)] = tvafpermcalc(latentPerm); % get tvaf
            end
        else
            parfor permi = 1:numPerms
                dataPerm = premvars(inData);
                [coeffPerm,scorePerm,latentPerm,tsquaredPerm,explainedPermAll(:,permi),muPerm] = pca(dataPerm,'Economy',false);
                [tvafPerm(:,permi)] = tvafpermcalc(latentPerm); % get tvaf
            end
        end
    case 'variables'
        for v = 1:size(inData,2) % permute the observations for each variable separately
            waitbar(v/size(inData,2),w,['Working on variable ' num2str(v)]);
            %disp(['Working on variable index ' num2str(v) '...'])
            for permi = 1:numPerms
                %disp(['Working on permutation number ' num2str(permi) ' of ' num2str(numPerms) ' ...' ])
                dataPerm = inData;
                dataPerm(:,v) = dataPerm(randperm(size(inData,1)),v);
                %disp(['Getting pca results for permutation' num2str(permi) ' ...'])
                [coeffPerm,scorePerm,latentPerm,tsquaredPerm,explainedPerm,muPerm] = pca(dataPerm,'Economy',false);
                disp('Geneating loadings for permutation ...')
                [vafPerm_tmp,vafpPerm_tmp] = corr(dataPerm,scorePerm);
                vafPerm(:,:,permi) = vafPerm_tmp;
                vafpPerm(:,:,permi) = vafpPerm_tmp;
            end
        end
end

%% 3. Get p-value for components
% Calculating the proportion of the values in the permutation
% distribution that is equal to or exceeds the observed statistic (the
% p-value). The p-value is then, as usual, compared to a prechosen
% significance level ?: If p < ?, the result is called significant. A
% p-value is computed as p = (q+1)/(P +1), with q the number of times a
% statistic from the permutation distribution is greater than or equal to
% the observed statistic, and P the number of permutations (Buja &
% Eyuboglu, 1992; Noreen, 1989).

switch permType
    case 'whole model'
        for c = 1:length(latent)
            tmpIdx = find(tvafPerm(c,:) > tvaf(c));
            pTvaf(c,1) = (length(tmpIdx)+1)/(numPerms +1);
        end
        for c = 1:length(latent)
            tmpIdx = find(explainedPermAll(c,:) > explained(c));
            pExp(c,1) = (length(tmpIdx)+1)/(numPerms +1);
        end
    case 'variables'
        vafs = vaf.^2;
        vafPermS = vafPerm.^2;
        toFix = isnan(vafs);
        vafs(isnan(vafs))=0;
        vafPermS(isnan(vafPermS))=0;
        
        for v = 1:size(inData,2)
            for c = 1:size(scorePerm,2)
                tmpIdx = find(vafPermS(v,c,:) > vafs(v,c));
                pVaf(v,c) = (length(tmpIdx)+1)/(numPerms +1);
            end
        end
        pVaf(toFix) = 1;
end
if ~parallel && verbose
    close(w)
end

function dataPerm = premvars(inData)
for vari = 1:size(inData,2) % permute the observations for each variable independently
    dataPerm(:,vari) = inData(randperm(size(inData,1),size(inData,1)),vari);
end

function [tvafPerm] = tvafpermcalc(latentPerm)
for c = 1:length(latentPerm)
    tvafPerm(c,1) = sum(latentPerm(1:c));
end