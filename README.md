# StabilitySelection
Perform stability selection in matlab using a variety of feature selection methods. You can also perform some outlier detection either inside the subsampling scheme, or prior to subsampling. 

Here is a very simple example for getting started. We will assume that you have already loaded into matlab a matrix called X that contains the samples of your data as rows (n) and the features as columns (p). We will also assume that you have already loaded in an n x 1 vector called y which contains the responses you would like to predict or explain using X.

# First, let's partition our data into 10 folds. Within each fold, we will find a stable set of features using stabSel. We will then use those features to predict our response variable.
c = cvpartition(length(y),'Kfold',10); % This partitions our data into 10 folds.
for i = 1:c.NumTestSets % now we will loop through our folds
    disp(['Working on test set ' num2str(i) ' of ' num2str(c.NumTestSets)])
    trX = X(training(c,i),:); % this determines our training data for fold i
    trY = y(training(c,i)); % this determines our training responses for fold i
    teX = X(test(c,i),:); % this determines our test data for fold i
    teY = y(test(c,i)); % this determines our test responses for fold i. We will predict this value used teX and see if feature selection improves our basic model.
    [idx{i},~,~,~,~,~,~] = stabSel2(trX,trY,'selAlgo','ftest','rep',100,'thresh',0.5,'prop',0.5,'verbose',true); % We perform stability selection here. Our feature selection algorithm is the F-test (technically, this is a filter). We will subsample 50% of our training data and on each subsample we will rank our features using the F-test. We will then select a number of variables automatically determined by stabSel to keep FDR at p < 0.05. After going through 100 subsamples, we will put all features that were selected in more than 50% of the subsamples into our stable set, which is output as idx{i}.
    mdl = fitrlinear(trX(:,idx{i}),trY); % now we will train a linear regression model to predict y using X, but we restrict X to only include those features that were selected. 
    yhat(test(c,i),1) = predict(mdl,teX(idx{i})); % here are the predictions of this model on our left out data.
    mdl = fitrlinear(trX,trY); % now we repeat this processs, but we use ALL of the features in our data
    yhat2(test(c,i),1) = predict(mdl,teX(idx{i}));
end

% Finally, we can compare and see if feature selection using stabSel made a difference to our model's performance. Here we use a simple example, the F-test. However, stabSel can also perform much more robust feature selection, like an adaptive elastic net (see stabSel.m for more details). 
corr(y,yhat) % this is the model w/feature selection
corr(y,yhat2) % this is the model w/out feature selection
