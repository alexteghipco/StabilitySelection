# StabilitySelection
This is a package for performing stability selection in MATLAB, using a variety of feature selection/ranking algorithms that come packaged with MATLAB's statistics and machine learning toolbox. This comes with a variety of live code tutorials that show how to: 1) run stabSel, 2) properly set up cross-validation and nested cross-validation schemes, 3) build predictive models using other algorithms available in MATLAB's statistics and machine learning toolbox after performing stability selection. 

For more info that may be useful to you, see our preprint. We showcase some results that use this toolbox. But mainly, we demonstrate that stability selection can be a powerful approach for feature selection under a variety of selection algorithms when sample sizes are relatively small (as they tend to be in clinical neuroimaging studies): https://www.medrxiv.org/content/10.1101/2023.09.13.23295370v1

## Why this kind of feature selection?
Algorithms that cultivate feature sparsity demonstrably tradeoff stability. This is a problem because you can build a model that finds a sparse feature set for making relatively good predictions about the data, and yet the model would be difficult to interrogate and understand because feature contributions towards predictions may have a substantial amount of instability.

Stability selection can help you identify stable features by wrapping around your favorite algorithm for feature selection. It repeatedly performs feature selection on perturbed versions of your dataset for each possible hyperparameter associated with your chosen feature selection algorithm. Using information about the consistency of feature selection, it then identifies a set of stable features while ensuring a user-specified per-family error rate (or per-comparison error rate) that controls the upper bound on the number of false positives in the set. Once you have extracted the most stable features in your dataset, you can then confirm that these features are predictive by building additional predictive models and testing them on out-of-sample data. 

In other words, this tool will help you find stable features that you can then whittle down to features that are both stable and substantively predictive.

## Installation
You must have the MATLAB statistics and machine learning toolbox installed. Some ranking algorithms require the Bioinformatics toolbox. It's best to use the most recent version of MATLAB that you have access to but this package tries to be smart about letting you use stability selection even if you have access to older versions of MATLAB (i.e., it will warn you if you select an algorithm that's unavailable to your MATLAB version). To use the package, simply add it to your MATLAB path.

## Getting started
For a better understanding of stabSel.m and how to use it, please see the first live code tutorial: Tutorial1_PredictSongYear.mlx. Here, you will attempt to predict the year in which a song was produced using audio features. This example explains stability selection and some of the math behind how we can ensure a certain per-family error rate in our stable set of features. The examples showcase regression techniques in stabSel and will help you understand and code a properly setup nested cross validation scheme. Example 1 is particularly important. 

A second live code tutorial (Tutorial2_PredictMOCA.mlx) shows how to predict at-risk MOCA scorers (Montreal Cognitive Assessment) from narrative discourse measures. This tutorial showcases more basic classification techniques in stabSel. It also demonstrates how to build more complex predictive models after performing stability selection in a nested CV scheme (e.g., random forest, SVMs, etc). This example demonstrates settings in which stabSel massively boosts performance and settings where it does very little because the predictive model is already doing a good job of modeling signal features. This tutorial also contains code that may be helpful for tracking/visualizing hyperparameter selection across your nested CV scheme. 

Tutorial3_PredictLSM.mlx is a work-in-progress. In this example, we will use more complex classification for predicting aphasia severity from lesion location. However, this data is still being prepared so for now, this tutorial exists to introduce you to how you should load in brain data for analysis with stabSel.

Note: I am continuing to develop and add features to this package. If you encounter any issues, or if you have any suggestions, please let me know at: alex.teghipco@sc.edu. 

Here are some coming updates: 
- The documentation for stabSel.m is sprawling. This will move into a wiki for this repository and the documentation within the main function will be trimmed down 
- More options for multi-task learning
- Support for testing interactions in the context of algorithms that do not automatically do this
- Tutorials with brain data forthcoming

Supported regression algorithms: 
- [lasso](https://www.mathworks.com/help/stats/lasso.html)
- [elastic net](https://www.mathworks.com/help/stats/lasso.html)
- [adaptive lasso](https://www.tandfonline.com/doi/abs/10.1198/016214506000000735)
- [adaptive elastic net](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2864037/)
- [random forest (permuted oob error)](https://www.mathworks.com/help/stats/select-predictors-for-random-forests.html)
- [partial least squares (with Variable Importance Projection) <-- supports multi-task learning](https://www.mathworks.com/help/stats/plsregress.html)
- [gaussian process regression (ardsquaredexponential kernel)](https://www.mathworks.com/help/stats/fsrnca.html#bveghxy-3)
- [neighborhood component analysis](https://www.mathworks.com/help/stats/fsrnca.html#bveghxy-3)
- [F-test](https://www.mathworks.com/help/stats/fsrftest.html)
- [Rrelieff](https://www.mathworks.com/help/stats/relieff.html)
- [MRMR](https://www.mathworks.com/help/stats/fsrmrmr.html)
- [pearson correlation (p-value to retain or coeff. for ranking)](https://www.mathworks.com/help/stats/corr.html)
- [robust linear regression (p-value to retain or coeff. for ranking)](https://www.mathworks.com/help/stats/fitlm.html)

Supported classification algorithms:
- [random forest](https://www.mathworks.com/help/stats/select-predictors-for-random-forests.html)
- [neighborhood component analysis](https://www.mathworks.com/help/stats/fscnca.html)
- [Rrelieff](https://www.mathworks.com/help/stats/relieff.html)
- [MRMR](https://www.mathworks.com/help/stats/fscmrmr.html)
- [t-test or wilcoxon test (ranking based on two-sample test statistic; bioinformatics toolbox)](https://www.mathworks.com/help/bioinfo/ref/rankfeatures.html)
- [bhattacharyya: minimum attainable classification error  (ranking; bioinformatics toolbox)](https://www.mathworks.com/help/bioinfo/ref/rankfeatures.html)
- [entropy or Kullback-Leibler divergence (ranking; bioinformatics toolbox)](https://www.mathworks.com/help/bioinfo/ref/rankfeatures.html)
- [area between the empirical ROI curve and random classifier slope (ranking; bioinformatics toolbox)](https://www.mathworks.com/help/bioinfo/ref/rankfeatures.html)

Why are these not implemented?:
- sequential feature selection: computational costs are surprisingly high
- laplacian scoring: will be implemented, but it's unsupervised so does not cleanly fit into the categories above
- chi squared: will be implemented in the future
- lasso classification: will be implemented
- generalized lasso: will be implemented
- LDA for classification: will be implemented

For methods, see: 
Meinshausen, N., & Bühlmann, P. (2010). Stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4), 417-473.
