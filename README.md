# StabilitySelection
This is package for performing stability selection in MATLAB, using a variety of feature selection/ranking algorithms that come packaged with MATLAB's machine learning toolbox. This comes with a variety of live code tutorials that show how to: 1) run stabSel, 2) properly set up cross-validation and nested cross-validation schemes, 3) build predictive models using other algorithms available in MATLAB's machine learning toolbox after performing stability selection.

For a better understanding of stabSel and how to use it, I recommend starting with Tutorial1_PredictSongYear.mlx, where you will try predicting song years using audio features. Example 1 is particularly important, the remaining examples are more supplemental and build on example 1. This tutorial showcases regression techniques in stabSel and provides an introduction for how you can properly setup nested CV schemes when using matlab's machine learning toolbox. 

Tutorial2_PredictMOCA.mlx attempts to predict at-risk moca scorers from discourse measures. This tutorial showcases more basic classification techniques in stabSel. It also showcases how to build more complex predictive models after performing stability selection in a nested CV scheme (e.g., random forest, SVMs, etc). This example demonstrates settings in which stabSel massively boosts performance and settings where it does very little because the predictive model is already doing a good job of modeling signal features. This tutorial also contains code that may be helpful for tracking/visualizing hyperparameter selection across your nested CV scheme. 

Tutorial3_PredictLSM.mlx shows how to fold stability selection into a multivariate lesion symptom mapping model to predict aphasia severity from lesion location. Here, we benchmark this "stable multivariate lesion symptom mapping" model against standard multivariate lesion symptom mapping, as well as models that only rely on lesion size.

Note: I am continuing to develop and add features to this package. If you encounter any issues, or if you have any suggestions, please let me know at: alex.teghipco@sc.edu. 

Here are some coming updates: 
- The documentation for stabSel.m is sprawling. This will move into a wiki for this repository and the documentation within the main function will be trimmed down 
- More options for multi-task learning
- Support for testing interactions in the context of algorithms that do not automatically do this
- Tutorials with brain data forthcoming

Supported regression algorithms: 
- lasso
- elastic net
- adaptive lasso
- adaptive elastic net
- random forest (permuted oob error)
- partial least squares (with Variable Importance Projection) <-- supports multi-task learning
- gaussian process regression (ardsquaredexponential kernel)
- neighborhood component analysis
- Rrelieff
- MRMR
- correlation (p-value to retain or coeff. for ranking)
- robust linear regression (p-value to retain or coeff. for ranking)

Supported classification algorithms:
- random forest
- neighborhood component analysis
- Rrelieff
- MRMR
- t-test or wilcoxon test (ranking based on two-sample test statistic)
- f-test (ranking based on test statistic)
- bhattacharyya: minimum attainable classification error  (ranking)
- entropy or Kullback-Leibler divergence (ranking)
- area between the empirical ROI curve and random classifier slope (ranking)

For methods, see: 
Meinshausen, N., & Bühlmann, P. (2010). Stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4), 417-473.
