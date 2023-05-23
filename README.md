# StabilitySelection
Perform stability selection in matlab using a variety of feature selection methods. You can also perform some outlier detection either inside the resampling scheme, or prior to resampling. 

Stability selection identifies a stable set of features by repeatedly resampling your data and performing feature selection each time using some method (traditionally LASSO using lots of different regularization parameters). It can be folded into a cross-validation scheme, and provides a way to try to implement error control while selecting features. 

Refer to stabSel.m for documentation on both the code and the stability selection method. Tutorial.mlx contains several thorough examples for using stabSel to shrink your feature space, and to discover features that significantly predict some variable you are interested in. This only has regression problems but will be updated to cover classification and multi-task learning (for now refer to stabSel.m documentation for these features)

Disclaimer: stabSel implements a ton of options, including many different ways to select features within the stability selection framework. The defaults have been tested fairly well and work fine. Lots of the combinations of options available to you have been debugged and I'm continuing testing. However, the sheer number of options available means that you are likely to use stabSel in a way I have not anticipated yet. Also note, group lasso is currently experimental. Send any issues/comments to: alex.teghipco@sc.edu

For methods, see: 
Meinshausen, N., & Bühlmann, P. (2010). Stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4), 417-473.

Shah, R. D., & Samworth, R. J. (2013). Variable selection with error control: another look at stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 75(1), 55-80.
