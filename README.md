# StabilitySelection
This is a package for performing stability selection in MATLAB, using a variety of feature selection/ranking algorithms that come packaged with MATLAB's machine learning toolbox. This comes with a variety of live code tutorials that show how to: 1) run stabSel, 2) properly set up cross-validation and nested cross-validation schemes, 3) build predictive models using other algorithms available in MATLAB's machine learning toolbox after performing stability selection.

Stability selection can help you understand which features are reliably predictive. We have several tutorials below that demonstrate it can also help you build more generalizable models. Here is a visual demonstration showing how stability selection lets elastic net converge on a more consistent set of brain lesion features that predict aphasia severity after stroke (brighter = more consistent across models, yellow = perfect consistency):
<p align="center">
  <kbd><img width="450" height="250" src="https://i.imgur.com/zRTwqEm.gif"/></kbd>
</p>

For a better general understanding of stabSel and how to use it, I recommend starting with Tutorial1_PredictSongYear.mlx, where you will try predicting the year a song was produced using audio features. Example 1 may be clearer for beginners, and the remaining examples build on this one. This tutorial showcases regression techniques in stabSel and provides an introduction for how you can properly setup nested CV schemes when using matlab's machine learning toolbox. 

Tutorial2_PredictMOCA.mlx attempts to predict at-risk moca scorers from discourse measures. This tutorial showcases more basic classification techniques in stabSel. It also showcases how to build more complex predictive models after performing stability selection in a nested CV scheme (e.g., random forest, SVMs, etc). This example demonstrates settings in which stabSel massively boosts performance and settings where it does very little because the predictive model is already doing a good job of modeling signal features. This tutorial also contains code that may be helpful for tracking/visualizing hyperparameter selection across your nested CV scheme. 

* This tutorial covers analyses I performed for: [Wilson, S. C., Teghipco, A., Sayers, S., Newman-Norlund, R., Newman-Norlund, S., & Fridriksson, J. (2024). Story recall in peer conflict resolution discourse task to identify older adults testing within range of cognitive impairment. American Journal of Speech-Language Pathology. https://doi.org/10.1044/2024_AJSLP-24-00005](https://pubs.asha.org/doi/full/10.1044/2024_AJSLP-24-00005)

Tutorial3_PredictLSM.mlx is the most extensive, describing injection of stability selection into multivariate lesion symptom mapping. We are calling this technique stable multivariate lesion mapping (sMLSM) because identifying good, predictive, features is the starting point rather than the end-point of the analysis (it's more common for multivariate lesion mapping studies to treat this as an end-point). In this tutorial, we use our publicly available database of chronic stroke patients with aphasia to show that SVM prediction of aphasia severity from lesion location is better when we include stability selection to identify more stable predictors. We show that stability selection permits SVM to focus on more complex patterns of brain damage and that it helps lower variance in feature importance. This tutorial contains code for importing brain data into matlab, setting up our sMLSM pipeline, interpreting the outputs of this pipeline, and generating pretty brain figures using our other repositories. It also shows you how you might tune the per family error rate in stability selection to attain good models without having to preselect this value (albeit in a relatively computationally expensive way, we will update this to be more efficient). 

* This tutorial is a companion to the following article: [Teghipco A, Newman-Norlund R, Gibson M, et al. Stable multivariate lesion symptom mapping. Aperture Neuro. 2024;4. doi:10.52294/001c.117311](https://apertureneuro.org/article/117311-stable-multivariate-lesion-symptom-mapping)

Note: I am continuing to develop and add features to this package. If you encounter any issues, or if you have any suggestions, please let me know at: alex.teghipco@sc.edu. There are some feature selection algorithms that have been subjected to less debugging because they take much longer to run so if you run into any issues with them, feedback is extra greatly appreciated (e.g., RFs and GPRs).

Here are some coming updates: 
- The documentation for stabSel.m is sprawling. This will move into a wiki for this repository and the documentation within the main function will be trimmed down 
- Tuning PFER inside of bayesopt call
- Outlier removal is supported but the documentation is currently lacking and will be updated
- More options for multi-task learning
- Support for testing interactions in the context of algorithms that do not automatically do this

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

Stability selection is based on: 
Meinshausen, N., & Bühlmann, P. (2010). Stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4), 417-473.
