# StabilitySelection
Perform stability selection in matlab using a variety of feature selection methods. You can also perform some outlier detection either inside the resampling scheme, or prior to resampling. 

Refer to stabSel.m for documentation on both the code and the stability selection method. Tutorial.mlx contains several thorough examples for using stabSel to shrink your feature space, and to discover features that significantly predict some variable you are interested in.

Disclaimer: stabSel implements a ton of options, including many different ways to select features within the stability selection framework. The defaults have been tested fairly well and work fine. Lots of the combinations of options available to you have been debugged and I'm continuing testing (will update this readme when this is completed). However, the sheer number of options available means that you are likely to use stabSel in a way I have not anticipated yet. Send any issues/comments to: alex.teghipco@sc.edu
