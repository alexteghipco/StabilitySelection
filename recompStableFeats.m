function [fk] = recompStableFeats(probs,selVars,nfp)
% This function is used to recompute stable features using a new number of
% false positives than originally used when calling stabSel.m. 
%
% Inputs: 
%   probs -- empirical probability of feature selection (n x 1 where
%   n is the number of features). This is fsc in stabSel.m.
% 
%   selVars -- average number of features selected across
%   resampled/perturbed datasets. This is empMaxVars in stabSel.m.
%
%   nfp -- number of false positives in the stable set. This is numFalsePos
%   in stabSel.m.
%
% Outputs: 
%   fk -- stable features based on above settings

fk = find(probs > (((((mean(selVars).^2)/length(probs))/nfp)+1)/2));

