function [Xroi, unROI, id] = roiBin(X,roi)
% Gets proportion of an ROI that is damaged in a lesion mask, but operates
% over an entire atlas and a matrix of participants' lesion data.
%
% Inputs------------------------------------------------------------------ 
%   X: brain data, where rows are subjects and columns are features or
%   voxels. Here, we assume the values are 0s (preserved) or 1s (lesioned)
%
%   roi: atlas image that matches the shape of brain images in X.
%   Vectorized.
% 
% Outputs------------------------------------------------------------------ 
%   Xroi: rows are subjects, columns represent rois. Values inside show the
%   proportion of the roi that was damaged in the individual (i.e., the
%   percentage of voxels in the ROI that were 1s)
%
%   unROI: a vector showing the ROI index (i.e., what integer or value in
%   the passed in roi map) correspond to each column in Xroi
%
%   id: voxel indices for this ROI

if size(roi,1) ~= size(X,2)
    error('Rows in roi should be identical in length to columns in X')
end

unROI = unique(roi); % get all unique ROIs
unROI = setdiff(unROI,0); % remove 0

for i = 1:length(unROI)
    id{i} = find(roi == unROI(i)); % roi voxels
    for j = 1:size(X,1)
        Xroi(j,i) = sum(X(j,id{i}))/length(id{i});
    end
end