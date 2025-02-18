function [Xroi, unROI, id] = roiBin(X,roi)
% Gets the mean of roi voxels as defined in unROI within the X matrix of
% brain data.
%
% Inputs------------------------------------------------------------------ 
%   X: brain data, where rows are subjects and columns are features or
%   voxels. Here, we assume the values are 0s (preserved) or 1s (lesioned)
%   but they need not be.
%
%   roi: atlas image that matches the shape of brain images in X.
%   Vectorized.
%
% Outputs------------------------------------------------------------------ 
%   Xroi: rows are subjects, columns represent rois. Values inside show the
%   proportion of the roi that was damaged in the individual if X
%   represented binary lesion masks, or the mean of whatever values are
%   represented in X otherwise
%
%   unROI: a vector showing the ROI index (i.e., what integer or value in
%   the passed in roi map) correspond to each column in Xroi
%
%   id: voxel indices for this ROI
%

if size(roi,1) ~= size(X,2)
    error('Rows in roi should be identical in length to columns in X')
end

unROI = unique(roi); % get all unique ROIs
unROI = setdiff(unROI,0); % remove 0

for i = 1:length(unROI)
    id{i} = find(roi == unROI(i)); % roi voxels
    for j = 1:size(X,1)
        Xroi(j,i) = nansum(X(j,id{i}))/length(id{i});
    end
end
