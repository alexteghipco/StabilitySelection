function [fileNames,X,bid,initSz] = brainLoad(folder,mask)
% This function loads in all of the .nii or .nii.gz files within a folder,
% constraining the portion of the images being extracted to a mask if
% supplied. The way files are loaded is by first trying to use matlab's
% image processing toolbox (niftiread.m). If that fails, then we try to
% load using the freesurfer load_nifti.m function. If that fails, then we
% try to load using SPM. Make sure one of these sets of files is in your
% matlab path!
%
% Inputs: 
% folder : leave empty for UI folder selection, otherwise it is a string
% path to your folder containing .nii or .nii.gz files
%
% mask: same as above for folder. Note, if you pass this in as empty, you
% will trigger the file selection UI. Just hit cancel if you do not intend
% to use a mask.
%
% Outputs:
% fileNames: the name of the file, which forms every col of the output X
% X: each row is a voxel, each col is a file (i.e., subject)
% bid: voxel identities from the whole brain image that we used as a mask
% (short for brain IDs)
% initSz: the initial size of the brain data before it is vectorized
%
% Example calls:
% [fileNames,X,bid,initSz] = brainLoad([],[])
%[fileNames,X,bid,initSz] = brainLoad(['C:\Users\alext\toAnalyze'],['C:\Users\alext\Downloads\MNI152_2mm.nii.gz'])

% permitted extensions
exta = {'*.nii.gz','*.nii'};

% define file separator based on system
if ispc
    s = '\';
elseif isunix || ismac
    s = '/';
else
    error('Cannot identify your system...pc or mac or unix?')
end

% ask user where to pull nifti files from and mask
if isempty(folder)
    disp('Please select a folder that contains the nifti files you want to import. It must not have any other nifti files!')
    folder = uigetdir([pwd],'Please select your folder with .nii or .nii.gz files');
    if folder == 0
        error('You must select a folder! Looks like you hit cancel')
    end
end
if isempty(mask)
    disp('Please select your brain mask nifti file. All nonzero values in the mask will be treated as areas of the brain you want to keep.')
    [mask1,mask2] = uigetfile(exta,'Please select your brain mask. All nonzero values in the mask will be treated as areas of the brain you want to keep.',pwd);
    mask = [mask2 mask1];
end

% load in your mask
try
    %disp('Trying to load using matlabs niftiread.m')
    tmpM = niftiread(mask);
    bid = find(tmpM ~= 0);
catch
    try
        %disp('Did not work! Trying to load using freesurfers load_nifti.m')
        tmp = load_nifti(mask);
        tmpM = tmp.vol;
        bid = find(tmpM ~= 0);
    catch
        try
            %disp('Did not work! Trying to load using spm load_nifti.m')
            tmp1 = spm_vol(mask);
            tmpM = spm_read_vols(tmp1);
            bid = find(tmpM ~= 0);
        catch
            bid = [];
            disp(['No loadable mask supplied Proceeding without the mask...if this is not expected behavior please make sure spm_read_vols or load_nifti or niftiread is in your path' ])
        end
    end
end

% combine files of both kinds of extensions: nii or nii.gz
files = [];
for i = 1:length(exta)
    files = [files; dir(fullfile([folder], [exta{i}]))];
end

% now loop through and load
for i = 1:length(files)
    disp(['Working on file ' num2str(i) ' of ' num2str(length(files))]);
    disp(['File is: ' files(i).name])
    try
        %disp('Trying to load using matlabs niftiread.m')
        tmp = niftiread([files(i).folder s files(i).name]);
        initSz = size(tmp);
    catch
        try
            %disp('Did not work! Trying to load using freesurfers load_nifti.m')
            tmp = load_nifti([files(i).folder s files(i).name]);
            tmp = tmp.vol;
            initSz = size(tmp);
        catch
            try
                %disp('Did not work! Trying to load using spm load_nifti.m')
                tmp1 = spm_vol([files(i).folder s files(i).name]);
                tmp = spm_read_vols(tmp1);
                initSz = size(tmp);
            catch
                disp(['Could not load in file! ' files(i).folder s files(i).name])
                error('Double check that nifti is unzipped if necessary based on the software you are trying to use')
            end
        end
    end
    if ~isempty(bid)
        try
            X(:,i) = tmp(bid);
        catch
            error('Check your mask, it seems like it might be a different shape than your files that you are trying to load in.')
        end
    else
        X(:,i) = tmp(:);
    end
end
fileNames = {files.name};
