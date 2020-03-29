%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% N-number of control subjects - MRI data and analysis %%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
theElapsedTime = tic;

MaskData = y_Read(mask);
MaskDataOneDim = reshape(MaskData,1,[]);
n_voxels = length(find(MaskData));

unit = strcmpi(resp,'No'); % unit==2 = local connectivity

if unit == 1
    %%% Find path where the MRI data is located, and also total number of controls
    filepath = y_rdir([controls_path '/*/*.nii']);
    
    if isempty(filepath) == 0
        infile = {filepath.name};
        
    else
        filepath = y_rdir([controls_path '/*.nii']);
        infile = {filepath.name};
        
    end
    
    controls_number = length(infile);
    
    fprintf('\n Running Statistical inference of single-subject MRI (own metric) with %d controls ... \n',controls_number)
    
    MRI_control_data_smooth_norm = zeros(controls_number,n_voxels); % pre-allocate arrays for smoothed and normalised MRI data (N-controls)
    
    parfor n = 1:length(infile)
        %%% Mask MRI data and write to nifti files - control subjects
        [MRI_control_data,Header] = y_Read(infile{n});
        
        MRI_control_data = MRI_control_data.*MaskData;
        
        y_Write(MRI_control_data,Header,['Control_000' num2str(n) '_mask.nii']);
        
        %%% Smooth data
        spm_smooth(['Control_000' num2str(n) '_mask.nii'],['Control_000' num2str(n) '_mask_smooth_' num2str(smt) 'mm.nii'],smt);
        
        %%% Load smoothed data
        MRI_control_data_smooth = y_Read(['Control_000' num2str(n) '_mask_smooth_' num2str(smt) 'mm.nii']);
        
        %%% Smoothed data; from 2D to 1D
        MRI_control_data_smooth = reshape(MRI_control_data_smooth,[],1)';
        MRI_control_data_smooth = MRI_control_data_smooth(logical(MaskDataOneDim));
        MRI_control_data_smooth(isnan(MRI_control_data_smooth)) = eps;
        MRI_control_data_smooth(isinf(MRI_control_data_smooth)) = eps;
        
        %%% Normalise smoothed data --> into a Gaussian distribution using a rank-based inverse normal transformation (Blom, 1958)
        MRI_control_data_smooth = tiedrank(MRI_control_data_smooth);
        MRI_control_data_smooth = (MRI_control_data_smooth - 3/8) ./ (numel(MRI_control_data_smooth) + 1/4);
        MRI_control_data_smooth_norm(n,:) = norminv(MRI_control_data_smooth,0,1);

        delete(['Control_000' num2str(n) '_mask.nii']);
        delete(['Control_000' num2str(n) '_mask_smooth_' num2str(smt) 'mm.nii']);
    end
else
    %%% Find path where the MRI data is located, and also total number of controls
    filepath = y_rdir([controls_path '/*/*.nii']);
    
    if isempty(filepath) == 0
        infile = {filepath.name};
        
    else
        filepath = y_rdir([controls_path '/*.nii']);
        infile = {filepath.name};sp
        
    end
    
    controls_number = length(infile);
    
    fprintf('\n Running Statistical inference of single-subject fMRI local connectivity (ReHo) with %d controls ... \n',controls_number)
    
    MRI_control_data_smooth_norm = zeros(controls_number,n_voxels); % pre-allocate arrays for smoothed and normalised MRI data (N-controls)
    
    parfor n = 1:length(infile)
        %%% Calculate fMRI local connectivity - control subjects
        [MRI_control_data,Header] = y_Read(infile{n});        
        MRI_control_data = y_reho(MRI_control_data, 27, mask, ['Control_000' num2str(n) '_mask.nii'], 0, '', 3, '', '', '', Header, 10);
        
        %%% Smooth data
        spm_smooth(['Control_000' num2str(n) '_mask.nii'],['Control_000' num2str(n) '_mask_smooth_' num2str(smt) 'mm.nii'],smt);
        
        %%% Load smoothed data
        MRI_control_data_smooth = y_Read(['Control_000' num2str(n) '_mask_smooth_' num2str(smt) 'mm.nii']);
        
        %%% Smoothed data; from 2D to 1D
        MRI_control_data_smooth = reshape(MRI_control_data_smooth,[],1)';
        MRI_control_data_smooth = MRI_control_data_smooth(logical(MaskDataOneDim));
        MRI_control_data_smooth(isnan(MRI_control_data_smooth)) = eps;
        MRI_control_data_smooth(isinf(MRI_control_data_smooth)) = eps;
        
        %%% Normalise smoothed data --> into a Gaussian distribution using a rank-based inverse normal transformation (Blom, 1958)
        MRI_control_data_smooth = tiedrank(MRI_control_data_smooth);
        MRI_control_data_smooth = (MRI_control_data_smooth - 3/8) ./ (numel(MRI_control_data_smooth) + 1/4);
        MRI_control_data_smooth_norm(n,:) = norminv(MRI_control_data_smooth,0,1);
        
        delete(['Control_000' num2str(n) '_mask.nii']);
        delete(['Control_000' num2str(n) '_mask_smooth_' num2str(smt) 'mm.nii']);
        
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Single clinical subject - MRI data and analysis %%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% same procedure as above (section A) therefore no comments here in section B. We proceed with comments in section C.
unit = strcmpi(resp,'No'); % unit==2 = local connectivity

if unit == 1   
    [MRI_single_subject_data,Header] = y_Read(clinical);
    [nDim1,nDim2,nDim3,~] = size(MRI_single_subject_data);
    MRI_single_subject_data = MRI_single_subject_data.*MaskData;
    
    y_Write(MRI_single_subject_data,Header,['Subject_' cell2mat(xx) '_masked.nii']);
    
    %%% Smooth data
    spm_smooth(['Subject_' cell2mat(xx) '_masked.nii'],['Subject_' cell2mat(xx) '_mask_smooth_' num2str(smt) 'mm.nii'],smt);
    
    %%% Load smoothed data
    MRI_single_subject_data_smooth = y_Read(['Subject_' cell2mat(xx) '_mask_smooth_' num2str(smt) 'mm.nii']);
    
    %%% Smoothed data; from 2D to 1D
    MRI_single_subject_data_smooth = reshape(MRI_single_subject_data_smooth,[],1)';
    MRI_single_subject_data_smooth = MRI_single_subject_data_smooth(logical(MaskDataOneDim));
    MRI_single_subject_data_smooth(isnan(MRI_single_subject_data_smooth)) = eps;
    MRI_single_subject_data_smooth(isinf(MRI_single_subject_data_smooth)) = eps;
    
    %%% Normalise smoothed data --> into a Gaussian distribution using a rank-based inverse normal transformation (Blom, 1958)
    MRI_single_subject_data_smooth = tiedrank(MRI_single_subject_data_smooth);
    MRI_single_subject_data_smooth = (MRI_single_subject_data_smooth - 3/8) ./ (numel(MRI_single_subject_data_smooth) + 1/4);
    MRI_single_subject_data_smooth_norm = norminv(MRI_single_subject_data_smooth,0,1);
    
    %%% Write smoothed and normalised nifti files for all control subjects
    out = zeros(size(MaskDataOneDim));
    out(1,find(MaskDataOneDim)) = MRI_single_subject_data_smooth_norm;
    out = reshape(out,nDim1,nDim2,nDim3);
    
    y_Write(out,Header,['Subject_' cell2mat(xx) '_ReHo_mask_smooth_' num2str(smt) 'mm_norm.nii']);
    
else
       
    [MRI_single_subject_data,Header] = y_Read(clinical);
    [nDim1,nDim2,nDim3,~] = size(MRI_single_subject_data);
   
    MRI_single_subject_data = y_reho(MRI_single_subject_data, 27, mask,['Subject_' cell2mat(xx) '_ReHo_masked.nii'], 0, '', 3, '', '', '', Header, 10);
        
    %%% Smooth data
    spm_smooth(['Subject_' cell2mat(xx) '_ReHo_masked.nii'],['Subject_' cell2mat(xx) '_ReHo_mask_smooth_' num2str(smt) 'mm.nii'],smt);
    
    %%% Load smoothed data
    MRI_single_subject_data_smooth = y_Read(['Subject_' cell2mat(xx) '_ReHo_mask_smooth_' num2str(smt) 'mm.nii']);
    
    %%% Smoothed data; from 2D to 1D
    MRI_single_subject_data_smooth = reshape(MRI_single_subject_data_smooth,[],1)';
    MRI_single_subject_data_smooth = MRI_single_subject_data_smooth(logical(MaskDataOneDim));
    MRI_single_subject_data_smooth(isnan(MRI_single_subject_data_smooth)) = eps;
    MRI_single_subject_data_smooth(isinf(MRI_single_subject_data_smooth)) = eps;
    
    %%% Normalise smoothed data --> into a Gaussian distribution using a rank-based inverse normal transformation (Blom, 1958)
    MRI_single_subject_data_smooth = tiedrank(MRI_single_subject_data_smooth);
    MRI_single_subject_data_smooth = (MRI_single_subject_data_smooth - 3/8) ./ (numel(MRI_single_subject_data_smooth) + 1/4);
    MRI_single_subject_data_smooth_norm = norminv(MRI_single_subject_data_smooth,0,1);
    
    %%% Write smoothed and normalised nifti files for all control subjects
    out = zeros(size(MaskDataOneDim));
    out(1,find(MaskDataOneDim)) = MRI_single_subject_data_smooth_norm;
    out = reshape(out,nDim1,nDim2,nDim3);
    
    y_Write(out,Header,['Subject_' cell2mat(xx) '_ReHo_mask_smooth_' num2str(smt) 'mm_norm.nii']);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Z-test between single-subject and control group %%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mean_diff = zeros(controls_number,n_voxels);

%%% Find mean difference between clinical subject and N-number control subjects
parfor i = 1:controls_number
    mean_diff(i,:) = MRI_single_subject_data_smooth_norm - MRI_control_data_smooth_norm(i,:);
    
end

%%% Calculate z-score between a clinical subject and N-number control subjects
z_score = mean(mean_diff) ./ std(MRI_control_data_smooth_norm);
z_score(isinf(z_score)) = eps; % avoid inf's' in map - replace with ~0
z_score(isnan(z_score)) = eps; % avoid NaN's' in map - replace with ~0

%%%%%%%%%%%%%%%%%%%%%%%%

%%% Write nifti-file %%%

%%%%%%%%%%%%%%%%%%%%%%%%

out = zeros(size(MaskDataOneDim));
out(1,find(MaskDataOneDim)) = z_score;
out = reshape(out,nDim1,nDim2,nDim3);

%unit2 = strcmpi(resp2,'Yes');

if unit == 1
    save(['Z_map_subject_' cell2mat(xx) '.mat'],'MRI_control_data_smooth_norm','MRI_single_subject_data_smooth_norm','z_score'); % save data in .mat format (+relevant struct name)
    y_Write(out,Header,['Z_map_' cell2mat(xx) '.nii']);
    
else
   save(['Z_map_ReHo_subject_' cell2mat(xx) '.mat'],'MRI_control_data_smooth_norm','MRI_single_subject_data_smooth_norm','z_score'); % save data in .mat format (+relevant struct name)
   y_Write(out,Header,['Z_map_ReHo_' cell2mat(xx) '.nii']);
    
end

theElapsedTime = toc(theElapsedTime);
fprintf('\n Calculation is finshed - elapsed time: %g minutes\n', theElapsedTime/60);
clear