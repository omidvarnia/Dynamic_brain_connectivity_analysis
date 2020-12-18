function FC_fMRI = cGSP_FC_fMRI(X_RS)
% Extract functional connectivity strength measures from the fMRI data
%
% Inputs:
%           X_RS         : (N_ROI x N_T x N_subj) ROI-wise resting state fMRI data for all ROIs and all subjects.
%
% Outputs:
%           FC_fMRI      : (N_ROI x N_subj) functional connectivity strength of fMRI for all ROIs and all subjects
%
% Written by : Amir Omidvarnia, PhD
%              Medical Image Processing lab (MIPLAB), EPFL, Switzerland
% Email      : amir.omidvarnia@gmail.com
% 2019 - 2020

%% Z-score the complexity curves over time
[N_ROI, ~, N_subj]   = size(X_RS);
Y                    = X_RS; %zscore(X_RS,0,2);

%% Extract FC from the structural eigenmodes
FC_fMRI              = zeros(N_ROI,N_subj); % Array of FC strength for all ROIs and all subjects
fprintf(['\n**** FC analysis of fMRI startsed ...']);
fprintf(['\n' repmat('.',1,N_subj) '\n\n']);

t0                   = tic;
for n_subj = 1 : N_subj
    
    tmp = squeeze(Y(:,:,n_subj));
    FC_corr = corr(tmp', 'type', 'pearson'); % --> Convert to Fisher Z
    FC_strength = sum(FC_corr); % Similar to the 'strengths_und' function in BCT 
    FC_fMRI(:,n_subj) = FC_strength;
    
    fprintf('\b|\n');

end
fprintf('**** FC analysis of fMRI was finished. Elapsed time = %0.2f \n', toc(t0));


