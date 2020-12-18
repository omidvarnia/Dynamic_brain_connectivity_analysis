function MSE_fMRI = cGSP_MSE_fMRI(X_RS, N_scale, dim_m, tol_r)
% Extract multiscale entropy (MSE) curves from the input data
%
% Inputs:
%           X_RS            : (N_ROI x N_T x N_subj) ROI-wise resting state fMRI data for all ROIs and all subjects.
%           D_G             : (N_ROI x 1) Sorted spatial harmonies (frequencies) from lowest to highest
%           N_scale         : (integer) Number of time scales for computing MSE (default = 15)
%           dim_m           : (integer) Embedding dimension of MSE (default = 2)
%           tol_r           : (real positive) Tolerance parameter r of MSE (default = 0.2)
%
%
% Outputs:
%           MSE_fMRI        : (N_ROI x N_scale x N_subj) MSE curves of fMRI for all ROIs and all subjects
%
% Reference 1: M. G. Preti & D. Van De Ville, "Decoupling of brain function
% from structure reveals regional behavioral specialization in humans", Nat. Comm., Vol 10, No. 4747, 2019
%
% Reference 2: M. Costa, A. Goldberger, C. Peng, "Multiscale entropy
% analysis of complex physiologic time series", Phys Rev Lett. 89, 068102, 2002
%
% Code reference for projection to the structural space and computation of the structural decoupling index: https://github.com/gpreti/GSP_StructuralDecouplingIndex
%
% Written by : Amir Omidvarnia, PhD
%              Medical Image Processing lab (MIPLAB), EPFL, Switzerland
% Email      : amir.omidvarnia@gmail.com
% 2019 - 2020

%% Z-score the complexity curves over time
[N_ROI, ~, N_subj]      = size(X_RS);
Y                       = X_RS; %zscore(X_RS,0,2);

%% Extract multiscale entropy from the structural eigenmodes
MSE_fMRI                = zeros(N_ROI,N_scale,N_subj); % Array of multiscale entropy curves for all ROIs and all subjects
fprintf(['\n**** MSE analysis of fMRI startsed ...']);
fprintf(['\n' repmat('.',1,N_subj) '\n\n']);

t0                      = tic;
parfor n_subj = 1 : N_subj
    
    for n_roi = 1 : N_ROI
        
        tmp                      = msentropy(Y(n_roi,:,n_subj),dim_m,tol_r,N_scale);
        tmp(find(isinf(tmp)))    = NaN;
        MSE_fMRI(n_roi,:,n_subj) = tmp;
        
    end
    
    fprintf('\b|\n');
        
end
fprintf('**** MSE analysis of fMRI was finished. Elapsed time = %0.2f \n', toc(t0));
