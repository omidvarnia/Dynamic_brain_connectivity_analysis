clear all
clc
close all

%%%%%%%%%%%%% Temporal complexity analysis of rsfMRI %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This script implements tes-retest analysis of the paper and reproduces Figure S5    
% for two tolerance values (r = 0.15, 0.5), three downsampling options    
% (no DS, DS at the rate of 2, DS at the rate of 4) and an embedding dimension of m=2:4.
%
% Dependency: 
%             - IPN_icc: Computes the interclass correlations for indexing the reliability analysis according to shrout & fleiss' schema.
%                        The code is written by XINIAN ZUO (zuoxinian@gmail.com) and available at: 
%                        https://au.mathworks.com/matlabcentral/fileexchange/22122-ipn-tools-for-test-retest-reliability-analysis
%
% Written by : Amir Omidvarnia, PhD
%              Medical Image Processing lab (MIPLAB),
%              EPFL, Geneva, Switzerland
% Email      : amir.omidvarnia@gmail.com
% 2019 - 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: A. Omidvarnia, A. Zalesky, S. Mansour, D. Van De Ville, G.
% Jackson; M. Pedersen, 'Temporal complexity of fMRI is reproducible and
% correlates with higher order cognition', To appear in NeuroImage, 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define the necessary file/folder paths and parameters.
Home_dir              = [pwd filesep 'MATLAB_codes']; % The home folder including all MATLAB codes of the paper. A sub-folder called 'Saved_Results' will
                                                      % be created inside Home_dir in which all analysis results will be copied.
Data_folder           = [pwd filesep 'Datasets'];     % Data folder

N_ROI                 = 379;
N_scale               = 25;                      % Number of scales for MSE
m_span                = 2:4;                       % Embedding dimension for MSE
tol_r                 = .15;                      % Tolerance r for MSE: 0.5 or 0.15
Downsamp              = 'DS2';                   % Downsampling rate: 'DS0' for no downsampling, DS2' and 'DS4'

MSE_results_folder    = [Home_dir filesep 'Saved_Results' filesep 'MSE_fMRI_' Downsamp]; % Location of the MSE results for the original data with no downsampling
Test_retest_folder    = [Home_dir filesep 'Saved_Results' filesep 'MSE_fMRI_Test_Retest_' Downsamp]; % Results folder name

%% Load the associated Yeo's ordering of 7 RSNs
tmp                   = load([Home_dir filesep 'Sina_match_Yeo.mat']);
yeoROIs               = tmp.ind_full;
yeoROIs               = yeoROIs(1:N_ROI);  % Labels of 7 RSNs (from -1 to 7 --> -1 being subcortical networks)
N_RSN                 = 8;

%%% Sort the ROI labels
[yeoROIs_sorted, yeoROIs_sorted_ind] = sort(yeoROIs);
yeoOrder_sorted                      = yeoROIs_sorted_ind;
Yeo_RSN_names                        = {'SUBC' 'VIS' 'SM' 'DA' 'VA' 'L' 'FP' 'DMN'};
Yeo_RSN_names                        = Yeo_RSN_names(1:N_RSN);
reordered_RSNs                       = [7 4 8 2 3 5 1 6];
Yeo_RSN_names2                       = Yeo_RSN_names(reordered_RSNs);
yeoLabels                            = unique(yeoROIs_sorted);

%% Test-retest analysis (Intra-Class Correlation coefficient or ICC)
figure
for n_dim = 1 : length(m_span)
    
    dim_m                      = m_span(n_dim); % Embedding dimension for MSE
    
    if(tol_r==0.5)
        MSE_results_mat_file   = [MSE_results_folder filesep 'MSE_Glasser_r05_m' num2str(dim_m) '.mat']; % Filename of thefinal results in mat file
    elseif(tol_r==0.15)
        MSE_results_mat_file   = [MSE_results_folder filesep 'MSE_Glasser_r015_m' num2str(dim_m) '.mat']; % Filename of thefinal results in mat file
    end
    
    if(~exist(MSE_results_mat_file, 'file'))
        disp('Please generate the MSE results of rsfMRI by running the script ''Extract_MSE_of_fMRI.m'' first!')
        return
    else
        %%% Load MSE results
        out_MSE                = load(MSE_results_mat_file);
        Glasser_MSE_maps_ROI   = out_MSE.Glasser_MSE_maps;      % N_ROI x N_scale x N_subj x N_run
        bad_subjects_MSE       = out_MSE.bad_subjects;
    end
        
    %%% Remove problematic subjects 
    bad_subjects_MSE           = unique(bad_subjects_MSE);
    Glasser_MSE_maps_ROI(:, :, bad_subjects_MSE, :)         = [];
    N_subj                     = size(Glasser_MSE_maps_ROI,3);
    
    %% Extract RSN-wise MSE curves from Yeo's atlas
    MSE_RSNs_m                = zeros(N_RSN, N_scale, N_subj, 4); % N_RSN x N_scale x N_subj x N_Run
    for n_yeo = 1 : N_RSN
        
        RSN_ind                     = find(yeoROIs_sorted==yeoLabels(n_yeo));
        MSE_fMRI_tmp                = Glasser_MSE_maps_ROI(RSN_ind, :, :, :);
        MSE_RSNs_m(n_yeo, :, :, :)  = squeeze(nanmean(MSE_fMRI_tmp,1));
        
    end
    
    %% ICC analysis
    ICCs = zeros(N_RSN,N_scale);
    for n_rsn = 1 : N_RSN
        
        for n_scale = 1 : N_scale
            
            x = squeeze(MSE_RSNs_m(n_rsn, n_scale, :, :)); % N_subj x N_run: matrix of observations. Each row is an object of measurement and each column is a judge or measurement.
            
            %%% Analysis
            cse = 3;
            typ = 'single';
            ICCs(n_rsn, n_scale) = IPN_icc(x,cse,typ);
        end
        
    end
    
    %% Plot
    ICCs2          = ICCs(reordered_RSNs,:);
    
    subplot(1, length(m_span),n_dim), imagesc(1:N_scale, 1:N_RSN, ICCs2), caxis([0 .8])
    yticks(1:N_RSN), yticklabels(Yeo_RSN_names2)
    title(['m = ' num2str(dim_m)]), axis square, xlabel('Scale')
    
    if(n_dim==length(m_span))
        colorbar
    end
    
end

disp(['ICC = ' num2str(nanmean(ICCs(:))) ', \pm ' num2str(nanstd(ICCs(:)))])
