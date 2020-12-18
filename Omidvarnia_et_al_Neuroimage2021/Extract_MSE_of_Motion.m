clear all
clc
close all

%%%%%%%%%%%%% Temporal complexity analysis of rsfMRI %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This script extracts Multiscale entropy from head motion (franewise displacement) 
% of HCP database (N = 1000, four rest runs per subject) using two
% tolerance parameters (r = 0.15, 0.5) and 9 embedding dimensions
% (m = 2 to 10) over 25 time scales. The results are saved in mat files and
% are loaded for generating Figure 3 and Fugure 4 in the script
% 'Extract_MSE_of_fMRI.m'.
%
% Dependencies:
%             - FDCalc       : Calculates the Frame-Wise Displacement of head motion parameters
%                              The code is written by Soroosh Afyouni, NISOx.org, 2017 (srafyouni@gmail.com)
%             - msentropy    : Function for calculating multiscale entropy,
%                              based on the function 'sampenc', available at http://people.ece.cornell.edu/land/PROJECTS/Complexity/sampenc.m             
%             - cGSP_MSE_fMRI: Extract multiscale entropy (MSE) curves from the input data
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
%% Define the necessary file/folder paths and parameters
force_analysis        = 0;          % 0 or 1. If the complexity analysis should be repeated.

Home_dir              = [pwd filesep 'MATLAB_codes']; % The home folder including all MATLAB codes of the paper. A sub-folder called 'Saved_Results' will
                                                      % be created inside Home_dir in which all analysis results will be copied.
Motion_folder         = [pwd filesep 'Motion'];       % Motion parameters folder

N_scale               = 25;
m_span                = 2:10;                       % Embedding dimension for MSE
tol_r                 = .15;
Motion_results_folder = [Home_dir filesep 'Saved_Results' filesep 'MSE_Motion']; % Results folder name

N_subj               = 1000;                % Number of subjects or datasets
N_T                  = 1200;       % Number of time points in the LR run of session 1
TR_start             = 1;
subjs                = dir(Motion_folder);
subjs                = subjs(3:end);

if(~exist(Motion_results_folder,'dir') || force_analysis)
    mkdir(Motion_results_folder)
end

%% Calculate Frame-wise Displacement of the motion parameters for all subjects
Framewise_Displacem          = zeros(4, N_T , N_subj);   % Motion parameters for all subjects across 4 resting state runs

ss                           = 0;
bad_datasets                 = [];
for n_subj = 1 : N_subj
    
    subj_ID                  = subjs(n_subj).name; % Subject ID
    
    try
        %%% Define the motion folders of two sessions and LR-RL slicing optins
        Motion_RL1_folder    = [Motion_folder filesep subj_ID filesep 'REST1_RL'];
        Motion_LR1_folder    = [Motion_folder filesep subj_ID filesep 'REST1_LR'];
        Motion_RL2_folder    = [Motion_folder filesep subj_ID filesep 'REST2_RL'];
        Motion_LR2_folder    = [Motion_folder filesep subj_ID filesep 'REST2_LR'];
        
        %%% Collect FD from all subjects and all runs
        tmp1                             = load([Motion_RL1_folder filesep 'Movement_Regressors.txt']);
        tmp1                             = tmp1(TR_start:end, 1:6);
        [~,tmp1]                         = FDCalc(tmp1);
        Framewise_Displacem(1,:,n_subj)  = [0;tmp1.SS]; % Frame-wise displacement
        
        tmp2                             = load([Motion_LR1_folder filesep 'Movement_Regressors.txt']);
        tmp2                             = tmp2(TR_start:end, 1:6);
        [~,tmp2]                         = FDCalc(tmp2);
        Framewise_Displacem(2,:,n_subj)  = [0;tmp2.SS]; % Frame-wise displacement
        
        tmp3                             = load([Motion_RL2_folder filesep 'Movement_Regressors.txt']);
        tmp3                             = tmp3(TR_start:end, 1:6);
        [~,tmp3]                         = FDCalc(tmp3);
        Framewise_Displacem(3,:,n_subj)  = [0;tmp3.SS]; % Frame-wise displacement
        
        tmp4                             = load([Motion_LR2_folder filesep 'Movement_Regressors.txt']);
        tmp4                             = tmp4(TR_start:end, 1:6);
        [~,tmp4]                         = FDCalc(tmp4);
        Framewise_Displacem(4,:,n_subj)  = [0;tmp4.SS]; % Frame-wise displacement
        clear tmp1 tmp2 tmp3 tmp4
                
    catch
        ss                    = ss + 1;
        bad_datasets          = [bad_datasets n_subj];
        disp(['Dataset No. ' num2str(n_subj) ' (' subj_ID ') is problematic!'])
    end
    
end

Framewise_Displacem(:,:,bad_datasets) = [];

%% MSE analysis of motion
for n_dim = 1 : length(m_span)
    
    dim_m                      = m_span(n_dim); % Embedding dimension for MSE
    if(tol_r==0.5)
        Motion_MSE_mat_file   = [Motion_results_folder filesep 'MSE_motion_r05_m' num2str(dim_m) '.mat']; % Filename of thefinal results in mat file
    elseif(tol_r==0.15)
        Motion_MSE_mat_file   = [Motion_results_folder filesep 'MSE_motion_r015_m' num2str(dim_m) '.mat']; % Filename of thefinal results in mat file
    end
    
    if(~exist(Motion_MSE_mat_file,'file') || force_analysis)

        MSE_Framewise_Displacem      = cGSP_MSE_fMRI(Framewise_Displacem, N_scale, dim_m, tol_r);
        MSE_Framewise_Displacem(find(isnan(MSE_Framewise_Displacem))) = nanmean(MSE_Framewise_Displacem(:));
        save(Motion_MSE_mat_file, 'MSE_Framewise_Displacem')
        
    else
        out                         = load(Motion_MSE_mat_file);
        MSE_Framewise_Displacem     = out.MSE_Framewise_Displacem;
    end
    
    %% Plot the histograms of Motion parameters
    figure,
    for fMRI_run_ind = 1 : 4
        
        m_MSE_FD    = mean(squeeze(MSE_Framewise_Displacem(fMRI_run_ind,:,:)),2);
        std_MSE_FD  = std(squeeze(MSE_Framewise_Displacem(fMRI_run_ind,:,:)),0,2);
        subplot(2, 4, fMRI_run_ind); errorbar(m_MSE_FD, std_MSE_FD, 'linewidth', 4);
        xlabel('Scale', 'FontSize',15,'FontWeight','bold'), ylabel(sprintf('MSE'), 'FontSize',15,'FontWeight','bold')
        xlim([1 N_scale]), %ylim([1 2])
        title(['Rest run ' num2str(fMRI_run_ind)], 'FontSize',15,'FontWeight','bold'), axis square
        set(gca, 'fontsize',15, 'fontweight', 'bold')
        sgtitle(['MSE of Motion (FD), Embedding dimension = ' num2str(m_span(n_dim)) ', r = ' num2str(tol_r)])
        
    end
end
