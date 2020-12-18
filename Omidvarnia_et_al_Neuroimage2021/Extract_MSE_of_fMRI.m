clear all
clc
close all

%%%%%%%%%%%%% Temporal complexity analysis of rsfMRI %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This script generates Figure 3, Figure 4, Figure 5 and Figures S3 to
% S5 of the manuscript. For reproducing Figure 3 and Figure 4, please make
% sure that the MSE analysis results of head motion have been already
% generated and saved in the approperiate mat file through the scariot
% 'Extract_MSE_of_Motion.m'.
%
% Dependencies:
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
%% Define the necessary file/folder paths 
force_analysis        = 0;                       % 0 or 1. If the complexity analysis should be repeated.

Home_dir              = [pwd filesep 'MATLAB_codes']; % The home folder including all MATLAB codes of the paper. A sub-folder called 'Saved_Results' will
                                                      % be created inside Home_dir in which all analysis results will be copied.
Data_folder           = [pwd filesep 'Datasets'];     % Data folder
task_names            = {'REST1_LR' 'REST1_RL' 'REST2_LR' 'REST2_RL'};
FC_folder             = [Data_folder filesep 'Functional'];
subjects_IDs          = dir(FC_folder);
subjects_IDs          = subjects_IDs(~ismember({subjects_IDs(:).name},{'.','..'}));
subjects_IDs          = {subjects_IDs(:).name};
N_subj                = length(subjects_IDs);
N_tasks               = length(task_names);
N_T                   = [1200 1200 1200 1200];   % Number of TRs

%% Define the multiscale entropy parameters  
N_scale               = 25;                      % Number of scales for MSE

Figure_option         = 'Figure 3';
switch Figure_option
    
    case 'Figure 3'
        m_span        = 2:10;                    % Embedding dimension for MSE
        tol_r         = .5;                      % Tolerance r for MSE: 0.5 or 0.15
        Downsamp      = 'DS0';                   % Downsampling rate: 'DS0' for no downsampling, DS2' and 'DS4'
        
    case 'Figure 4'
        m_span        = 2:10;                    % Embedding dimension for MSE
        tol_r         = .15;                     % Tolerance r for MSE: 0.5 or 0.15
        Downsamp      = 'DS0';                   % Downsampling rate: 'DS0' for no downsampling, DS2' and 'DS4'
        
    case 'Figure 5'
        m_span        = 2:4;                     % Embedding dimension for MSE
        tol_r         = .5;                      % Tolerance r for MSE: 0.5 or 0.15
        Downsamp      = 'DS2';                   % Downsampling rate: 'DS0' for no downsampling, DS2' and 'DS4'
        
    case 'Figure S1'
        m_span        = 2:4;                     % Embedding dimension for MSE
        tol_r         = .15;                      % Tolerance r for MSE: 0.5 or 0.15
        Downsamp      = 'DS2';                   % Downsampling rate: 'DS0' for no downsampling, DS2' and 'DS4'
        
    case 'Figure S2'
        m_span        = 2:4;                     % Embedding dimension for MSE
        tol_r         = .5;                      % Tolerance r for MSE: 0.5 or 0.15
        Downsamp      = 'DS4';                   % Downsampling rate: 'DS0' for no downsampling, DS2' and 'DS4'
        
    case 'Figure S3'
        m_span        = 2:4;                     % Embedding dimension for MSE
        tol_r         = .15;                      % Tolerance r for MSE: 0.5 or 0.15
        Downsamp      = 'DS4';                   % Downsampling rate: 'DS0' for no downsampling, DS2' and 'DS4'
        
        
end

MSE_results_folder    = [Home_dir filesep 'Saved_Results' filesep 'MSE_fMRI_' Downsamp]; % Results folder name
Motion_results_folder = [Home_dir filesep 'Saved_Results' filesep 'MSE_Motion']; % Results folder name

%% Load the associated Yeo's ordering of 8 RSNs
N_ROI                 = 379;
tmp                   = load([Home_dir filesep 'Sina_match_Yeo.mat']);
yeoROIs               = tmp.ind_full;
yeoROIs               = yeoROIs(1:N_ROI);  % Labels of 7 RSNs (from -1 to 7 --> -1 being subcortical networks)
N_RSN                 = 8;

%%% Sort the ROI labels
[yeoROIs_sorted, yeoROIs_sorted_ind] = sort(yeoROIs);
yeoOrder_sorted                      = yeoROIs_sorted_ind;
Yeo_RSN_names                        = {'SUBC' 'VIS' 'SM' 'DA' 'VA' 'L' 'FP' 'DMN'};
reordered_RSNs                       = [7 4 8 2 3 5 1 6];
Yeo_RSN_names                        = Yeo_RSN_names(1:N_RSN);
yeoLabels                            = unique(yeoROIs_sorted);

%% Check if the data and result folders need to be created.
if(force_analysis==1)
    force_answer     = questdlg('Be careful! The force flag is 1. Do you want to continue?');
else
    force_answer     = 'No';
end

if(strcmp(force_answer,'Cancel'))
    return
end

if(strcmp(force_answer, 'Yes') && force_analysis)
    try, rmdir(MSE_results_folder,'s'), end
end

if(~exist(MSE_results_folder,'dir') || force_analysis)
    mkdir(MSE_results_folder)
end

%% Extract complexity from different lengths of the rsfMRI datasets and make the braim maps
figure(1)
figure(2)
figure(3)
for n_dim = 1 : length(m_span)
    
    dim_m                      = m_span(n_dim); % Embedding dimension for MSE
    Glasser_MSE_maps           = zeros(N_ROI, N_scale, N_subj, N_tasks);
    bad_subjects               = []; % Problematic subjects
    
    if(tol_r==0.5)
        MSE_results_mat_file   = [MSE_results_folder filesep 'MSE_Glasser_r05_m' num2str(dim_m) '.mat']; % Filename of the final results in mat file
        Motion_MSE_mat_file    = [Motion_results_folder filesep 'MSE_motion_r05_m' num2str(dim_m) '.mat'];      % Filename of the final results in mat file
    elseif(tol_r==0.15)
        MSE_results_mat_file   = [MSE_results_folder filesep 'MSE_Glasser_r015_m' num2str(dim_m) '.mat']; % Filename of the final results in mat file
        Motion_MSE_mat_file    = [Motion_results_folder filesep 'MSE_motion_r015_m' num2str(dim_m) '.mat'];      % Filename of the final results in mat file
    end
    
    if(~exist(Motion_MSE_mat_file, 'file') && strcmp(Downsamp,'DS0'))
        disp('MSE analysis results of head motion are missing. Please run ''Extract_MSE_of_Motion.m'' first!')
        return
    end
    
    if(~exist(MSE_results_mat_file, 'file') || force_analysis)
        for fMRI_run_ind = 1 : N_tasks
            
            %%% Set the filenames of the resulting complexity measures
            task_name                     = task_names{fMRI_run_ind};
            
            %% Concatenate the fMRI time series of the resquested run
            disp(['################## fMRI run ' num2str(fMRI_run_ind) ' (' task_name ') started ...']);
            t0 = tic;
            
            fMRI                                = zeros(N_ROI, N_T(fMRI_run_ind), N_subj);
            for n_subj = 1 : N_subj
                
                subj_ID       = subjects_IDs{n_subj};
                
                try
                    filename_subj = [FC_folder filesep subj_ID filesep task_names{fMRI_run_ind} filesep 'atlas_rfMRI.mat'];
                    out           = load(filename_subj);
                    TS            = double(out.ts); clear out
                    TS            = TS(yeoOrder_sorted,:);
                    fMRI(:, :, n_subj) = TS(:, 1:N_T(fMRI_run_ind));        clear TS
                catch
                    
                    bad_subjects      = [bad_subjects n_subj];
                    
                end
                
            end
            
            %%% Downsampling, if needed
            if(strcmp(Downsamp,'DS2'))
                fMRI = fMRI(:, 1:2:end, :);
            elseif(strcmp(Downsamp, 'DS4'))
                fMRI = fMRI(:, 1:4:end, :);
            end
            
            %%% MSE of fMRI
            MSE_fMRI                  = cGSP_MSE_fMRI(fMRI, N_scale, dim_m, tol_r);
            
            %%% Make the ROI-wise complexity map by taking the average across subjects
            Glasser_MSE_maps(:, :, :, fMRI_run_ind)    = MSE_fMRI; % ROI-wise complexity over subjects at each scale;
            save(MSE_results_mat_file, 'Glasser_MSE_maps', 'bad_subjects', 'N_scale', 'dim_m', 'tol_r')
            
            %%% Clear unnecessay variables
            clear MSE_fMRI
            
            disp(['################## fMRI run ' num2str(fMRI_run_ind) ' (' task_name ') was completed! Total elapsed time = ' num2str(toc(t0)) '\n']);
            sgtitle(['fMRI run: ' task_name])
            
        end
        
    else
        
        out                = load(MSE_results_mat_file);
        Glasser_MSE_maps   = out.Glasser_MSE_maps;
        bad_subjects       = out.bad_subjects;
        N_scale            = out.N_scale;
        dim_m              = out.dim_m;
        tol_r              = out.tol_r;
        
    end
    
    %%% Remove problematic subjects (in Sina's database, some datasets have
    %%% different length at different runs and all datasets do not have 1200
    %%% TRs necessarily).
    bad_subjects                                    = unique(bad_subjects);
    Glasser_MSE_maps(:, :, bad_subjects, :)         = [];
    % Glasser_MSE_maps(find(isnan(Glasser_MSE_maps))) = nanmean(Glasser_MSE_maps(:));
    N_subj2                                         = size(Glasser_MSE_maps,3);
    
    %% Compute the complexity index
    MSE_fMRI              = squeeze(mean(Glasser_MSE_maps,4)); % Mean Hurst exponent map over fMRI runs
    MSE_fMRI(find(isnan(MSE_fMRI))) = nanmean(MSE_fMRI(:));
    MSE_CI                = zeros(N_ROI, N_subj2); % Complexity index (area under the MSE curve) for all ROIs and all subjects
    for n_subj = 1 : N_subj2
        MSE_CI(:, n_subj) = trapz(1:N_scale, squeeze(MSE_fMRI(:, :, n_subj)),2)/N_scale;
    end
    

    %% Load motion file
    if(strcmp(Downsamp,'DS0'))
        out                       = load(Motion_MSE_mat_file);
        MSE_Framewise_Displacem   = out.MSE_Framewise_Displacem;
    end
    
    %% Plot the mean Hurst exponent of RSNs from Yeo's atlas
    colormap_RSNs             = flipud(colormap(jet));
    colormap_ind              = fix(linspace(1,256,N_RSN));
    colormap_RSNs             = colormap_RSNs(colormap_ind, :);
    
    MSE_RSNs_m                = zeros(N_RSN, N_scale);
    MSE_RSNs_std              = zeros(N_RSN, N_scale);
    CI_RSNs_m                 = zeros(N_RSN, 1);
    CI_RSNs_std               = zeros(N_RSN, 1);
    for n_yeo = 1 : N_RSN
        
        RSN_ind                   = find(yeoROIs_sorted==yeoLabels(reordered_RSNs(n_yeo)));
        MSE_fMRI_tmp              = MSE_fMRI(RSN_ind, :, :);
        MSE_RSNs_m(n_yeo, :)      = nanmean(squeeze(nanmean(MSE_fMRI_tmp,1)),2);
        MSE_RSNs_std(n_yeo,:) = nanstd(squeeze(std(MSE_fMRI_tmp,0,1)),0,2);%/sqrt(length(RSN_ind));
        
        CI_RSNs_m(n_yeo)          = nanmean(nanmean(MSE_CI(RSN_ind,:),1));
        CI_RSNs_std(n_yeo)        = nanstd(nanstd(MSE_CI(RSN_ind,:),0,1));
                
        %%% Plot MSE of rsfMRI
        figure(1), hold on, subplot(1, length(m_span),n_dim), errorbar(MSE_RSNs_m(n_yeo, :), MSE_RSNs_std(n_yeo, :), 'linewidth', 2, 'color', colormap_RSNs(n_yeo,:)), axis square, xlim([1 N_scale])
    end
    [CI_RSNs_m_sorted, CI_RSNs_m_sorted_ind] = sort(CI_RSNs_m, 'descend');
    
    
    %%% Plot MSE of FD
    if(strcmp(Downsamp,'DS0'))
        MSE_motion_FD                 = squeeze(mean(MSE_Framewise_Displacem,1)); % Mean Hurst exponent map over fMRI runs
        MSE_motion_FD_m               = mean(MSE_motion_FD,2);
        MSE_motion_FD_std_err         = std(MSE_motion_FD, 0, 2);
        figure(1), hold on, subplot(1, length(m_span),n_dim), errorbar(MSE_motion_FD_m, MSE_motion_FD_std_err, 'linewidth', 2); axis square, xlim([1 N_scale])
    end
        
    legend_txt = cell(N_RSN+1,1); legend_txt(1:N_RSN) = Yeo_RSN_names(reordered_RSNs); legend_txt(N_RSN+1) = {'FD'}; %legend_txt(N_RSN+2) = {'Rel'};
    if(n_dim==length(m_span))
        legend(legend_txt)
    end
    
    
    %%% Plot the cortical and subcortical MSE curves
    MSE_fMRI_subcort             = squeeze(nanmean(MSE_fMRI(1:19,:,:),3)); % Average over subjects
    MSE_fMRI_cort                = squeeze(nanmean(MSE_fMRI(20:end,:,:),3)); % Average over subjects
    figure(2), subplot(1, length(m_span),n_dim), plot(MSE_fMRI_cort', 'b', 'linewidth', 2), hold on, plot(MSE_fMRI_subcort', 'r', 'linewidth', 2), axis square, xlim([1 N_scale])
    
    figure(3), subplot(1, length(m_span),n_dim), bar(1:N_RSN, CI_RSNs_m_sorted), 
    hold on, errorbar(1:N_RSN, CI_RSNs_m_sorted, CI_RSNs_std(CI_RSNs_m_sorted_ind), 'linewidth', 4), er.Color = [0 0 0]; er.lineStyle = 'none'; axis square
    xticks(1:N_ROI); xticklabels(Yeo_RSN_names(reordered_RSNs(CI_RSNs_m_sorted_ind)))
    if(tol_r==.5), ylim([0 1]), else, ylim([0 2.5]), end

    
end






























