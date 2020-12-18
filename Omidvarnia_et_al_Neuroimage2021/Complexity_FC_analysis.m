clear all
clc
close all

%%%%%%%%%%%%% Temporal complexity analysis of rsfMRI %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This script reproduces Figure 8, representing the relationship between
%%% temporal complexity of fMRI and functional connectivity for the following 
% parameters: r = 0.5, m = 2 and no downsampling.
%
% Dependency:
%             - cGSP_FC_fMRI: Extract functional connectivity strength measures from the fMRI data
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
force_analysis        = 1;                       % 0 or 1. If the complexity analysis should be repeated.

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
N_T                   = [1200 1200 1200 1200];  % Number of TRs

FC_results_folder     = [Home_dir filesep 'Saved_Results' filesep filesep 'FC_fMRI_D0']; % Results folder name for FC strength analysis of fMRI
FC_results_mat_file   = [FC_results_folder filesep 'FC_Glasser.mat']; % Filename of thefinal results in mat file

MSE_results_folder    = [Home_dir filesep 'Saved_Results' filesep 'MSE_fMRI_DS0'];        % Results folder name of MSE analysis of fMRI with no downsampling
MSE_results_mat_file  = [MSE_results_folder filesep 'MSE_Glasser_r05_m2.mat']; % Filename of thefinal results in mat file

%% Load the MSE analysis of fMRI for r = 0.5, m = 2, and no downsampling
if(~exist(MSE_results_mat_file, 'file'))
    disp('Please generate the MSE results of rsfMRI by running the script ''Extract_MSE_of_fMRI.m'' first!')
    return
else
    %%% Load MSE results
    out_MSE                = load(MSE_results_mat_file);
    Glasser_MSE_maps_ROI   = out_MSE.Glasser_MSE_maps;      % N_ROI x N_scale x N_subj x N_run
    bad_subjects_MSE       = out_MSE.bad_subjects;
    N_scale                = out_MSE.N_scale;
end

%% Load the associated Yeo's ordering of 8 RSNs
N_ROI                 = 379;
tmp                   = load([Data_folder filesep 'Atlas' filesep 'Sina_match_Yeo.mat']);
yeoROIs               = tmp.ind_full;
yeoROIs               = yeoROIs(1:N_ROI);  % Labels of 8 RSNs (from -1 to 7 --> -1 being subcortical networks)
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
    try, rmdir(FC_results_folder,'s'), end
end

if(~exist(FC_results_folder,'dir') || force_analysis)
    mkdir(FC_results_folder)
end

%% Extract functional connectivity strength measures from rsfMRI datasets and make the braim maps
if(~exist(FC_results_mat_file, 'file') || force_analysis)
    
    Glasser_FC_maps_ROI    = zeros(N_ROI, N_subj, N_tasks, 1);
    bad_subjects_FC        = []; % Problematic subjects
    
    for fMRI_run_ind = 1 : N_tasks
        
        %%% Set the filenames of the resulting complexity measures
        task_name             = task_names{fMRI_run_ind};
        
        %% Concatenate the fMRI time series of the resquested run
        disp(['################## fMRI run ' num2str(fMRI_run_ind) ' (' task_name ') started ...']);
        t0 = tic;
        
        fMRI                  = zeros(N_ROI, N_T(fMRI_run_ind), N_subj);
        for n_subj = 1 : N_subj
            
            subj_ID           = subjects_IDs{n_subj};
            
            try
                filename_subj = [FC_folder filesep subj_ID filesep task_names{fMRI_run_ind} filesep 'atlas_rfMRI.mat'];
                tmp           = load(filename_subj);
                TS            = double(tmp.ts); clear out
                TS            = TS(yeoOrder_sorted,:);
                fMRI(:, :, n_subj) = TS(:, 1:N_T(fMRI_run_ind));        clear TS tmp
            catch
                
                bad_subjects_FC  = [bad_subjects_FC n_subj];
                
            end
            
        end
        
        %%% FC analysis of fMRI
        FC_fMRI               = cGSP_FC_fMRI(fMRI);
        
        %%% Make the ROI-wise complexity map by taking the average across subjects
        Glasser_FC_maps_ROI(:, :, fMRI_run_ind)    = FC_fMRI; % ROI-wise complexity over subjects at each scale;
        save(FC_results_mat_file, 'Glasser_FC_maps_ROI', 'bad_subjects_FC')
                
        disp(['################## fMRI run ' num2str(fMRI_run_ind) ' (' task_name ') was completed! Total elapsed time = ' num2str(toc(t0)) '\n']);
        sgtitle(['fMRI run: ' task_name])
        
    end
    
else
    
    %%% Load FC strength results
    out_FC                 = load(FC_results_mat_file);
    Glasser_FC_maps_ROI    = out_FC.Glasser_FC_maps_ROI;        % N_ROI x N_subj x N_run
    bad_subjects_FC        = out_FC.bad_subjects_FC;
        
end

%% ROI-wise FC strengths and Hurst exponents
%%% Remove problematic subjects (in Sina's database, some datasets have
%%% different length at different runs and all datasets do not have 1200
%%% TRs necessarily).
bad_subjects                                                = unique([bad_subjects_FC bad_subjects_MSE]);
N_subj                                                      = N_subj - length(bad_subjects);

Glasser_FC_maps_ROI(:, bad_subjects, :)                     = [];
Glasser_FC_maps_ROI(find(isnan(Glasser_FC_maps_ROI)))       = nanmean(Glasser_FC_maps_ROI(:));

Glasser_MSE_maps_ROI(:, :, bad_subjects, :)                 = [];
Glasser_MSE_maps_ROI(find(isnan(Glasser_MSE_maps_ROI)))     = nanmean(Glasser_MSE_maps_ROI(:));

%% RSN-wise FC strengths and Hurst exponents
Glasser_FC_maps_RSN       = zeros(N_RSN, N_subj, N_tasks);           % N_RSN x N_subj x N_run
Glasser_MSE_maps_RSN      = zeros(N_RSN, N_scale, N_subj, N_tasks);  % N_RSN x N_scale x N_subj x N_run

for n_yeo = 1 : N_RSN
    RSN_ind                             = find(yeoROIs_sorted==yeoLabels(reordered_RSNs(n_yeo)));
    
    %%% FC strength
    FC_fMRI_tmp                         = Glasser_FC_maps_ROI(RSN_ind, :, :);
    Glasser_FC_maps_RSN(n_yeo, :, :)    = nanmean(FC_fMRI_tmp,1);                   % N_RSN x N_subj x N_run
    
    %%% MSE
    MSE_fMRI_tmp                        = Glasser_MSE_maps_ROI(RSN_ind, :, :, :);
    Glasser_MSE_maps_RSN(n_yeo, :, :, :)= nanmean(MSE_fMRI_tmp,1);                  % N_RSN x N_scale x N_subj x N_run
end
 
%%% Averaging over fMRI runs
Glasser_FC_maps_ROI_m    = mean(Glasser_FC_maps_ROI,3);     % FC strength --> N_ROI x N_subj
Glasser_MSE_maps_ROI_m   = mean(Glasser_MSE_maps_ROI,4);    % MSE         --> N_ROI x N_scale x N_subj

Glasser_FC_maps_RSN_m    = mean(Glasser_FC_maps_RSN,3);     % FC strength --> N_RSN x N_subj
Glasser_MSE_maps_RSN_m   = mean(Glasser_MSE_maps_RSN,4);    % MSE         --> N_RSN x N_scale x N_subj

%% Correlation analysis between rsfMRI complexity (Hurst) and functional connectivity strength
corr_FC_MSE_RSN                   = zeros(N_RSN, N_scale);
for n_yeo = 1 : N_RSN
    corr_FC_MSE_RSN(n_yeo, :)     = corr(Glasser_FC_maps_RSN_m(n_yeo, :)', squeeze(Glasser_MSE_maps_RSN_m(n_yeo, :, :))', 'type', 'spearman');
end

%% Plot
colormap_RSNs             = flipud(colormap(jet));
colormap_ind              = fix(linspace(1,256,N_RSN));
colormap_RSNs             = colormap_RSNs(colormap_ind, :);

figure
for n_yeo = 1 : N_RSN
    hold on, plot(1:N_scale, corr_FC_MSE_RSN(n_yeo, :), 'linewidth', 6, 'color', colormap_RSNs(n_yeo,:)), axis square, xlim([1 N_scale]), ylim([-1 1])
end
hold on, line([1 N_scale], [0 0], 'linewidth', 6, 'color', 'k', 'linestyle', '--')
xlabel('Scale'), ylabel('Spearman correlation'), legend(Yeo_RSN_names(reordered_RSNs), 'location', 'se')








