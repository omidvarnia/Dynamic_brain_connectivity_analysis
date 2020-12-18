clear all
clc
close all

%%%%%%%%%%%%% Temporal complexity analysis of rsfMRI %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This script reproduces Figure S4 as well as the results of Tables S7 
% to S12 summarizing the relationship between rsfMRI temporal complexity 
% and five behavioural measures for two tolerance values (r = 0.15, 0.5),  
% three downsampling options (no DS, DS at the rate of 2, DS at the rate  
% of 4) and an embedding dimension of m=2.
%
% Dependency:
%            - fdr_bh      : FDR controlling of the familywise errors. Code
%                            written by David M. Groppe, Kutaslab, Dept. of Cognitive Science, University of California, San Diego, March 24, 2010
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
force_analysis        = 0;                       % 0 or 1. If the complexity analysis should be repeated.
Home_dir              = [pwd filesep 'MATLAB_codes']; % The home folder including all MATLAB codes of the paper. A sub-folder called 'Saved_Results' will
                                                      % be created inside Home_dir in which all analysis results will be copied.
Data_folder           = [pwd filesep 'Datasets'];     % Data folder

behav_vars_file       = [Home_dir filesep 'HCP_behav_vars_936_by_50_v2.mat'];
behav_vars_list_file  = [Home_dir filesep 'list_of_behavioural_variables.txt'];

FC_folder             = [Data_folder filesep 'Functional'];
selected_subj_IDs     = dir(FC_folder);
selected_subj_IDs     = {selected_subj_IDs.name};
selected_subj_IDs     = selected_subj_IDs(3:end);
N_subj                = length(selected_subj_IDs);
N_ROI                 = 379;

%%%% For reproducing Figure S4: Downsamp = 'DS0', tol_r = 0.5, dim_m = 2,
%%%% N_perm = 10000, N_scale = 25
N_scale               = 25;                      % Number of scales for MSE
dim_m                 = 2;                       % Embedding dimension for MSE
tol_r                 = .5;                      % Tolerance r for MSE: 0.5 or 0.15
N_perm                = 10000;                   % Number of randomizations for permutation testing
Downsamp              = 'DS0';                   % Downsampling rate: 'DS0' for no downsampling, DS2' and 'DS4'

MSE_results_folder    = [Home_dir filesep 'Saved_Results' filesep 'MSE_fMRI_' Downsamp]; % Location of the MSE results for the original data with no downsampling
Brain_maps_folder     = [Home_dir filesep 'Saved_Results' filesep 'MSE_Brain_maps_' Downsamp]; % Location of the maen and STD brain maps of temporal complexity
Behav_results_folder  = [Home_dir filesep 'Saved_Results' filesep 'MSE_fMRI_Behav_' Downsamp]; % Location of the maen and STD brain maps of temporal complexity

if(~exist(Brain_maps_folder,'dir') || force_analysis)
    mkdir(Brain_maps_folder)
end

if(~exist(Behav_results_folder,'dir') || force_analysis)
    mkdir(Behav_results_folder)
end

%% Load the associated Yeo's ordering of 7 RSNs
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
Yeo_RSN_names                        = Yeo_RSN_names(reordered_RSNs);
yeoLabels                            = unique(yeoROIs_sorted);

ROI_labels                           = readtable([Home_dir filesep 'Sina_Glasser_map.csv']);
ROI_labels                           = ROI_labels(yeoROIs_sorted_ind,:);

%% Load the MSE results for the original datasets (no downsampling) of all HCP subjects
if(tol_r==0.5)
    MSE_results_mat_file   = [MSE_results_folder filesep 'MSE_Glasser_r05_m' num2str(dim_m) '.mat']; % Filename of the final results in mat file
    Corr_Surr_mat_file     = [Behav_results_folder filesep 'Corr_Surr_Glasser_r05_m' num2str(dim_m) '.mat']; % Filename of the surrogate data for correlation analysis between MSE and behavioural measures
elseif(tol_r==0.15)
    MSE_results_mat_file   = [MSE_results_folder filesep 'MSE_Glasser_r015_m' num2str(dim_m) '.mat']; % Filename of the final results in mat file
    Corr_Surr_mat_file     = [Behav_results_folder filesep 'Corr_Surr_Glasser_r015_m' num2str(dim_m) '.mat']; % Filename of the surrogate data for correlation analysis between MSE and behavioural measures
end

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
    bad_subjects           = out_MSE.bad_subjects;
end

%% Remove problematic subjects (in Sina's database, some datasets have
bad_subjects          = unique(bad_subjects);
Glasser_MSE_maps_ROI(:, :, bad_subjects, :)   = [];

% Glasser_MSE_maps(find(isnan(Glasser_MSE_maps))) = nanmean(Glasser_MSE_maps(:));
selected_subj_IDs(bad_subjects) = [];
N_subj2               = length(selected_subj_IDs);

%% Compute the complexity index
MSE_fMRI              = squeeze(nanmean(Glasser_MSE_maps_ROI,4)); % Mean Hurst exponent map over fMRI runs
MSE_CI                = zeros(N_ROI, N_subj2); % Complexity index (area under the MSE curve) for all ROIs and all subjects
for n_subj = 1 : N_subj2
    MSE_CI(:, n_subj) = trapz(1:N_scale, squeeze(MSE_fMRI(:, :, n_subj)),2)/N_scale;
end
clear MSE_fMRI

%%% Extract RSN-wise MSE curves from Yeo's atlas
MSE_RSNs              = zeros(N_subj2, N_RSN); % N_subj x N_RSN
for n_yeo = 1 : N_RSN
    
    RSN_ind           = find(yeoROIs_sorted==yeoLabels(n_yeo));
    
    MSE_fMRI_tmp      = MSE_CI(RSN_ind, :);
    MSE_RSNs(:, n_yeo)  = squeeze(nanmean(MSE_fMRI_tmp,1)); clear MSE_fMRI_tmp
    
end

MSE_RSNs   = MSE_RSNs(:, reordered_RSNs);

%% Load the behavioral variable info
behav_var_names    = {};
fileID             = fopen(behav_vars_list_file,'r');
tline              = fgetl(fileID);
behav_var_names{1} = tline;
s                  = 1;
while ischar(tline)
    s                  = s + 1;
    tline              = fgetl(fileID);
    behav_var_names{s} = tline;
end
behav_var_names    = behav_var_names(1:end-1);
fclose(fileID);

%%% Load the subject IDs and find the corresponding ones
behav_vars_and_IDs = load(behav_vars_file);
all_subj_IDs_1206  = behav_vars_and_IDs.full_subject_list;     % The subject indices associated with 'behav_vars'
X                  = behav_vars_and_IDs.behavioural_variables; % A selection of 50 behavioural variables associated with the subjects
N_behav            = size(X,2);

%% Find the corresponding subject IDs with MSE results and behavioural variables (matrix X)
selected_IDs       = zeros(size(selected_subj_IDs));
for k = 1 : N_subj2
    selected_IDs(k) = find(str2num(cell2mat(selected_subj_IDs(k))) == all_subj_IDs_1206);
end
X                   = X(selected_IDs,:); % Behavioural variables associated with the MSE results

%% Find the correlation between MSE and behavioural variables over subjects
for n_yeo = 1 : N_RSN
    y                   = MSE_RSNs(:,n_yeo);
    [max_corr, ind_max] = max(corr(y,X));
    disp([Yeo_RSN_names{n_yeo} ': max corr = ' num2str(max_corr) ', behav variable = ' num2str(ind_max)])
end

%% Linear regression
% {'task1 = behaviouraldata.Flanker_Unadj;'       }
% {'task2 = behaviouraldata.CardSort_Unadj;'      }
% {'task4 = behaviouraldata.WM_Task_Acc;'         }
% {'task5 = behaviouraldata.PMAT24_A_CR;'         }
% {'task41 = behaviouraldata.Relational_Task_Acc;'}
selected_behav    = [1 2 4 5 41];   % 5: Index of the fluid intelligence in HCP behavioral variables (PMAT24_A_CR)
behav_var_names(selected_behav)'

X_selected        = X(:,selected_behav);
X_selected        = zscore(X_selected,0,1);
N_behav_selected  = size(X_selected,2);

beta_FI           = zeros(N_RSN,1); % beta estimates of the fluid intelligence variable
p_FI              = zeros(N_RSN,1); % p-values of the fluid intelligence variable
t_FI              = zeros(N_RSN,1); % t-stats of the fluid intelligence variable
Corr_y            = zeros(N_RSN,1); % Spearman correlation between original CIs and estimated CIs

% Collect the 3 outputs for all 5 behavioural variables (beta, p-value, t-statistic for each variable)
out_measures      = zeros(N_behav_selected, N_RSN,3);

figure
for n_rsn = 1 : N_RSN
    
    y = zscore(MSE_RSNs(:,n_rsn));
    
    table_with_all_variables = table(y,X_selected(:,1),X_selected(:,2),X_selected(:,3),X_selected(:,4),X_selected(:,5),...
        'VariableNames',{'Complexity_index','Flanker_Inhibition','CardSort_Flexibility','NBack_WorkingMemory', 'Ravens_FluidIntelligence','Relational_task'});
    
    disp(['******  ' Yeo_RSN_names{n_rsn} ' ********'])
    lm_cognition   = fitlm(table_with_all_variables, 'Complexity_index ~ Flanker_Inhibition + CardSort_Flexibility + NBack_WorkingMemory + Ravens_FluidIntelligence + Relational_task');
    [y_pred,ci1]   = predict(lm_cognition,X_selected,'Alpha',0.05);
    
    % Stepwise regression for estimating the importance of each variable
    [b,se,pval,inmodel,stats,nextstep,history] = stepwisefit(X_selected,y,'inmodel',[true true true true true],'penter',.05/50,'premove',0.05/50);
    
    Corr_y(n_rsn)  = corr(y,y_pred, 'type','spearman');
    
    % Summarize the 3 outputs for each behavioural variable (beta, p-value, t-statistic)
    for n_beh = 1 : N_behav_selected
        
        out_measures(n_beh,n_rsn,1) = lm_cognition.Coefficients(n_beh+1,1).Estimate;
        out_measures(n_beh,n_rsn,2) = lm_cognition.Coefficients(n_beh+1,4).pValue;
        out_measures(n_beh,n_rsn,3) = lm_cognition.Coefficients(n_beh+1,3).tStat;
        
    end
    
    % Outputs of the fluid intelligence (var 4) only
    beta_FI(n_rsn) = lm_cognition.Coefficients(5,1).Estimate;
    p_FI(n_rsn)    = lm_cognition.Coefficients(5,4).pValue;
    t_FI(n_rsn)    = lm_cognition.Coefficients(5,3).tStat;
    
    %
    pf = polyfit( y, y_pred, 1);
    pv = polyval(pf,y);
    
    subplot(2,4,n_rsn),scatter(y, y_pred,'MarkerEdgeColor',[0 .5 .5],'MarkerFaceColor',[0 .7 .7],'LineWidth',1.5), axis square
    hold on, plot(y,pv,'--','linewidth',4)
    title(['Corr=' sprintf('%0.2f',Corr_y(n_rsn))]);
    
    if(n_rsn==1 || n_rsn==6)
        ylabel('Predicted Complexity Index', 'fontweight','bold')
    end
    
    if(n_rsn>5)
        xlabel('Original Complexity Index', 'fontweight','bold')
    end
    
    text(-3,.5,Yeo_RSN_names{n_rsn}, 'fontweight','bold'), ylim([-.6 .6]), xlim([-5.5 5.5])
    
end

%% Correct for multiple comparisons using FDR and replace pvalues with corrected pvalues in 'out_measures'
qvalue                    = 0.05;
[~, ~, ~, adj_pvalues_FI] = fdr_bh(p_FI,qvalue,'pdep','yes');
h_q                       = zeros(N_behav_selected, N_RSN);
pvalues_all               = zeros(N_behav_selected, N_RSN);
adj_pvalues_all           = zeros(N_behav_selected, N_RSN);
for n_beh = 1 : 5
    pvalues_beh                        = out_measures(n_beh,:,2); % p-values of the n'th behav variable for all 10 RSNs
    pvalues_all(n_beh, :)              = pvalues_beh;
    [h_q(n_beh, :), ~, ~, adj_pvalues] = fdr_bh(pvalues_beh,qvalue,'pdep','yes');
    adj_pvalues_all(n_beh, :)          = adj_pvalues;
    out_measures(n_beh,:,2)            = adj_pvalues;
end

h_q2 = adj_pvalues_all<qvalue;

disp('*** Results for fluid intelligence (var 4) ***')
T = table(Yeo_RSN_names', beta_FI, t_FI, adj_pvalues_FI,'VariableNames',{'RSN','beta','t_stat','adj_pval'})

disp('*** Results for all variables ***')
T2 = table(Yeo_RSN_names', squeeze(out_measures(1,:,:)), squeeze(out_measures(2,:,:)), squeeze(out_measures(3,:,:)), squeeze(out_measures(4,:,:)), squeeze(out_measures(5,:,:)),...
    'VariableNames',{'RSN','Flanker','CardSort','N_back','Fluid_intel','Relational'})

%% Permutation testing
if(strcmp(Downsamp, 'DS0'))
    if(~exist(Corr_Surr_mat_file,'file') || force_analysis)
        Corr_Surr = zeros(N_RSN, N_perm);
        for n_rsn = 1 : N_RSN
            
            y     = zscore(MSE_RSNs(:,n_rsn));
            
            tic
            %%% Permutation testing
            for n_perm = 1 : N_perm
                
                rand_vec = randperm(N_subj2);
                y_perm   = y(rand_vec);
                
                %% Multiple regression of five behavioural variables
                table_with_all_variables = table(y_perm,X_selected(:,1),X_selected(:,2),X_selected(:,3),X_selected(:,4),X_selected(:,5),...
                    'VariableNames',{'Complexity_index','Flanker_Inhibition','CardSort_Flexibility','NBack_WorkingMemory', 'Ravens_FluidIntelligence','Relational_task'});
                lm_cognition   = fitlm(table_with_all_variables, 'Complexity_index ~ Flanker_Inhibition + CardSort_Flexibility + NBack_WorkingMemory + Ravens_FluidIntelligence + Relational_task');
                [y_pred,~]   = predict(lm_cognition,X_selected,'Alpha',0.05);
                
                Corr_Surr(n_rsn,n_perm)  = corr(y,y_pred, 'type','spearman');
                
                clear y_pred lm_cognition table_with_all_variables
                
            end
            
            disp(['RSN ' num2str(n_rsn) ' was completd! Elapsed time = ' num2str(toc)])
            
        end
        
        save(Corr_Surr_mat_file,'Corr_Surr','Yeo_RSN_names','Corr_y')
        
    else
        
        out       = load(Corr_Surr_mat_file);
        Corr_Surr = out.Corr_Surr;
        Corr_y    = out.Corr_y;
        
    end
    
    figure,
    for n_rsn = 1 : N_RSN
        subplot(2,4,n_rsn), histogram(Corr_Surr(n_rsn,:),1000), hold on, line([Corr_y(n_rsn) Corr_y(n_rsn)],[0 100], 'color','red','linestyle','--','linewidth',3), xlim([-.25 .25]), ylim([0 30])
        title(sprintf(['Corr (y_{real} vs. y_{pred}) =  %0.2f%'],Corr_y(n_rsn)))
    end
    
    
end

%% Create brain maps of MSE-based complexity indices
%% Map1: Group-average of the subject-level ROIs
ref_nifti_path       = [Home_dir filesep 'HCP-MMP1_onMNI152_2mm_Glasser360.nii']; % Low-res Glasser atlas for comparison with the Neurosynth database
[refnii, refhdr]     = y_Read(ref_nifti_path);

% Zscore at the subject level after considering the possible NaN values
tmp_z = zeros(size(MSE_CI));
for n_subj = 1 : size(MSE_CI,2)
    tmp_z(:, n_subj) = (MSE_CI(:, n_subj) - nanmean(MSE_CI(:, n_subj))) / nanstd(MSE_CI(:, n_subj));
end
map_to_consider2     = nanmean(tmp_z, 2); % ROI-wise vector of group-level complexity according to the Glasser atlab
tmp_map              = zeros(size(refnii));
for n_roi = 1 : N_ROI
    tmp_map(round(refnii) == n_roi) = map_to_consider2(n_roi);
end

% Write the mp into a nifti format
y_Write(tmp_map, refhdr, [Brain_maps_folder filesep 'MSE_CI_brain_map_Glasser_r05_m' num2str(dim_m) '_' Downsamp '_Map_z.nii']);

% Print the ROIs with highrst and lowest complexy values
disp('*********** Map 2: ROIs with highest and lowest complexity:')
[map_to_consider_sorted, map_to_consider_sorted_ind] = sort(map_to_consider2, 'descend');
RSNs_with_max_CI                                     = yeoROIs_sorted(map_to_consider_sorted_ind(1:5));
RSNs_with_min_CI                                     = yeoROIs_sorted(map_to_consider_sorted_ind((end-4):end));

ROIs_with_max_CI                                     = ROI_labels(map_to_consider_sorted_ind(1:5), 2) % PFm and PG are parts of IPL (DMN)
ROIs_with_min_CI                                     = ROI_labels(map_to_consider_sorted_ind((end-4):end), 2)

%% Map2: Group-level variability of the subject-level ROIs
ref_nifti_path       = [Home_dir filesep 'HCP-MMP1_onMNI152_2mm_Glasser360.nii']; % Low-res Glasser atlas for comparison with the Neurosynth database
[refnii, refhdr]     = y_Read(ref_nifti_path);

% Select the map to split into percentiles
tmp                  = nanstd(MSE_CI,0,2);      % STD map over subjects
tmp2                 = std(nanmean(MSE_CI,2));  % STD of group-average map
map_to_consider3     = tmp/tmp2;
tmp_map              = zeros(size(refnii));
for n_roi = 1 : N_ROI
    tmp_map(round(refnii) == n_roi) = map_to_consider3(n_roi);
end

% Write the mp into a nifti format
y_Write(tmp_map, refhdr, [Brain_maps_folder filesep 'MSE_CI_brain_map_Glasser_r05_m' num2str(dim_m) '_' Downsamp '_Map_sigma.nii']);
spm_smooth([Brain_maps_folder filesep 'MSE_CI_brain_map_Glasser_r05_m' num2str(dim_m) '_' Downsamp '_Map_sigma.nii'], ...
    [Brain_maps_folder filesep 'MSE_CI_brain_map_Glasser_r05_m' num2str(dim_m) '_' Downsamp '_Map_sigma_smooth.nii'], [2 2 2]);

% Print the ROIs with highrst and lowest complexy values
disp('*********** Map 3: ROIs with highest and lowest variability across subjects:')
[map_to_consider_sorted, map_to_consider_sorted_ind] = sort(map_to_consider3, 'descend');
RSNs_with_max_variability                                     = yeoROIs_sorted(map_to_consider_sorted_ind(1:5));
RSNs_with_min_variability                                     = yeoROIs_sorted(map_to_consider_sorted_ind((end-4):end));

ROIs_with_max_variability                                     = ROI_labels(map_to_consider_sorted_ind(1:5), 2) % PFm and PG are parts of IPL (DMN)
ROIs_with_min_variability                                     = ROI_labels(map_to_consider_sorted_ind((end-4):end), 2)


