clear; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%%% This script will enable you to run analysis pipeline used in Pedersen et al. BioRxiv 403105 %%%
%%%   We used rsfMRI data of 1003 subjects from HCP downloaded from www.humanconnectome.org)    %%%
%%%         If you have any queries please contact me at mangor.pedersen@florey.edu.au          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% download external function 'multiord','postprocess_ordinal_multilayer' and 'iterated_genlouvain' at https://github.com/GenLouvain/GenLouvain
% download external function 'flexibility' at http://commdetect.weebly.com/
% download external function 'phaseran2' at https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/32621/versions/1/previews/phaseran.m/index.html
% download external function 'SampEn' at https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/35784/versions/1/previews/SampEn.m/index.html
% version 2 update October 2019 - added Elastic net model to correlate behavioural and fMRI data

%% preliminaries
load('fMRI_data_all_subjects.mat'); % load 3D fMRI data. In our analysis the size of the 3D array was [1003,4800,25] (subjects,time-points,nodes)
n_subjects = size(fMRI_data_all_subjects,1); % total number of subjects
time_points = size(fMRI_data_all_subjects,2); % total number of time-points in fMRI data
N_nodes = size(fMRI_data_all_subjects,3); % number of brain nodes in fMRI data
window_overlap = 1; % overlap between sliding-windows in number number of TRs
window_length = round(139/window_overlap); % total window length (here, 139 time-points correspond to 100s with a TR of 0.72s)
phase_randomisation_iterations = 500; % number of phase randomisations for dynamic connectivity analysis
intralayer_resolution = 1; % intralayer resolution for multi-layer modularity
interlayer_resolution = 1; % interlayer resolution for multi-layer modularity

%% pre-allocate arrays - this is the output data (also saved at the end of script)
network_switching = zeros(n_subjects,N_nodes); % network switching data
Q_value = zeros(1,n_subjects); % Q-value of networks
number_of_communities = zeros(n_subjects,(time_points/window_overlap)-window_length); % number of communities
modularity_iterations = zeros(1,n_subjects); % number of iterations needed before the multilayer modularity algorithm converges
dFC_p_values_FDR_corrected = zeros(n_subjects,N_nodes); % node degree of dynamic connections
switching_absolute_correlation = zeros(1,n_subjects); % correlation coefficient at time-points with network switching
no_switching_absolute_correlation = zeros(1,n_subjects); % correlation coefficient at time-points with no network switching
entropy_top5 = zeros(1,n_subjects); % average entropy of fMRI conncetivity time-series with most switching (top 5 nodes)
entropy_bottom5 = zeros(1,n_subjects); % average entropy of fMRI conncetivity time-series with least switching (bottom 5 nodes)
percentage_dynamic_connections = zeros(1,n_subjects); % percentage of statstically significant dynamic connections (corrected with False Discovery Rate)

parallel_loop_info = parpool(); % this will intitialize the parallel pool starting at line #40
parallel_loop_NumWorkers = parallel_loop_info.NumWorkers; % number of workers are needed to estimate total time-needed to run this script (see line #199)

parfor n = 1:n_subjects
    reverseStr = '';
    theElapsedTime = tic;
    
    %% Time-varying connectivity (A is for dynamic connectivity; AA is for multilayer modularity)
    A = zeros((time_points/window_overlap)-window_length,N_nodes,N_nodes); % pre-allocate connectivity matrices - multilayer modularity
    AA = cell(1,(time_points/window_overlap)-window_length); % pre-allocate connectivity matrices - dynamic connectivity
    r_abs = zeros(1,N_nodes);
    fMRI_data_current_subject = squeeze(fMRI_data_all_subjects(n,:,:)); % extract fMRI data from subject n
    
    for i = 1:(time_points/window_overlap)-window_length
        window = (squeeze(fMRI_data_current_subject((i:i+window_length-1)*window_overlap,:))).*repmat(hamming(window_length),1,N_nodes); % current fMRI window - use hamming window
        tmp = corrcoef(window); % correlation-based sliding window analysisw on fMRI data
        r_abs(i,:) = nanmean(abs(tmp)); % average absolute correlation, at time t
        A(i,:,:) = tmp; % matrices for dynamic connectivity analysis
        tmp(tmp<0) = 0; % remove negative correlation values for multilayer modularity analysis
        AA{i} = tmp; % matrices for multilayer modularity analysis
        
    end
    
    %% Ordinal iterative multilayer modularity calculation
    [B,mm] = multiord(AA,intralayer_resolution,interlayer_resolution); % B=supra-adjacency matrix
    PP = @(S) postprocess_ordinal_multilayer(S,(time_points/window_overlap)-window_length); % multilayer modularity set-up 
    [S,Q1,mod_iter_tmp] = iterated_genlouvain(B,10000,0,1,'moverandw',[],PP); % S=modularity assignments; Q1=modularity value; mod_iter_tmp=iterations
    S = reshape(S,N_nodes,(time_points/window_overlap)-window_length); % 2D representation of modularity assignments
    number_of_communities(n,:) = sum(diff(sort(S)) ~= 0) +1; % total number of community in network
    network_switching(n,:) = flexibility(S'); % network switching
    Q_value(n) = Q1/mm; % adjusted modularity value of spatiotemporal network
    modularity_iterations(n) = mod_iter_tmp; % number of iterations before optimal modularity convergence
    
    %% Dynamic connectivity calculation
    C = permute(A,[2 3 1]);
    C = reshape(C,[],size(A,1),1); % fMRI connectivity data - from 3D to 2D array
    C(C==1) = 0; % remove matrix 'diagonal' values
    dFC_std = std(C,[],2)'; % dynamic connectivity analysis - i.e, variance of fMRI connectivity time-series
        
    %% Phase randomise dynamic connectivity data with N-iterations
    fMRI_data_randomised = squeeze(phaseran2(fMRI_data_current_subject,phase_randomisation_iterations));
    fMRI_data_randomised = padarray(fMRI_data_randomised,1,'post'); % phase randomisation procedure retaining correlational structure in the data 
    dFC_std_randomised = zeros(phase_randomisation_iterations,N_nodes*N_nodes); % allocate arrays to phase randomised data
    
    for nn = 1:phase_randomisation_iterations
        cc_wind_randomised = zeros((time_points/window_overlap)-window_length,N_nodes,N_nodes); % correlation-based sliding window analysis for phase randomised data
        
        for i = 1:(time_points/window_overlap)-window_length % sliding-windows of 500 phase randomised surrogates
            window_randomised = (squeeze(fMRI_data_randomised((i:i+window_length-1)*window_overlap,:,nn))).*repmat(hamming(window_length),1,N_nodes); % current fMRI window - use hamming window
            cc_wind_randomised(i,:,:) = corrcoef(window_randomised); % correlation-based sliding window analysis for phase randomised fMRI data
            
        end
        
        C_randomised = permute(cc_wind_randomised,[2 3 1]);
        C_randomised = reshape(C_randomised,[],size(cc_wind_randomised,1),1); % fMRI connectivity surrogate data - from 3D to 2D array
        C_randomised(C_randomised==1) = 0; % convert diagonal values from 1 to 0
        dFC_std_randomised(nn,:) = std(C_randomised,[],2)'; % null data - i..e, variance of surrogate time-series
        
    end
    
    % line #98-111: convert all fMRI std data to symmetric matrices (real and randomised data)
    dFC_std_matrix = zeros(N_nodes,N_nodes);
    dFC_std_randomised_mat = zeros(phase_randomisation_iterations,N_nodes,N_nodes);
    
    [qq1,qq2] = meshgrid(1:N_nodes,1:N_nodes);
    qq1 = qq1(:); qq2 = qq2(:); % matrix indices
    
    for nn = 1:length(qq1)
        dFC_std_matrix(qq1(nn),qq2(nn)) = dFC_std(nn);
        
        for ii = 1:size(dFC_std_randomised,1)
            dFC_std_randomised_mat(ii,qq1(nn),qq2(nn)) = dFC_std_randomised(ii,nn); 
            
        end
    end
    
    % line #114-125: only consider upper triangle of matrix for statistical testing
    mask = triu(true(N_nodes,N_nodes),1);
    [tmp1,tmp2] = find(mask);
    ind = [tmp1,tmp2];
    
    x = dFC_std_matrix(mask);
    y = zeros(length(x),size(dFC_std_randomised,1));
    
    for ii = 1:size(dFC_std_randomised,1)
        tmp = squeeze(dFC_std_randomised_mat(ii,:,:));
        y(:,ii) = tmp(mask);
        
    end
    
    %% Statistically quantify dynamic connectivity with false discovery rate (original vs. phase randomised data)
    [~,idx] = sort([y,x],2,'ascend'); % rank each std connection agains 500 phase randomizations
    [~,mx]  = max(idx,[],2);
    pvals_uncorr = 1-(mx-0.5)/size([y,x],2); % uncorrected p-values
    pvals_uncorr_sorted = sort(pvals_uncorr); % sort uncorrected p-values

    thresh = (1:size(ind,1)) * 0.05 / size(ind,1); % range of p-thrshold
    rej = pvals_uncorr_sorted' <= thresh; % find rejection threshold
    max_id = find(rej,1,'last'); % find greatest significant pvalue
    crit_p = pvals_uncorr_sorted(max_id); % critical p-value
    pvals_fdr_corrected_tmp = pvals_uncorr <= crit_p; % p-values corrected with false discovery rate
    
    pvals_fdr_corrected_mat_tmp = zeros(N_nodes,N_nodes);
    
    for ii = 1:length(pvals_fdr_corrected_tmp)
        pvals_fdr_corrected_mat_tmp(ind(ii,1),ind(ii,2)) = pvals_fdr_corrected_tmp(ii); % corrected p-values as a matrix
        
    end
    
    dFC_p_values_FDR_corrected(n,:) = nansum([pvals_fdr_corrected_mat_tmp pvals_fdr_corrected_mat_tmp'],2); % node degree of significant dynamic connections    
    percentage_dynamic_connections(n) = ((nnz(pvals_fdr_corrected_tmp)/length(pvals_fdr_corrected_tmp))) * 100; % percentage dynamic connections for subject n

    %% calculate absolute correlation at time-points with switching versus no switching    
    corr_switching_mean = zeros(1,size(r_abs,2));
    corr_no_switching_mean = zeros(1,size(r_abs,2));
    
    for i = 1:size(r_abs,2)
        r1 = squeeze(r_abs(:,i));
        
        swtch = zeros(1,size(r1,1));
        
        for ii = 1:size(swtch,2)-1 % this loop calculated whether a node switch, or doesn't switch, network at time, t
            swtch(ii+1) = 1-ismember(S(i,ii),S(i,ii+1));
            
        end
        
        corr_swtch = r1 .* swtch'; 
        corr_swtch(corr_swtch <= 0) = NaN;
        corr_switching_mean(i) = nanmean(corr_swtch); % correlation when node switch networks, at time-point, t
        
        corr_no_swtch = r1 .* 1-swtch';
        corr_no_swtch(corr_no_swtch <= 0) = NaN;
        corr_no_switching_mean(i) = nanmean(corr_no_swtch); % correlation when node does not switch networks, at time-point, t
        
    end
    
    switching_absolute_correlation(n) = nanmean(corr_switching_mean); % correlation when node switch networks, averaged across all time points
    no_switching_absolute_correlation(n) = nanmean(corr_no_switching_mean); % correlation when node does not switch networks, averaged across all time points
 	
 	%% calculate entropy of fMRI dynamic connectivity time-series in five nodes with most and least network switching 
    SampleEntropy = zeros(1,size(C,1));
    
    for i = 1:size(C,1) % note SampEn can be time-consuming in long time-series
        tmp_data = squeeze(C(i,:));
        SampleEntropy(i) = SampEn(2,0.2.*std(tmp_data),tmp_data); % calculate sample entropy for all connection-pairs
        
    end
    
    SampleEntropy_matrix = zeros(N_nodes,N_nodes);

    for nn = 1:length(qq1)
        SampleEntropy_matrix(qq1(nn),qq2(nn)) = SampleEntropy(nn); % sample entropy in matrix form
        
    end
    
	[~,indices_for_entropy] = sort(network_switching(n,:)); % sort nodes how of they switch networks
	entropy_top5(n) = nanmean(SampleEntropy_matrix(indices_for_entropy(N_nodes-4:N_nodes))); % entropy for the 5 nodes with most network switching
	entropy_bottom5(n) = nanmean(SampleEntropy_matrix(indices_for_entropy(1:5))); % entropy for the 5 nodes with least network switching
	
 	theElapsedTime = toc(theElapsedTime);
    theElapsedTime = theElapsedTime/60;
    fprintf('\n\t In subject %d/%d, switching occured %gpct. of time with %gpct. dynamic connections; current loop took %gmin. with an estimated total run time of %ghrs. ... \n',...
            n,n_subjects,mean(network_switching(n,:))*100,percentage_dynamic_connections(n),theElapsedTime,((n_subjects*nanmean(theElapsedTime))/60)/parallel_loop_NumWorkers);

end

%% Elastic net - first normalize brain averaged switching between -1 and 1
Y_tmp = nanmean(network_switching,2); % average network switching for all subjects
Y = (Y_tmp-nanmean(Y_tmp))/(nanmax(Y_tmp)-min(Y_tmp)); % normalize network switching data (1 by subject vector - here, 1x1003 vector)

%% list of behavioural variables from HCP
% task1 = behaviouraldata.Flanker_Unadj;
% task2 = behaviouraldata.CardSort_Unadj;
% task3 = behaviouraldata.ProcSpeed_AgeAdj;
% task4 = behaviouraldata.WM_Task_Acc;
% task5 = behaviouraldata.PMAT24_A_CR;
% task6 = behaviouraldata.MMSE_Score;
% task7 = behaviouraldata.PSQI_AmtSleep;
% task8 = behaviouraldata.PicSeq_Unadj;
% task9 = behaviouraldata.ReadEng_Unadj;
% task10 = behaviouraldata.PicVocab_Unadj;
% task11 = behaviouraldata.DDisc_AUC_200;
% task12 = behaviouraldata.DDisc_AUC_40K;
% task13 = behaviouraldata.VSPLOT_CRTE;
% task14 = behaviouraldata.SCPT_SEN;
% task15 = behaviouraldata.SCPT_SPEC;
% task16 = behaviouraldata.IWRD_TOT;
% task17 = behaviouraldata.IWRD_RTC;
% task18 = behaviouraldata.ER40ANG;
% task19 = behaviouraldata.ER40HAP;
% task20 = behaviouraldata.ER40FEAR;
% task21 = behaviouraldata.ER40SAD;
% task22 = behaviouraldata.ER40NOE;
% task23 = behaviouraldata.AngAffect_Unadj;
% task24 = behaviouraldata.AngAggr_Unadj;
% task25 = behaviouraldata.AngHostil_Unadj;
% task26 = behaviouraldata.FearAffect_Unadj;
% task27 = behaviouraldata.FearSomat_Unadj;
% task28 = behaviouraldata.Sadness_Unadj;
% task29 = behaviouraldata.LifeSatisf_Unadj;
% task30 = behaviouraldata.PosAffect_Unadj;
% task31 = behaviouraldata.Friendship_Unadj;
% task32 = behaviouraldata.Loneliness_Unadj;
% task33 = behaviouraldata.PercHostil_Unadj;
% task34 = behaviouraldata.PercReject_Unadj;
% task35 = behaviouraldata.PercStress_Unadj;
% task36 = behaviouraldata.EmotSupp_Unadj;
% task37 = behaviouraldata.InstruSupp_Unadj;
% task38 = behaviouraldata.SelfEff_Unadj;
% task39 = behaviouraldata.Emotion_Task_Acc;
% task40 = behaviouraldata.Language_Task_Acc;
% task41 = behaviouraldata.Relational_Task_Acc;
% task42 = behaviouraldata.Social_Task_Perc_NLR;
% task43 = behaviouraldata.Social_Task_Perc_Random;
% task44 = behaviouraldata.Social_Task_Perc_TOM;
% task45 = behaviouraldata.Social_Task_Perc_Unsure;
% task46 = behaviouraldata.NEOFAC_A;
% task47 = behaviouraldata.NEOFAC_C;
% task48 = behaviouraldata.NEOFAC_E;
% task49 = behaviouraldata.NEOFAC_N;
% task50 = behaviouraldata.NEOFAC_O;

%% sort behavioural data in a subject by variables arrray (i.e., 1003x50 array)
num_var = 50; % total number of tasks
X = zeros(length(Y_tmp),num_var); % pre-allocate array

%% normalize behavioural data between -1 and 1
for n=1:num_var
    X(:,n) = (x(:,n)-nanmean(x(:,n)))/(nanmax(x(:,n))-nanmin(x(:,n)));
end

%% Elastic net regression model
lambda = 0:0.001:1; % L2 budget - i.e, lambda values
alpha = 0.5; % alpha: 1 = lasso; 0 = ridge regression; in-between 0 and 1 = elastic net
[B,FitInfo] = lasso(X,Y,'Alpha',alpha,'NumLambda',length(lambda),'Lambda',lambda,'CV',10); % elastic net regression with cross-validation of k=10
B_minMSE = B(:,FitInfo.IndexMinMSE); % beta values at minumum MSE based on cross-validation
y_pred = X'.*beta_weights+FitInfo.Intercept(FitInfo.IndexMinMSE); % prediction score
y_pred = y_pred.*(y_pred.*B_minMSE~=0); % prediction score - only for non-zero elastic net values

%% Save output data
save(['switching_and_dynamics_analysis with_' num2str(N_nodes) '_nodes_' num2str(time_points) '_timepoints_and_' num2str(n_subjects) '_subjects.mat']);