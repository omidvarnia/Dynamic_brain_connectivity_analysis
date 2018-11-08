clear; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

%%% This script will enable you to run analysis pipeline used in Pedersen et al. BioRxiv 403105 %%%
%%%   We used fMRI and behavioral data from 1003 subjects from HCP (www.humanconnectome.org)    %%%
%%%         If you have any queries please contact me at mangor.pedersen@florey.edu.au          %%%
%%%              Below are a set of web-links needed to download external functions             %%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% download external function 'multiord','postprocess_ordinal_multilayer' and 'iterated_genlouvain' at https://github.com/GenLouvain/GenLouvain
% download external function 'flexibility' at http://commdetect.weebly.com/
% download external function 'phaseran2' at https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/32621/versions/1/previews/phaseran.m/index.html
% download external function 'SampEn' at https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/35784/versions/1/previews/SampEn.m/index.html
% In our paper we also tested whether behavioural data from HCP (www.humanconnectome.org) predicted 'network_switching'. The built-in MATLAB function 'lasso' can be used for this purpose

%% preliminaries
load('fMRI_data_all_subjects.mat'); % load 3D fMRI data; in our analysis the size of the 3D array was [1003,4800,25] (subjects,time-points,nodes)
n_subjects = size(fMRI_data_all_subjects,1); % total number of subjects
time_points = size(fMRI_data_all_subjects,2); % total number of time-points in fMRI data
N_nodes = size(fMRI_data_all_subjects,3); % number of brain nodes in fMRI data
window_overlap = 1; % overlap between sliding-windows in number number of TRs
window_length = round(139/window_overlap); % total window length (here, 139 time-points correspond to 100s with a TR of 0.72s)
phase_randomisation_iterations = 500; % number of phase randomisations for dynamic connectivity analysis
intralayer_resolution = 1; % intralayer resolution for multilayer modularity
interlayer_resolution = 1; % interlayer resolution for multilayer modularity

%% pre-allocate arrays - this is the output data (also saved at the end of script)
network_switching = zeros(n_subjects,N_nodes); % network switching data
Q_value = zeros(1,n_subjects); % Q-value of networks
number_of_communities = zeros(n_subjects,(time_points/window_overlap)-window_length); % number of communities
modularity_iterations = zeros(1,n_subjects); % number of iterations needed before the multilayer modularity algorithm converges
dynamic_connectivity_p_values_FDR_corr = zeros(n_subjects,N_nodes); % node degree of dynamic connections
absolute_correlation_at_switch_points = zeros(1,n_subjects); % correlation coefficient at time-points with network switching
absolute_correlation_at_no_switch_points = zeros(1,n_subjects); % correlation coefficient at time-points with no network switching
Entropy_top5 = zeros(1,n_subjects); % average entropy of fMRI conncetivity time-series with most switching (top 5 nodes)
Entropy_bottom5 = zeros(1,n_subjects); % average entropy of fMRI conncetivity time-series with least switching (bottom 5 nodes)
percentage_dynamic_connections = zeros(1,n_subjects); % percentage of statstically significant dynamic connections (corrected with False Discovery Rate)

parallel_loop_info = parpool(); % this will intitialize the parallel pool used next
parallel_loop_NumWorkers = parallel_loop_info.NumWorkers; % number of workers are needed to estimate total time-needed to run this script (see line #202)

parfor n = 1:n_subjects
    reverseStr = '';
    theElapsedTime = tic;
    
    %% Time-varying connectivity (A is for dynamic connectivity; AA is for multilayer modularity)
    A = zeros((time_points/window_overlap)-window_length,N_nodes,N_nodes); % pre-allocate connectivity matrices - multilayer modularity
    AA = cell(1,(time_points/window_overlap)-window_length); % pre-allocate connectivity matrices - dynamic connectivity
    time_varying_absolute_correlation = zeros(1,N_nodes);
    fMRI_data_current_subject = squeeze(fMRI_data_all_subjects(n,:,:)); % extract fMRI data from subject n
    
    for i = 1:(time_points/window_overlap)-window_length
        current_window = (squeeze(fMRI_data_current_subject((i:i+window_length-1)*window_overlap,:))).*repmat(hamming(window_length),1,N_nodes); % current fMRI window - use hamming window
        sliding_window = corrcoef(current_window); % correlation-based sliding window analysisw on fMRI data
        time_varying_absolute_correlation(i,:) = nanmean(abs(sliding_window)); % average absolute correlation, at time t
        A(i,:,:) = sliding_window; % matrices for dynamic connectivity analysis
        sliding_window(sliding_window<0) = 0; % remove negative correlation values for multilayer modularity analysis
        AA{i} = sliding_window; % matrices for multilayer modularity analysis
        
    end

    %% Ordinal iterative multilayer modularity calculation
    [B,mm] = multiord(AA,intralayer_resolution,interlayer_resolution); % B=supra-adjacency matrix
    PP = @(S) postprocess_ordinal_multilayer(S,(time_points/window_overlap)-window_length); % multilayer modularity set-up 
    [S,Q1,mod_iter_tmp] = iterated_genlouvain(B,10000,0,1,'moverandw',[],PP); % S = modularity assignments; Q1 = modularity value; mod_iter_tmp = iterations
    S = reshape(S,N_nodes,(time_points/window_overlap)-window_length); % 2D representation of modularity assignments
    number_of_communities(n,:) = sum(diff(sort(S)) ~= 0) +1; % total number of community in network
    network_switching(n,:) = flexibility(S'); % network switching
    Q_value(n) = Q1/mm; % adjusted modularity value of spatiotemporal network
    modularity_iterations(n) = mod_iter_tmp; % number of iterations before optimal modularity convergence
    
    %% Dynamic connectivity calculation
    fMRI_connectivity_2D = permute(A,[2 3 1]);
    fMRI_connectivity_2D = reshape(fMRI_connectivity_2D,[],size(A,1),1); % fMRI connectivity data - from 3D to 2D array
    fMRI_connectivity_2D(fMRI_connectivity_2D==1) = 0; % remove matrix 'diagonal' values
    fMRI_connectivity_std = std(fMRI_connectivity_2D,[],2)'; % dynamic connectivity analysis - i.e, variance of fMRI connectivity time-series
        
    %% Phase randomise dynamic connectivity data with N-iterations
    fMRI_data_randomised = squeeze(phaseran2(fMRI_data_current_subject,phase_randomisation_iterations)); % phase randomisation procedure retaining correlational structure in the data 
    fMRI_data_randomised = padarray(fMRI_data_randomised,1,'post');
    fMRI_connectivity_std_randomised = zeros(phase_randomisation_iterations,N_nodes*N_nodes); % allocate arrays to phase randomised data
    
    for nn = 1:phase_randomisation_iterations
        sliding_window_randomised = zeros((time_points/window_overlap)-window_length,N_nodes,N_nodes); % correlation-based sliding window analysis for phase randomised data
        
        for i = 1:(time_points/window_overlap)-window_length % sliding-windows of 500 phase randomised surrogates
            current_window_randomised = (squeeze(fMRI_data_randomised((i:i+window_length-1)*window_overlap,:,nn))).*repmat(hamming(window_length),1,N_nodes); % current fMRI window - use hamming window
            sliding_window_randomised(i,:,:) = corrcoef(current_window_randomised); % correlation-based sliding window analysis for phase randomised fMRI data
            
        end
        
        fMRI_connectivity_2D_randomised = permute(sliding_window_randomised,[2 3 1]);
        fMRI_connectivity_2D_randomised = reshape(fMRI_connectivity_2D_randomised,[],size(sliding_window_randomised,1),1); % fMRI connectivity surrogate data - from 3D to 2D array
        fMRI_connectivity_2D_randomised(fMRI_connectivity_2D_randomised==1) = 0; % convert diagonal values from 1 to 0
        fMRI_connectivity_std_randomised(nn,:) = std(fMRI_connectivity_2D_randomised,[],2)'; % null data - i.e. the variance of surrogate time-series
        
    end
    
    % line #101-114: convert all fMRI std data to symmetric matrices (real and randomised data)
    fMRI_connectivity_std_matrix = zeros(N_nodes,N_nodes);
    fMRI_connectivity_std_randomised_matrix = zeros(phase_randomisation_iterations,N_nodes,N_nodes);
    
    [qq1,qq2] = meshgrid(1:N_nodes,1:N_nodes);
    qq1 = qq1(:); qq2 = qq2(:); % all matrix indices
    
    for nn = 1:length(qq1)
        fMRI_connectivity_std_matrix(qq1(nn),qq2(nn)) = fMRI_connectivity_std(nn);
        
        for ii = 1:size(fMRI_connectivity_std_randomised,1)
            fMRI_connectivity_std_randomised_matrix(ii,qq1(nn),qq2(nn)) = fMRI_connectivity_std_randomised(ii,nn); 
            
        end
    end
    
    % line #117-128: only consider upper triangle of matrix for statistical testing
    mask = triu(true(N_nodes,N_nodes),1);
    [tmp1,tmp2] = find(mask);
    ind = [tmp1,tmp2];
    
    x = fMRI_connectivity_std_matrix(mask);
    y = zeros(length(x),size(fMRI_connectivity_std_randomised,1));
    
    for ii = 1:size(fMRI_connectivity_std_randomised,1)
        sliding_window = squeeze(fMRI_connectivity_std_randomised_matrix(ii,:,:));
        y(:,ii) = sliding_window(mask);
        
    end
    
    %% Statistically quantify dynamic connectivity with false discovery rate (original vs. phase randomised data)
    [~,idx] = sort([y,x],2,'ascend'); % rank each std connection agains 500 phase randomizations
    [~,mx]  = max(idx,[],2);
    pvalues_uncorr = 1-(mx-0.5)/size([y,x],2); % uncorrected p-values
    pvalues_uncorr_sorted = sort(pvalues_uncorr); % sort uncorrected p-values

    p_thr = (1:size(ind,1)) * 0.05 / size(ind,1); % range of p-thrshold
    rejection_cutoff = pvalues_uncorr_sorted' <= p_thr; % find rejection threshold
    rejection_cutoff_index = find(rejection_cutoff,1,'last'); % find greatest significant pvalue
    crit_pvalue = pvalues_uncorr_sorted(rejection_cutoff_index); % critical p-value
    pvalues_fdr_corrected_tmp = pvalues_uncorr <= crit_pvalue; % p-values corrected with false discovery rate
    
    pvalues_fdr_corrected_mat_tmp = zeros(N_nodes,N_nodes);
    
    for ii = 1:length(pvalues_fdr_corrected_tmp)
        pvalues_fdr_corrected_mat_tmp(ind(ii,1),ind(ii,2)) = pvalues_fdr_corrected_tmp(ii); % corrected p-values as a matrix
        
    end
    
    dynamic_connectivity_p_values_FDR_corr(n,:) = nansum([pvalues_fdr_corrected_mat_tmp pvalues_fdr_corrected_mat_tmp'],2); % node degree of significant dynamic connections    
    percentage_dynamic_connections(n) = ((nnz(pvalues_fdr_corrected_tmp)/length(pvalues_fdr_corrected_tmp))) * 100; % percentage dynamic connections for subject n

    %% calculate absolute correlation at time-points with switching versus no switching    
    corr_switching_mean = zeros(1,size(time_varying_absolute_correlation,2));
    corr_no_switching_mean = zeros(1,size(time_varying_absolute_correlation,2));
    
    for i = 1:size(time_varying_absolute_correlation,2)
        r1 = squeeze(time_varying_absolute_correlation(:,i));
        
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
    
    absolute_correlation_at_switch_points(n) = nanmean(corr_switching_mean); % correlation when node switch networks, averaged across all time points
    absolute_correlation_at_no_switch_points(n) = nanmean(corr_no_switching_mean); % correlation when node does not switch networks, averaged across all time points
 	
 	%% calculate entropy of fMRI dynamic connectivity time-series in five nodes with most and least network switching 
    SampleEntropy = zeros(1,size(fMRI_connectivity_2D,1));
    
    for i = 1:size(fMRI_connectivity_2D,1) % NB! SampEn can be time-consuming when dealing with long time-series and/or many fMRI connections
        tmp_data = squeeze(fMRI_connectivity_2D(i,:));
        SampleEntropy(i) = SampEn(2,0.2.*std(tmp_data),tmp_data); % calculate sample entropy for all connection-pairs
        
    end
    
    SampleEntropy_matrix = zeros(N_nodes,N_nodes);

    for nn = 1:length(qq1)
        SampleEntropy_matrix(qq1(nn),qq2(nn)) = SampleEntropy(nn); % sample entropy in a matrix form
        
    end
    
	[~,indices_for_entropy] = sort(network_switching(n,:)); % sort nodes in order of how often they switch between networks
	Entropy_top5(n) = nanmean(SampleEntropy_matrix(indices_for_entropy(N_nodes-4:N_nodes))); % entropy for the 5 nodes with most network switching
	Entropy_bottom5(n) = nanmean(SampleEntropy_matrix(indices_for_entropy(1:5))); % entropy for the 5 nodes with least network switching
	
 	theElapsedTime = toc(theElapsedTime);
    theElapsedTime = theElapsedTime/60;
    fprintf('\n\t In subject %d/%d, switching occured %gpct. of time with %gpct. dynamic connections; current loop took %gmin. with an estimated total run time of %ghrs. ... \n',...
            n,n_subjects,mean(network_switching(n,:))*100,percentage_dynamic_connections(n),theElapsedTime,((n_subjects*nanmean(theElapsedTime))/60)/parallel_loop_NumWorkers);

end

%% Save output data
save(['switching_and_dynamics_analysis with_' num2str(N_nodes) '_nodes_' num2str(time_points) '_timepoints_and_' num2str(n_subjects) '_subjects.mat']);