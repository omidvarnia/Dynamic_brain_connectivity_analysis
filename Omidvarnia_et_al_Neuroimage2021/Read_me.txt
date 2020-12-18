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

This MATLAB package includes all necessary scripts and functions for replicating the 
analysis steps of Omidvarnia et al, Neuroimage 2021 and reproducing its figures.

This study uses the datasets of 1000 HCP subjects with four resting state runs per subject.
Please see the paper for more details about the analysis pipelines and data preprocessing. 

The folder structure of input datasets is as follows: The datasets of each subject are located in
a single folder with the name of the subject ID (e.g., 996782). Each subject-specific folder 
includes four sub-folders named as REST1_LR, REST1_RL, REST2_LR and REST2_RL associated with
four resting state fMRI runs. Each dataset is assumed to be of size N_ROI x N_T where N_ROI
is the number of brain parcels (here, 379) and N_T is the number of time points or TRs (here, 1200).

Description of the MATLAB scripts:

- Extract_MSE_of_fMRI: This script generates Figure 3, Figure 4, Figure 5 and Figures S3 to
S5 of the manuscript. For reproducing Figure 3 and Figure 4, please make
sure that the MSE analysis results of head motion have been already
generated and saved in the approperiate mat file through the scariot
'Extract_MSE_of_Motion.m'.

- Extract_MSE_of_Motion: This script extracts Multiscale entropy from head motion (franewise displacement) 
of HCP database (N = 1000, four rest runs per subject) using two
tolerance parameters (r = 0.15, 0.5) and 9 embedding dimensions
(m = 2 to 10) over 25 time scales. The results are saved in mat files and
are loaded for generating Figure 3 and Fugure 4 in the script
'Extract_MSE_of_fMRI.m'.

- Complexity_FC_analysis: This script reproduces Figure 8, representing the relationship between
temporal complexity of fMRI and functional connectivity for the following 
parameters: r = 0.5, m = 2 and no downsampling.

- Test_retest_analysis: This script implements tes-retest analysis of the paper and reproduces Figure S5    
for two tolerance values (r = 0.15, 0.5), three downsampling options    
(no DS, DS at the rate of 2, DS at the rate of 4) and an embedding dimension of m=2:4.

- Effect_size_analysis: This script performs effect size analysis of the paper and reproduces Figure 6.

- Behavioural_analysis: This script reproduces Figure S4 as well as the results of Tables S7 
to S12 summarizing the relationship between rsfMRI temporal complexity 
and five behavioural measures for two tolerance values (r = 0.15, 0.5),  
three downsampling options (no DS, DS at the rate of 2, DS at the rate  
of 4) and an embedding dimension of m=2.
