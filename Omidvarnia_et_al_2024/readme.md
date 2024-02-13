# Resting-State fMRI Features and Individual Characteristics for The Prediction of Behavioural Phenotypes in UK Biobank

This package contains scripts organized into different steps for performing various analyses associated with this paper:

- Omidvarnia et al, Comparison Between Resting-State fMRI Features and Individual Characteristics for The Prediction of Behavioural Phenotypes: A Machine Learning Perspective, Biorxiv 2024

Please contact [Dr. Amir Omidvarnia](https://github.com/omidvarnia) if you have any questions regarding the codes. 

## Analysis steps
- **Step1_set_initial_folders.py**: Python script for setting initial folders for analysis.
- **Step2_create_job_files_for_HTCondor.py**: Python script for creating job files for HTCondor.
- **Step3_submit_jobs_to_HTCondor.sh**: Shell script for submitting jobs to HTCondor.
- **Step4_obtain_distributions.py**: Python script for obtaining distributions.
- **Step5_preform_prediction_juseless_tsnr0.py**: Python script for performing prediction on rsfMRI features with tSNR threshold 0.
- **Step5_preform_prediction_juseless_tsnrAll.py**: Python script for performing prediction on rsfMRI features with all tSNR thresholds.
- **Step6_create_job_files_for_HTCondor.py**: Python script for creating job files for HTCondor.
- **Step7_submit_prediction_jobs_to_HTCondor.sh**: Shell script for submitting prediction jobs to HTCondor.
- **Step8_perform_prediction_ConfoundsOnly_juseless.py**: Python script for performing prediction with confound-only data on rsfMRI features.
- **Step9_create_job_files_for_HTCondor.py**: Python script for creating job files for HTCondor.
- **Step10_submit_ConfoundsOnly_prediction_jobs_to_HTCondor.sh**: Shell script for submitting prediction jobs with confound-only data to HTCondor.
- **Step11_plot_prediction_accuracies_allThreshs.py**: Python script for plotting prediction accuracies across all thresholds.
- **Step11_plot_prediction_accuracies_thr0_boxplot.py**: Python script for plotting prediction accuracies at threshold 0 as boxplots.
- **Step11_plot_prediction_accuracies_thr0.py**: Python script for plotting prediction accuracies at threshold 0.
- **Step12_fingerprinting_analysis.py**: Python script for performing fingerprinting analysis.
- **Step13_create_job_files_for_HTCondor.py**: Python script for creating job files for HTCondor.
- **Step14_submit_identification_jobs.sh**: Shell script for submitting identification jobs to HTCondor.
- **Step15_plot_identification_accuracies.py**: Python script for plotting identification accuracies.

## Common modules
- **complexity.py**: Python script containing functions for complexity analysis.
- **config_HCP_icbm152_mask.yaml**: YAML configuration file.
- **feature_extraction.py**: Python script containing functions for feature extraction.
- **htcondor_submit_file_generation.py**: Python script for generating HTCondor submit files.
- **LinearSVHeuristicC.py**: Python script containing a linear support vector heuristic function.
- **vec2nii.py**: Python script containing functions for converting vectors to NIfTI files.

## Additional data
- **MNI_Glasser_HCP_v1.0.nii.gz**: NIfTI file containing the Glasser parcellation atlas.
- **subs_with_filtered_func_data_clean_and_movement_and_warp_data_261121.mat**: MAT file containing subject data.

## An executable for feature extraction
- **fmri_feature_extraction_executable.py**: Python script for executing rsfMRI feature extraction.

## Perform additional analyses to the above steps
- **check_alpha_parameters.py**: Python script for checking alpha parameters.
- **plot_mean_tSNR_nifti.py**: Python script for plotting mean tSNR as a NIfTI file.
- **Step1B_check_tsnr_and_nroi_relationship.py**: Python script for checking the relationship between tSNR and NROI.
- **Step1_check_tsnr_and_nroi_relationship.py**: Python script for checking the relationship between tSNR and NROI.
- **Step2_check_tsnr_and_nroi_create_job_files.py**: Python script for creating job files to check tSNR and NROI.
- **Step3_check_tsnr_and_nroi_submit_job_files.sh**: Shell script for submitting job files to check tSNR and NROI.
- **Step4_plot_auuracy_vs_tsnr_N20k.py**: Python script for plotting accuracy vs. tSNR for N=20k.
- **Step5_plot_acuracy_vs_nroi_N20k.py**: Python script for plotting accuracy vs. NROI for N=20k.
