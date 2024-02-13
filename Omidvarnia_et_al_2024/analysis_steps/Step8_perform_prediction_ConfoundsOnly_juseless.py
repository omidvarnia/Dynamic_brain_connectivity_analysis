"""
Preform prediction modelling of PCA-reduced FC features on Juseless.

This script performs prediction modelling of PCA-reduced FC features on
Juseless. The prediction is performed using the following models:
    - Ridge regression
    - Ridge classifier
    - Random forest
    - Heuristic SVM

Parameters
----------
model_type : str
    Type of the preditive model: ridge, rf, svm
target_name : str
    Name of the desired target
N_subj_all : int
    Population size of the analysis

Usage
-----
    Step8_perform_prediction_ConfoundsOnly.py --model_type <model_type> \
        --target_name <target_name> --N_subj_all <N_subj_all> \
        
Usage example
-------
    $ python Step8_perform_prediction_ConfoundsOnly.py \
        --model_type ridge \
        --target_name  31-0.0 \ # Sex
        --N_subj_all 1000 \

Note:
    This script assumes the availability of a configuration file specified in
    the YAML format.
    
Written by: Amir Omidvarnia, PhD
Email: amir.omidvarnia@gmail.com
"""
import numpy as np
import pandas as pd
from pathlib import Path
import os, yaml, joblib, argparse
import sys
import warnings
warnings.simplefilter("ignore")

sys.path.append((Path(__file__).parent.parent / "common").as_posix())
from complexity import complexity_analysis

# ----- Input arguments
parser = argparse.ArgumentParser(description="Preform prediction modelling of "
                                             "rsfMRI features on Juseless. ")

parser.add_argument(
    "--model_type", 
    required=True, 
    type=str, 
    help="Type of the preditive model: ridge, rf, svm"
)

parser.add_argument(
    "--target_name", 
    required=True, 
    type=str, 
    help="Name of the desired target"
)

parser.add_argument(
    "--N_subj_all", 
    required=True, 
    type=int, 
    help="Population size of the analysis"
)

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
model_type = args.model_type
target_name = args.target_name
N_subj_all = args.N_subj_all
confound_removal = -1 # Confounds Only

assert model_type in ['ridge', 'rf', 'svm'], \
    "model_type must be one of the following: ridge, rf, svm"
assert isinstance(target_name, str), \
    "target_name must be a string"
assert isinstance(N_subj_all, int), \
    "N_subj_all must be an integer"

# from ptpython.repl import embed
# print('Stop: Step8_perform_prediction_ConfoundsOnly_juseless.py')
# embed(globals(), locals())

# ----- Input arguments
yaml_file = Path(
    ("/home/aomidvarnia/GIT_repositories/Comm_Biol_manuscript/analysis_codes"
     "/common/config_HCP_icbm152_mask.yaml")
)
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)

# Atlas name
atlas_name = 'glasser'
assert atlas_name in ['schaefer', 'glasser'], 'Atlas name is not correct.'

# ----- Parcellation and graph theory analysis parameters
# Spatial resolution of the parcellation atlas
resolution_mm = \
    config['feature_extraction'][f'brain_atlas_{atlas_name}']['resolution_mm']
# Number of networks in the parcellation atlas
N_network = \
    config['feature_extraction'][f'brain_atlas_{atlas_name}']['N_network']
# Number of desired ROIs in the atlas
N_ROI = config['feature_extraction'][f'brain_atlas_{atlas_name}']['N_ROI']

assert atlas_name in ['schaefer', 'glasser'], \
    "Please provide a valid atlas name!"
assert N_network in [None, 7, 17], \
    "Please provide a valid number of networks!"
assert resolution_mm in [None, 1, 2], \
    "Please provide a valid resolution in mm!"
assert N_ROI in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 360], \
    "Please provide a valid number of ROIs!"

# ------ Main folder
main_folder = config['juseless_paths']['main_folder']
if not os.path.isdir(main_folder):
    raise ValueError(f"Folder {main_folder} does not exist!")

# ----- Folder of the subject-specific .npz file
npzfiles_folder = config['juseless_paths'][f'npzfiles_folder_{atlas_name}']
if not os.path.isdir(npzfiles_folder):
    raise ValueError(
        (f"Folder {npzfiles_folder} does not exist! Please provide the "
          "path to the folder containing the subject-specific .npz files.")
    )

# ----- Folder of the mean maps and nifti files
prediction_results_folder = \
    config['juseless_paths'][f'prediction_results_folder_{atlas_name}']
    
prediction_results_folder = os.path.join(
    prediction_results_folder, f'prediction_results_{model_type}'
)

if not os.path.isdir(main_folder):
    raise ValueError(f"Folder {main_folder} does not exist!")

if not os.path.isdir(prediction_results_folder):
    raise ValueError(f"Folder {prediction_results_folder} does not exist!")

# ----- Folder of the TIV csf file for ukb subjects
# This info has been extracted by Felix et al and is 
# already available on Juseless. The dataset can be
# obtained through datalad at this address:
# http://ukb.ds.inm7.de/#~cat_m0wp1 according to VK.
TIV_filename = config['juseless_paths']['TIV_filename']
if not os.path.isfile(TIV_filename):
    raise ValueError(f"File {TIV_filename} does not exist!")

# Make a balanced population with equal number of males and females
N_subj_males_or_females = int(N_subj_all/2)
assert isinstance(N_subj_males_or_females, int), \
    "Please provide an integer for N_subj_males_or_females!"

# ----- Features and targets for prediction
# Prediction on healthy subjects after excluding ICD10 conditions
# Ref: 
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0154222
prediction_params = config['prediction']

# A npy file containing a dictionary of dataframes for the concatenated 
# brain maps of all subjects for all complexity measures
brain_maps_all_file = os.path.join(
            npzfiles_folder, 'complex_measures_all_df.npy')
target_names = config['prediction']['target_names']
target_labels = config['prediction']['target_labels']
problem_types = config['prediction']['problem_types']
taget_label = target_labels[target_names.index(target_name)]
problem_type = problem_types[target_names.index(target_name)]

if(target_name=='AgeAtScan'): # Age prediction
    confounds = ['31-0.0', 'TIV']
elif(target_name=='31-0.0'): # Sex prediction
    confounds = ['AgeAtScan', 'TIV']
elif(target_name=='TIV'): # GMV prediction
    confounds = ['AgeAtScan', '31-0.0']
else:
    confounds = ['AgeAtScan', '31-0.0', 'TIV']

# Predicive model parameters for grid search
model_grid = config['prediction']['model_grid']

# ----- Input codes for ukbb_parser
ukbb_parser_out_folder = os.path.join(main_folder, "ukbb_parser_files")
if not os.path.isdir(ukbb_parser_out_folder):
    raise ValueError(f"Folder {ukbb_parser_out_folder} does not exist!")

ukbb_parser_out_prefix = "ukbb_parser_complexity"
ukbb_parser_out_file = os.path.join(
            ukbb_parser_out_folder, 
            f'{ukbb_parser_out_prefix}_img_subs_only.csv'
)

# ----- Name of output filename including all predicition accuracies for
# ----- the requested target
accuracy_out_filename = os.path.join(
prediction_results_folder,
    (f'accuracy_{model_type}_Nsubj{N_subj_all}'
     f'_{taget_label}_ConfoundsOnly.npz')
)

prediction_results_folder = os.path.join(
    prediction_results_folder, f'prediction_results_{model_type}'
)

accuracy_joblib_filename = os.path.join(
    prediction_results_folder,
    (f'accuracy_{model_type}_Nsubj{N_subj_all}_'
     f'{taget_label}_ConfoundsOnly.joblib')
)

# from ptpython.repl import embed
# print("Stop: Step8_preform_prediction_ConfoundsOnly_juseless.py")
# embed(globals(), locals())

if not os.path.isfile(accuracy_out_filename):
    print(f"File {accuracy_out_filename} does not exist! Generating ...")
    # ------------------------------------------
    # Load the subject-specific .npz files
    # and concatenate subject-specific maps.
    # Make a dictionary of dataframes containing  
    # the brain maps of all subjects for all complexity
    # measures. It takes some time .....
    # ------------------------------------------
    dict_of_brain_maps_df = np.load(
        brain_maps_all_file, 
        allow_pickle=True)
    dict_of_brain_maps_df = dict_of_brain_maps_df.item(0)

    # ----------------------------------------------------------
    # Analysis starts ....
    # ----------------------------------------------------------
    ukbb_parser_DF_codes = prediction_params['ukbb_parser_DF_codes']
    img_subs_only = pd.read_csv(ukbb_parser_out_file, sep=',')

    # -------------------------------------------------
    # Load the TIV (Total Intracranial Volume) data from 
    # the database
    # -------------------------------------------------
    TIV_column = np.load(TIV_filename, allow_pickle=True)
    TIV_column = pd.DataFrame(TIV_column)
    TIV_column.columns = ['eid', 'TIV']
    TIV_column = TIV_column.astype({'eid':int})

    # -------------------------------------------------
    # Extract the FC matrices from a typical feature in dict_of_brain_maps_df.
    # Note that the fMRI features are not important here, because we are
    # only interested in the confounds as input features. 
    # -------------------------------------------------
    brain_map_df = dict_of_brain_maps_df['wCC']
    brain_map_df['eid'] = brain_map_df['eid'].astype(int)

    # ------------------------------------------
    # Generate the ROI-wise mean maps after tSNR thresholding
    # ------------------------------------------
    thresh_level = 0 # No thresholding
    models = {}

    brain_map_df_thresholded = brain_map_df
    _, N_ROI_supraThresh = brain_map_df_thresholded.shape
    N_ROI_supraThresh = N_ROI_supraThresh - 1
    
    # -------------------------------------------------
    # Merging the maps with ukbb_parser and making a 
    # Julearn-compatible df. It also addes TIV as a
    # confound in the dataframe
    # -------------------------------------------------
    df_julearn = complexity_analysis.merge_ukbb_complexity(
        img_subs_only,
        brain_map_df_thresholded,
        TIV_column
    )

    # -------------------------------------------------
    # Make a balanced population with equal number of 
    # male and females
    # -------------------------------------------------
    df_julearn, mean_age_males, mean_age_females = \
        complexity_analysis.make_balanced_classes(
            df_julearn,
            target_name,
            N_subj_males_or_females
        )

    N_subj, _ = df_julearn.shape

    # -------------------------------------------------
    # Convert the target labels for classification, 
    # if needed.
    # -------------------------------------------------
    # from ptpython.repl import embed
    # print("Stop: Step5_preform_prediction_juseless.py")
    # embed(globals(), locals())

    if(problem_type=='multiclass_classification' or 
        problem_type=='binary_classification'):

        df_julearn = complexity_analysis.labelEncoder(
                df_julearn,
                problem_type,
                target_name
            )

    # Drop rows which may include NaN values
    # df_julearn = df_julearn.dropna(axis=0)

    # -------------------------------------------------
    # Pass the dataframe to Julearn for prediction
    # -------------------------------------------------
    model_trained, scores, scores_summary, inspector, mean_score, best_param, class_counts = \
        complexity_analysis.julearn_prediction(
            problem_type,
            target_name,
            df_julearn,
            model_grid,
            model_type,
            confound_removal,
            confounds
        )

    accuracy_all = mean_score
    scores_all = scores
    best_alpha_all = best_param
    N_ROI_supraThresh_all = N_ROI_supraThresh

    # -------------------------------------------------
    # Save the results along the way
    # -------------------------------------------------
    np.savez(
        accuracy_out_filename, 
        scores_all=scores_all,
        accuracy_all=accuracy_all,
        best_alpha_all=best_alpha_all,
        N_ROI_supraThresh_all=N_ROI_supraThresh_all
    )

    print(f"Saved the results in {accuracy_out_filename}!\n")
    print(f"Accuracies: {accuracy_all.T}")

else:

    print(f"The output file {accuracy_out_filename} already exists!")
