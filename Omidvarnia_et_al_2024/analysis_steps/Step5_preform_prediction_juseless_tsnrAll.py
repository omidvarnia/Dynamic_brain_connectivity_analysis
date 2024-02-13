"""
Preform prediction modelling of PCA-reduced FC features on Juseless.

This script preforms prediction modelling of PCA-reduced FC features on
Juseless. The script is designed to be run on Juseless. The script
requires the following input arguments:

Parameters
----------
model_type : str
    Type of the preditive model: ridge, rf, svm
target_name : str
    Name of the desired target
N_subj_all : int
    Population size of the analysis
confound_removal : int
    0 for no remove, 1 for confound removal, 2 for features+confounds
feature_name : str
    Desired rsfMRI feature name

Usage example
-------------
$ python Step5_preform_prediction_juseless.py \
    --model_type ridge \
    --target_name 31-0.0 \ # Sex
    --N_subj_all 1000 \
    --confound_removal 0 \
    --feature_name RangeEnB_AUC_fMRI

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
yaml_file = Path(
    ("/home/aomidvarnia/GIT_repositories/Comm_Biol_manuscript/analysis_codes"
     "/common/config_HCP_icbm152_mask.yaml")
)
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)

# Atlas name
atlas_name = 'schaefer'
assert atlas_name in ['schaefer', 'glasser'], 'Atlas name is not correct.'

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

parser.add_argument(
    "--confound_removal", 
    required=True, 
    type=int, 
    help="0 for no remove, 1 for confound removal, 2 for features+confounds"
)

parser.add_argument(
    "--feature_name", 
    required=True, 
    type=str, 
    help="Desired rsfMRI feature name"
)

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
model_type = args.model_type
target_name = args.target_name
feature_name = args.feature_name
confound_removal = args.confound_removal
N_subj_all = args.N_subj_all

assert model_type in ['ridge', 'rf', 'svm'], \
    "model_type must be one of the following: ridge, rf, svm"
assert isinstance(target_name, str), \
    "target_name must be a string"
assert isinstance(feature_name, str), \
    "feature_name must be a string"
assert isinstance(N_subj_all, int), \
    "N_subj_all must be an integer"
assert confound_removal in [0, 1, 2], \
    "confound_removal must be one of the following: 0, 1, 2"

# ----- Parcellation and graph params
N_network = config['feature_extraction'][f'brain_atlas_{atlas_name}']['N_network']
N_ROI = config['feature_extraction'][f'brain_atlas_{atlas_name}']['N_ROI']

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

# ----- Parcellation atlas folder
parcel_atlas_folder = config['juseless_paths']['parcel_atlas_folder']
if not os.path.isdir(parcel_atlas_folder):
    raise ValueError(
        (f"{parcel_atlas_folder} including the parcellation atlas and the "
          "associated ROI labels does not exist!")
    )

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
feature_names = prediction_params['feature_names']
N_features = len(feature_names)
if(prediction_params['feature_names'][0] != 'tSNR_fMRI'):
    raise ValueError('The first measure must be ''tSNR''.')

# A npy file containing a dictionary of dataframes for the concatenated 
# brain maps of all subjects for all complexity measures
brain_maps_all_file = os.path.join(
            npzfiles_folder, 'complex_measures_all_df.npy')
target_names = config['prediction']['target_names']
target_labels = config['prediction']['target_labels']
problem_types = config['prediction']['problem_types']
taget_label = target_labels[target_names.index(target_name)]
problem_type = problem_types[target_names.index(target_name)]
thresh_vec = config['prediction']['thresh_vec']

N_thresh = len(thresh_vec)

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
if(confound_removal==0):
    accuracy_out_filename = os.path.join(
        prediction_results_folder,
        (f'accuracy_{model_type}_{feature_name}_Nsubj{N_subj_all}'
         f'_{taget_label}_NoConfoundRemoved_tsnrAll.npz')
    )
    accuracy_joblib_filename = os.path.join(
        prediction_results_folder,
        (f'accuracy_{model_type}_{feature_name}_Nsubj{N_subj_all}'
         f'_{taget_label}_NoConfoundRemoved_tsnrAll.joblib')
    )
elif(confound_removal==1):
    accuracy_out_filename = os.path.join(
    prediction_results_folder,
    (f'accuracy_{model_type}_{feature_name}_Nsubj{N_subj_all}_'
     f'{taget_label}_confoundRemoved_tsnrAll.npz')
    )
    accuracy_joblib_filename = os.path.join(
        prediction_results_folder,
    (f'accuracy_{model_type}_{feature_name}_Nsubj{N_subj_all}_'
     f'{taget_label}_confoundRemoved_tsnrAll.joblib')
    )
elif(confound_removal==2): # Combined features of Confounds and brain maps
    accuracy_out_filename = os.path.join(
    prediction_results_folder,
    (f'accuracy_{model_type}_{feature_name}_Nsubj{N_subj_all}_'
     f'{taget_label}_CombinedFeature_tsnrAll.npz')
    )
    accuracy_joblib_filename = os.path.join(
        prediction_results_folder,
    (f'accuracy_{model_type}_{feature_name}_Nsubj{N_subj_all}_'
     f'{taget_label}_CombinedFeature_tsnrAll.joblib')
    )
else:
    raise ValueError(
        ('confound_removal must be either 0 (no confound removal), 1 '
         '(confound removal), or 2 (combined confounds and brain maps) '
         'for Step 5.'
        )
    )

# from ptpython.repl import embed
# print("Stop: Step5_preform_prediction_juseless_thrAll.py")
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
    # Extract the FC matrices from dict_of_brain_maps_df 
    # -------------------------------------------------
    brain_map_df = dict_of_brain_maps_df[feature_name]
    brain_map_df['eid'] = brain_map_df['eid'].astype(int)
    
    # -------------------------------------------------
    # Extract the tSNR thresholding map
    # -------------------------------------------------
    brain_map_df_tSNR = dict_of_brain_maps_df['tSNR_fMRI']
    brain_map_df_tSNR['eid'] = brain_map_df_tSNR['eid'].astype(int)
    tSNR_brain_map_df, tSNR_map_df_mean_ROI_normalized = \
        complexity_analysis.tSNR_MinMaxScaler(brain_map_df_tSNR)

    # ------------------------------------------
    # Initialize the variables for the threshlding loop
    # ------------------------------------------
    accuracy_all = np.zeros((N_thresh, 1))
    N_ROI_supraThresh_all = np.zeros((N_thresh, 1))
    scores_all = dict()
    best_alpha_all = np.zeros((N_thresh, 1))

    # ------------------------------------------
    # Generate the ROI-wise mean maps after tSNR thresholding at thr=0
    # ------------------------------------------
    for n_thresh in range(len(thresh_vec)):

        thresh_level = thresh_vec[n_thresh]
        
        # Thresholding at the group level
        brain_map_df_thresholded = \
            complexity_analysis.thresh_by_tSNR_at_groupLevel(
            tSNR_map_df_mean_ROI_normalized,
            brain_map_df,
            thresh_level
        )

        _, N_ROI_supraThresh = brain_map_df_thresholded.shape
        N_ROI_supraThresh = N_ROI_supraThresh - 1
        
        # brain_map_df_thresholded = brain_map_df
        # _, N_ROI_supraThresh = brain_map_df_thresholded.shape
        # N_ROI_supraThresh = N_ROI_supraThresh - 1

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
        # print("Stop: Step5_preform_prediction_juseless_thr0.py")
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

        accuracy_all[n_thresh] = mean_score
        scores_all[n_thresh] = scores
        best_alpha_all[n_thresh] = best_param
        N_ROI_supraThresh_all[n_thresh] = N_ROI_supraThresh

        # -------------------------------------------------
        # Save the results along the way
        # -------------------------------------------------
        # from ptpython.repl import embed
        # print("Stop: Step5_preform_prediction_juseless_thr0.py")
        # embed(globals(), locals())
        np.savez(
            accuracy_out_filename, 
            scores_all=scores_all,
            accuracy_all=accuracy_all,
            best_alpha_all=best_alpha_all,
            N_ROI_supraThresh_all=N_ROI_supraThresh_all
        )

        print(
            (f"Accuracy for {feature_name} at the "
             f"%{int(100*thresh_level)} tSNR threshold was saved!")
            )

    print(f"Saved the results in {accuracy_out_filename}!\n")
    print(f"Accuracies: {accuracy_all.T}")

else:

    print(f"The output file {accuracy_out_filename} already exists!")
