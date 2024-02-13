"""
This script performs fingerprinting analysis of rsfMRI measures.

It takes command-line arguments for the type of predictive model, population
size, and confound removal strategy. Then it loads configurations from a YAML
file, sets up folder paths, and parses input parameters.

The script loads subject-specific complexity measures from .npz files and
constructs a Julearn-compatible dataframe. It then performs pairwise
fingerprinting analysis between different complexity measures to compute
identification accuracy.

The identification accuracy matrix is saved as a NumPy .npz file along with
additional information such as confound removal strategy, complexity measure
names, and prediction parameters.

Arguments:
    model_type (str): Type of the predictive model, e.g., "ridge", "rf", "svm".
    N_subj_all (int): Population size of the analysis.
    confound_removal (int): Confound removal strategy: 0 for no removal,
    1 for confound removal, 2 for features + confounds.

Usage example:
    python Step12_fingerprinting_analysis.py --model_type ridge \
        --N_subj_all 1000 --confound_removal 1

Note:
    This script assumes the availability of subject-specific .npz files, a
    configuration file specified in the YAML format, and the identification
    module.
       
Written by: Amir Omidvarnia, PhD
Email: amir.omidvarnia@gmail.com
"""
import pandas as pd
import numpy as np
import os, yaml, argparse
from pathlib import Path
from merged_toolkit.fc_analysis.complexity import complexity_analysis
from identification import identify
from skbold.preproc import ConfoundRegressor


import warnings
warnings.simplefilter("ignore")

# ----- Input arguments
yaml_file = Path(
    ("/home/aomidvarnia/GIT_repositories/Comm_Biol_manuscript/analysis_codes"
     "/common/config_HCP_icbm152_mask.yaml")
)
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)
    
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

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
model_type = args.model_type
confound_removal = args.confound_removal
N_subj_all = args.N_subj_all

# Make a balanced population with equal number of males and females
N_subj_males_or_females = int(np.fix(N_subj_all/2))

# ----- Parcellation and graph params
atlas_name = "schaefer"     # Atlas name in nilearn
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
fingerprinting_results_folder = \
    config['juseless_paths'][f'fingerprinting_results_folder_{atlas_name}']
    
if(not os.path.isdir(fingerprinting_results_folder)):
    os.mkdir(fingerprinting_results_folder)

# Make a balanced population with equal number of males and females
N_subj_males_or_females = int(np.fix(N_subj_all/2))

prediction_params = config['prediction']
feature_names = prediction_params['feature_names']
n_features = len(feature_names)
target_name = np.nan
confounds = ['AgeAtScan', '31-0.0', 'TIV']

# ------------------------------------------
# Load the subject-specific .npz files
# and concatenate subject-specific maps.
# Make a dictionary of dataframes containing  
# the brain maps of all subjects for all complexity
# measures. It takes some time .....
# ------------------------------------------
brain_maps_all_file = os.path.join(
            npzfiles_folder, 'complex_measures_all_df.npy')
 
dict_of_brain_maps_df = np.load(
    brain_maps_all_file, 
    allow_pickle=True)
dict_of_brain_maps_df = dict_of_brain_maps_df.item(0)

# ----- Input codes for ukbb_parser
ukbb_parser_out_folder = os.path.join(main_folder, "ukbb_parser_files")
if not os.path.isdir(ukbb_parser_out_folder):
    raise ValueError(f"Folder {ukbb_parser_out_folder} does not exist!")

ukbb_parser_out_prefix = "ukbb_parser_complexity"
ukbb_parser_out_file = os.path.join(
            ukbb_parser_out_folder, 
            f'{ukbb_parser_out_prefix}_img_subs_only.csv'
)

img_subs_only = pd.read_csv(ukbb_parser_out_file, sep=',')

# -------------------------------------------------
# Load the TIV (Total Intracranial Volume) data from 
# the database
# -------------------------------------------------
TIV_filename = config['juseless_paths']['TIV_filename']

if not os.path.isfile(TIV_filename):
    raise ValueError(f"File {TIV_filename} does not exist!")

TIV_column = np.load(TIV_filename, allow_pickle=True)
TIV_column = pd.DataFrame(TIV_column)
TIV_column.columns = ['eid', 'TIV']
TIV_column = TIV_column.astype({'eid':int})

# -------------------------------------------------
# Perform the fingerprinting analysis
# -------------------------------------------------
ss = 0
Ideff_accu = np.zeros((n_features, n_features))
for n_feature1 in range(n_features):

    feature_name1 = feature_names[n_feature1]

    # -------------------------------------------------
    # Extract the FC matrices from dict_of_brain_maps_df 
    # -------------------------------------------------
    brain_map_df1 = dict_of_brain_maps_df[feature_name1]
    brain_map_df1['eid'] = brain_map_df1['eid'].astype(int)

    # -------------------------------------------------
    # Threshold the mean complexity maps of all
    # subjects using the tSNR map 
    # -------------------------------------------------
    brain_map_df1_thresholded = brain_map_df1
    
    # -------------------------------------------------
    # Merging the maps with ukbb_parser and making a 
    # Julearn-compatible df. It also addes TIV as a
    # confound in the dataframe
    # -------------------------------------------------
    df_julearn1 = complexity_analysis.merge_ukbb_complexity(
        img_subs_only,
        brain_map_df1_thresholded,
        TIV_column
        )

    # -------------------------------------------------
    # Make a balanced population with equal number of 
    # male and females
    # -------------------------------------------------
    target_name = np.nan
    df_julearn1, mean_age_males, mean_age_females = \
        complexity_analysis.make_balanced_classes(
            df_julearn1,
            target_name,
            N_subj_males_or_females,
            N_ROI
        )

    for n_feature2 in range(n_feature1+1, n_features):

        ss = ss + 1
        
        feature_name2 = feature_names[n_feature2]
        brain_map_df2 = dict_of_brain_maps_df[feature_name2]

        # -------------------------------------------------
        # Threshold the mean complexity maps of all
        # subjects using the tSNR map 
        # -------------------------------------------------
        brain_map_df2_thresholded = brain_map_df2
        
        # -------------------------------------------------
        # Merging the maps with ukbb_parser and making a 
        # Julearn-compatible df. It also addes TIV as a
        # confound in the dataframe
        # -------------------------------------------------
        # print('OKKKK: Spatiotemporal complexity')
        # embed(globals(), locals())
        df_julearn2 = complexity_analysis.merge_ukbb_complexity(
            img_subs_only,
            brain_map_df2_thresholded,
            TIV_column
            )

        # -------------------------------------------------
        # Make a balanced population with equal number of 
        # male and females
        # -------------------------------------------------
        df_julearn2, mean_age_males, mean_age_females = \
            complexity_analysis.make_balanced_classes(
                df_julearn2,
                target_name,
                N_subj_males_or_females,
                N_ROI
            )

        # ---------------------------------
        # Pair-wise Fingerprinting analysis
        # Ref: https://github.com/juaml/identification
        # ---------------------------------
        try:
            df1 = df_julearn1.iloc[:, 1:(N_ROI+1)]
            df2 = df_julearn2.iloc[:, 1:(N_ROI+1)]

            df1 = df1.astype(float)
            df2 = df2.astype(float)

            df1_confounds = df_julearn1[confounds]
            df2_confounds = df_julearn2[confounds]

            df1_confounds = df1_confounds.astype(float)
            df2_confounds = df2_confounds.astype(float)

            # ---------------------------------
            # Regress out confounds from the complexity
            # measures (features) before performing fingerprinting
            # https://skbold.readthedocs.io/en/latest/source/skbold.preproc.confounds.html
            # ---------------------------------
            if(confound_removal==1):

                ## df1
                X1 = np.array(df1)
                conf1 = np.array(df1_confounds)

                cfr = ConfoundRegressor(confound=conf1, X=X1)
                df1_ConfoundRemoved = cfr.fit_transform(X1)

                df1.iloc[:,:] = df1_ConfoundRemoved
                del X1, conf1, cfr, df1_ConfoundRemoved

                ## df2
                X2 = np.array(df2)
                conf2 = np.array(df2_confounds)

                cfr = ConfoundRegressor(confound=conf2, X=X2)
                df2_ConfoundRemoved = cfr.fit_transform(X2)

                df2.iloc[:,:] = df2_ConfoundRemoved
                del X2, conf2, cfr, df2_ConfoundRemoved

            # ---------------------------------
            # Compute Identification accuracy
            # ---------------------------------
            acc_df1_df2 = identify(df2, df1, metric="spearman")
            acc_df2_df1 = identify(df1, df2, metric="spearman")

            Ideff_accu[n_feature1, n_feature2] = \
                    (acc_df1_df2 + acc_df2_df1) / 2

            print(f'No {ss}: {feature_name1}({n_feature1}) - {feature_name2}({n_feature2}) --> Success!')

        except:

            print(f'No {ss}: {feature_name1}(f{n_feature1}) - {feature_name2}({n_feature2}) --> Problematic!')

# ---------------------------
# Save the output and figures
# ---------------------------
if(confound_removal==0):
    output_file = os.path.join(fingerprinting_results_folder,
        f'Ideff_accu_ishealthy_noConfRemoved_N{N_subj_all}.npz')

elif(confound_removal==1):
    output_file = os.path.join(fingerprinting_results_folder,
        f'Ideff_accu_ishealthy_ConfRemoved_N{N_subj_all}.npz')

else:
    output_file = os.path.join(fingerprinting_results_folder,
        f'Ideff_accu_ishealthy_CombinedFeature_N{N_subj_all}.npz')

np.savez(output_file, Ideff_accu=Ideff_accu,
                        confound_removal=confound_removal,
                        complex_measure_names=feature_names,
                        prediction_params=prediction_params)
                        