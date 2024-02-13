"""Obtain the complexity feature discributions for males and females. 

The script reads a configuration YAML file to set up various parameters 
for the analysis across males and females. The analysis includes the following
steps:

1. Reading configuration parameters from a config YAML file.
2. Setting up necessary paths.
3. Loading subject-specific data from the .npz files.
4. Performing complexity analysis on the data, including normalization,
   thresholding, and modeling.
5. Comparing and visualizing results between males/females and complexity
   measures.

This script requires external packages such as NumPy, Pandas, PyYAML, and
custom modules from 'merged_toolkit'. Make sure to set up the Python
environment with the required dependencies.

The script can be run from the command line as follows:
>> python3 Step4_obtain_distributions.py

Note:
    This script assumes the availability of a configuration file specified in
    the YAML format.
    
Written by: Amir Omidvarnia, PhD
Email: amir.omidvarnia@gmail.com
"""
import numpy as np
import pandas as pd
import os, yaml
from pathlib import Path
from merged_toolkit.fc_analysis.complexity import complexity_analysis
import warnings
warnings.simplefilter("ignore")

# ----- Input arguments
yaml_file = Path(
    ("/home/aomidvarnia/GIT_repositories/Comm_Biol_manuscript/analysis_codes"
     "/common/config_HCP_icbm152_mask.yaml")
)
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)

# Atlas name
atlas_name = config['analysis_name']['atlas_name']
assert atlas_name in ['schaefer', 'glasser'], 'Atlas name is not correct.'

# Make a balanced population with equal number of males and females
N_subj_males_or_females = config['N_subj_males_or_females']
assert isinstance(N_subj_males_or_females, int), \
    "Please provide an integer for N_subj_males_or_females!"

# ----- Parcellation and graph theory analysis parameters
# Number of networks in the parcellation atlas
N_network = config['feature_extraction'][f'brain_atlas_{atlas_name}']['N_network']
# Number of desired ROIs in the atlas
N_ROI = config['feature_extraction'][f'brain_atlas_{atlas_name}']['N_ROI']

assert N_network in [None, 7, 17], \
    "Please provide a valid number of networks!"
assert N_ROI in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 360], \
    "Please provide a valid number of ROIs!"

# ----- Main folder on Juseless
main_folder = config['juseless_paths']['main_folder']
if not os.path.isdir(main_folder):
    raise ValueError(f"Folder {main_folder} does not exist!")

# UKB database source for datalad
datalad_database_source = \
    config['juseless_software_settings']['datalad_database_source']

# Datalad dataset of anatomical measures using CAT12.7 (e.g., TIV) 
# ---> The dataset is located on Juseless. So, the csv files must be 
# copied to jureca beforehand.
CAT_database_source = \
    config['juseless_software_settings']['CAT_database_source']

# ----- Folder of the subject-specific .npz file
npzfiles_folder = config['juseless_paths'][f'npzfiles_folder_{atlas_name}']
if not os.path.isdir(npzfiles_folder):
    raise ValueError(
        (f"Folder {npzfiles_folder} does not exist! Please provide the "
          "path to the folder containing the subject-specific .npz files.")
    )

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

# ----- Input codes for ukbb_parser
ukbb_parser_out_folder = os.path.join(main_folder, "ukbb_parser_files")
if not os.path.isdir(ukbb_parser_out_folder):
    raise ValueError(f"Folder {ukbb_parser_out_folder} does not exist!")

ukbb_parser_out_prefix = "ukbb_parser_complexity"
ukbb_parser_out_file = os.path.join(
            ukbb_parser_out_folder, 
            f'{ukbb_parser_out_prefix}_img_subs_only.csv'
)

# ----- Analysis starts here
# from ptpython.repl import embed
# print('OKKKK: Step4')
# embed(globals(), locals())

# ---------------------------------------#
# Load the behavioral data of UKB subjects 
# with fMRI datasets
# ---------------------------------------#
ukbb_parser_DF_codes = prediction_params['ukbb_parser_DF_codes']
UKB_tsv_filename = prediction_params['UKB_tsv_filename']
img_subs_only = pd.read_csv(ukbb_parser_out_file, sep=',')

# -------------------------------------------------
# Load the TIV (Total Intracranial Volume) data from 
# the database
# -------------------------------------------------
TIV_column = np.load(TIV_filename, allow_pickle=True)
TIV_column = pd.DataFrame(TIV_column)
TIV_column.columns = ['eid', 'TIV']
TIV_column = TIV_column.astype({'eid':int})

# ------------------------------------------
# Load the subject-specific .npz files
# and concatenate subject-specific maps.
# Make a dictionary of dataframes containing  
# the brain maps of all subjects for all complexity
# measures. It takes some time .....
# ------------------------------------------
dict_of_brain_maps_df = np.load(brain_maps_all_file, allow_pickle=True)
dict_of_brain_maps_df = dict_of_brain_maps_df.item(0)

# --------------------------------------------------
# Load the dictionary of brain map dataframes and
# perform prediction modeling on them
# --------------------------------------------------
# from ptpython.repl import embed
# print('OKKKK: Step4')
# embed(globals(), locals())
for n_feature in range(N_features):

    # --------------------------------------------------
    # Normalize the tSNR values of all UKB subjects 
    # between 0 and 1 and compute the mean tSNR map
    # --------------------------------------------------
    feature_name = feature_names[n_feature]

    brain_map_df = dict_of_brain_maps_df[feature_name]

    if(feature_name=='tSNR_fMRI'):

        brain_map_df, tSNR_map_df_mean_ROI_normalized = \
            complexity_analysis.tSNR_MinMaxScaler(brain_map_df)

    # -------------------------------------------------
    # Thresholding at the group level
    # -------------------------------------------------
    brain_map_df_thresholded = \
        complexity_analysis.thresh_by_tSNR_at_groupLevel(
        tSNR_map_df_mean_ROI_normalized,
        brain_map_df
    )

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
    # Extract the age and target complexity measure of
    # males and females along with their histograms for
    # plotting (N = 20000)
    # -------------------------------------------------
    complexity_analysis.compare_male_female(
            main_folder,
            df_julearn,
            target_names,
            target_labels,
            feature_name,
            N_ROI
        )

    # -------------------------------------------------
    print(f"The complexity measure {feature_name} is done!")
