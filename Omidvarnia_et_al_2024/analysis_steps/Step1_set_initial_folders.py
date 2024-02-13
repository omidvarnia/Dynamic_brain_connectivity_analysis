"""This script creates the initial folders for the analysis.

The script creates the following folders:
    1. The analysis log folder
    2. The HTCondor log folder
    3. The HTCondor submit files folder
    4. The group level prediction folder
    5. The single subject npz files folder

To get an interactive session on the cluster, run the following command:
    >> condor_submit --interactive

To create a new conda environment, run the following command:
    >> conda create -n <env_name> python=3.9
    >> conda activate <env_name>

The script also deletes the log folders and submit files if they already
exist.

Note:
    This script assumes the availability of a configuration file specified in
    the YAML format.
    
Written by: Amir Omidvarnia, PhD
Email: amir.omidvarnia@gmail.com
"""
import os, sys
from pathlib import Path
import yaml
import warnings
warnings.simplefilter("ignore")

sys.path.append((Path(__file__).parent.parent / "common").as_posix())
from complexity import complexity_analysis

# from ptpython.repl import embed
# print("Starting the script...")
# embed(globals(), locals())

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

# Set the analysis name and the number of networks and ROIs
analysis_name1 = config['analysis_name']['feature_extraction']
analysis_name2 = config['analysis_name']['prediction']

# Number of networks in the parcellation atlas
N_network = config['feature_extraction'][f'brain_atlas_{atlas_name}']['N_network']
# Number of desired ROIs in the atlas
N_ROI = config['feature_extraction'][f'brain_atlas_{atlas_name}']['N_ROI']

assert isinstance(analysis_name1, str), \
    "Please provide a string for analysis_name!"
assert isinstance(analysis_name2, str), \
    "Please provide a string for analysis_name!"
assert N_network in [None, 7, 17], \
    "Please provide a valid number of networks!"
assert N_ROI in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 360], \
    "Please provide a valid number of ROIs!"

# Set the main folder of the analysis
main_folder = config['juseless_paths']['main_folder']
if not os.path.isdir(main_folder):
    raise ValueError(f"Folder {main_folder} does not exist!")

# Delete and recreate the HTCondor log folders if they already exist
logs_folder = config['juseless_paths']['logs_folder']
if not os.path.isdir(f"{logs_folder}"):
    os.system(f"mkdir {logs_folder}")
    print(f"Folder {logs_folder} does not exist. Creating it.")
# else:
#     os.system(f"rm -rf {logs_folder}")
#     os.system(f"mkdir {logs_folder}")
#     print(f"Folder {logs_folder} already exists. Deleting and recreating it.")

# ---------------------------------------------------------------------
# Parcellated time series are located here.
# ---------------------------------------------------------------------
parcellated_data_folder_schaefer = config['juseless_paths']['parcellated_data_folder_schaefer']
if not os.path.isdir(parcellated_data_folder_schaefer):
    raise ValueError(f"Folder {parcellated_data_folder_schaefer} does not exist!")

parcellated_data_folder_glasser = config['juseless_paths']['parcellated_data_folder_glasser']
if not os.path.isdir(parcellated_data_folder_glasser):
    raise ValueError(f"Folder {parcellated_data_folder_glasser} does not exist!")

# ---------------------------------------------------------------------
# FC/TC features are located here.
# ---------------------------------------------------------------------
npzfiles_folder_schaefer = config['juseless_paths']['npzfiles_folder_schaefer']
if not os.path.isdir(npzfiles_folder_schaefer):
    os.mkdir(npzfiles_folder_schaefer)
    print(f"Folder {npzfiles_folder_schaefer} does not exist. Creating it.")

npzfiles_folder_glasser = config['juseless_paths']['npzfiles_folder_glasser']
if not os.path.isdir(npzfiles_folder_glasser):
    os.mkdir(npzfiles_folder_glasser)
    print(f"Folder {npzfiles_folder_glasser} does not exist. Creating it.")

# ---------------------------------------------------------------------
# HTCondor submit files will be stored here.
# ---------------------------------------------------------------------
submit_files_folder_schaefer = config['juseless_paths']['submit_files_folder_schaefer']
if os.path.isdir(submit_files_folder_schaefer):
    os.system(f"rm -rf {submit_files_folder_schaefer}")
    os.system(f"mkdir {submit_files_folder_schaefer}")
    print(f"Folder {submit_files_folder_schaefer} already exists. Deleting and recreating it.")
else:
    os.system(f"mkdir {submit_files_folder_schaefer}")
    print(f"Folder {submit_files_folder_schaefer} does not exist. Creating it.")

prediction_submit_files_folder_schaefer = config['juseless_paths']['prediction_submit_files_folder_schaefer']
if os.path.isdir(prediction_submit_files_folder_schaefer):
    os.system(f"rm -rf {prediction_submit_files_folder_schaefer}")
    os.system(f"mkdir {prediction_submit_files_folder_schaefer}")
    print(f"Folder {prediction_submit_files_folder_schaefer} already exists. Deleting and recreating it.")
else:
    os.system(f"mkdir {prediction_submit_files_folder_schaefer}")
    print(f"Folder {prediction_submit_files_folder_schaefer} does not exist. Creating it.")
    
submit_files_folder_glasser = config['juseless_paths']['submit_files_folder_glasser']
if os.path.isdir(submit_files_folder_glasser):
    os.system(f"rm -rf {submit_files_folder_glasser}")
    os.system(f"mkdir {submit_files_folder_glasser}")
    print(f"Folder {submit_files_folder_glasser} already exists. Deleting and recreating it.")
else:
    os.system(f"mkdir {submit_files_folder_glasser}")
    print(f"Folder {submit_files_folder_glasser} does not exist. Creating it.")

prediction_submit_files_folder_glasser = config['juseless_paths']['prediction_submit_files_folder_glasser']
if os.path.isdir(prediction_submit_files_folder_glasser):
    os.system(f"rm -rf {prediction_submit_files_folder_glasser}")
    os.system(f"mkdir {prediction_submit_files_folder_glasser}")
    print(f"Folder {prediction_submit_files_folder_glasser} already exists. Deleting and recreating it.")
else:
    os.system(f"mkdir {prediction_submit_files_folder_glasser}")
    print(f"Folder {prediction_submit_files_folder_glasser} does not exist. Creating it.")
    
# ---------------------------------------------------------------------
# Prediction results for the schaefer atlas will be stored here.
# ---------------------------------------------------------------------
prediction_results_folder_schaefer = config['juseless_paths']['prediction_results_folder_schaefer']
if not os.path.isdir(prediction_results_folder_schaefer):
    os.system(f"mkdir {prediction_results_folder_schaefer}")
    print(f"Folder {prediction_results_folder_schaefer} does not exist. Creating it.")
      
prediction_results_folder_svm_schaefer = os.path.join(prediction_results_folder_schaefer, 'prediction_results_svm')
if not os.path.isdir(prediction_results_folder_svm_schaefer):
    os.mkdir(prediction_results_folder_svm_schaefer)
    print(f"Folder {prediction_results_folder_svm_schaefer} does not exist. Creating it.")
    
prediction_results_folder_tsnr_svm_schaefer = os.path.join(prediction_results_folder_schaefer, 'tsnr_analysis_svm')
if not os.path.isdir(prediction_results_folder_tsnr_svm_schaefer):
    os.mkdir(prediction_results_folder_tsnr_svm_schaefer)
    print(f"Folder {prediction_results_folder_tsnr_svm_schaefer} does not exist. Creating it.")

# ---------------------------------------------------------------------
# Prediction results for the glasser atlas will be stored here.
# ---------------------------------------------------------------------
prediction_results_folder_glasser = config['juseless_paths']['prediction_results_folder_glasser']
if not os.path.isdir(prediction_results_folder_glasser):
    os.system(f"mkdir {prediction_results_folder_glasser}")
    print(f"Folder {prediction_results_folder_glasser} does not exist. Creating it.")
      
prediction_results_folder_svm_glasser = os.path.join(prediction_results_folder_glasser, 'prediction_results_svm')
if not os.path.isdir(prediction_results_folder_svm_glasser):
    os.mkdir(prediction_results_folder_svm_glasser)
    print(f"Folder {prediction_results_folder_svm_glasser} does not exist. Creating it.")
    
prediction_results_folder_tsnr_svm_glasser = os.path.join(prediction_results_folder_glasser, 'tsnr_analysis_svm')
if not os.path.isdir(prediction_results_folder_tsnr_svm_glasser):
    os.mkdir(prediction_results_folder_tsnr_svm_glasser)
    print(f"Folder {prediction_results_folder_tsnr_svm_glasser} does not exist. Creating it.")

# ---------------------------------------------------------------------
# Create the feature file for all subjects
# ---------------------------------------------------------------------
from ptpython.repl import embed
print("Starting the script...")
embed(globals(), locals())

complex_measure_names = config['prediction']['feature_names']
npzfiles_folder = config['juseless_paths'][f'npzfiles_folder_{atlas_name}']
brain_maps_all_file = os.path.join(
            npzfiles_folder, 'complex_measures_all_df.npy')
if not os.path.isfile(brain_maps_all_file):
    print(f"File {brain_maps_all_file} does not exist. Creating it.")
    
    dict_of_brain_maps_df = complexity_analysis.brainmap2df(
        complex_measure_names,
        npzfiles_folder,
        brain_maps_all_file,
        N_ROI
    )
else:
    print(f"File {brain_maps_all_file} already exists!")