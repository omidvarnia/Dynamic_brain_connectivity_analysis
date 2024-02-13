"""This script is used to generate .submit files for HTCondor on Juseless."""

import os, yaml, sys, shutil, argparse
import numpy as np
from pathlib import Path
import scipy.io as sio

sys.path.append((Path(__file__).parent.parent / "common").as_posix())
from htcondor_submit_file_generation import HTCondorSubmissionTemplate

# ----- Input arguments
parser = argparse.ArgumentParser(description="Preform prediction modelling of "
                                             "rsfMRI features on Juseless. ")

parser.add_argument(
    "--model_type", 
    required=True, 
    type=str, 
    help=("Type of the model: ridge_regressor,"
          "ridge_classifier, rf, heuristic_svm")
)

parser.add_argument(
    "--confound_removal", 
    required=True, 
    type=int, 
    help="0 for no remove, 1 for confound removal, 2 for features+confounds"
)

# Parse the command-line arguments
args = parser.parse_args()
confound_removal = args.confound_removal
model_type = args.model_type

assert model_type in ['ridge', 'rf', 'svm'], \
    "model_type must be one of the following: ridge, rf, svm"
assert confound_removal in [0, 1, 2], \
    "confound_removal must be one of the following: 0, 1, 2"
    
# ----- Input arguments
yaml_file = Path(
    ("/home/aomidvarnia/GIT_repositories/Comm_Biol_manuscript/analysis_codes"
     "/common/config_HCP_icbm152_mask.yaml")
)
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)

# Atlas name
atlas_name = config['analysis_name']['atlas_name']
atlas_name = 'schaefer'
assert atlas_name in ['schaefer', 'glasser'], 'Atlas name is not correct.'

# ------ Main folder
main_folder = config['juseless_paths']['main_folder']
if not os.path.isdir(main_folder):
    raise ValueError(f"Folder {main_folder} does not exist!")

# ----- Folder of the analysis codes
codes_folder = config['juseless_paths']['codes_folder']
if not os.path.isdir(codes_folder):
    raise ValueError(f"Folder {codes_folder} does not exist!")

# ----- Folder of the HPC log files
logs_folder = config['juseless_paths']['logs_folder']
if(not os.path.isdir(logs_folder)):
    raise ValueError(
        f"Folder {logs_folder} does not exist! Please run Step1 first!"
    )

# ----- Executable file
executable_file_schaefer = config['juseless_paths']['executable_file_check_tsnr_and_roi_B']
if not os.path.isfile(executable_file_schaefer):
    raise ValueError(f"File {executable_file_schaefer} does not exist!")

# ----- Subjects list
subjects_list_file = config['juseless_paths']['subjects_list_file']
if not os.path.isfile(subjects_list_file):
    raise ValueError(f"File {subjects_list_file} does not exist!")
    
full_list = sio.loadmat(subjects_list_file)
full_list = full_list['subs_fitting']

# ----- htcondor files will be stored here.
submit_files_folder = config['juseless_paths'][f'prediction_submit_files_folder_{atlas_name}']
if not os.path.isdir(submit_files_folder):
    os.makedirs(submit_files_folder)
    
# ----- Features and targets for prediction
# Prediction on healthy subjects after excluding ICD10 conditions
# Ref: 
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0154222
prediction_params = config['prediction']
feature_names = prediction_params['feature_names']
target_names = prediction_params['target_names']
target_labels = prediction_params['target_labels']
problem_types = prediction_params['problem_types']

# ----- HTCondor settings
initial_dir = config['htcondor_settings']['initial_dir']
if not os.path.isdir(initial_dir):
    raise ValueError(f"Initial directory {initial_dir} does not exist!")

mail_user = config['htcondor_settings']['mail_user']
request_cpus = config['htcondor_settings']['request_cpus']
request_memory = config['htcondor_settings']['request_memory'] # in Gigabyte
request_disk = config['htcondor_settings']['request_disk'] # in Gigabyte

assert isinstance(mail_user, str), \
    "Please provide a valid email address for mail_user!"
assert isinstance(request_cpus, int), \
    "Please provide an integer for request_cpus!"
assert isinstance(request_memory, int), \
    "Please provide an integer for request_memory!"
assert isinstance(request_disk, int), \
    "Please provide an integer for request_disk!"
assert request_cpus > 0, \
    "Please provide a positive integer for request_cpus!"
assert request_memory > 0, \
    "Please provide a positive integer for request_memory!"
assert request_disk > 0, \
    "Please provide a positive integer for request_disk!"

# ----- Full list of good subjects in UKB (courtesy of Jan Kasper)
batch_subjects = config['prediction']['subjects_span']['batch_subjects']
N_batches = len(batch_subjects) - 1 #  Number of batch jobs

# ----- Generate slurm files for submission to slurm
n_features = range(len(feature_names))
for n_feature in range(1, len(feature_names)):

    feature_name = feature_names[n_feature]

    # ----- Subjects list for each batch
    batch_start = batch_subjects[0]
    batch_end = batch_subjects[-1]

    subjects_list = full_list[0, batch_start:batch_end]
    subjects_list = np.concatenate(subjects_list)
    subjects_list = subjects_list.tolist()
    subjects_list = \
        list({x.replace('sub-', '') for x in subjects_list})
    N_batch_size = len(subjects_list)

    # --------------------------------------------------------------------
    # Generate the slurm file with multiple sruns
    # --------------------------------------------------------------------
    job_name = f"check_tsnr_and_roi_model_{model_type}_feature{n_feature}_Batch20k"

    if(confound_removal == 0):
        submit_file = Path(
            f"{submit_files_folder}/NoConfRem_{job_name}.submit"
        )
    elif(confound_removal == 1):
        submit_file = Path(
            f"{submit_files_folder}/ConfRem_{job_name}.submit"
        )
    elif(confound_removal == 2):
        submit_file = Path(
            f"{submit_files_folder}/CombinedFeatures_{job_name}.submit"
        )
    else:
        raise ValueError(
            "Please provide a valid value for confound_removal (0, 1, or 2)!"
        )

    # ----- Input parameters for Spatiotemporal_RangeEn.py
    general_args = [
        model_type,
        feature_name,
        N_batch_size,
        confound_removal
    ]

    # ----- Create an instance from the 'HTCondorSubmissionTemplate' class
    template = HTCondorSubmissionTemplate(
        initial_dir = initial_dir,
        executable_file_schaefer = executable_file_schaefer,
        target_names=target_names,
        target_labels=target_labels,
        mail_user = mail_user,
        logs_folder = logs_folder,
        submit_file = submit_file,
        args = general_args,
        subjects_list = subjects_list,
        request_cpus = request_cpus,
        request_memory = request_memory,
        request_disk = request_disk
    )

    # ----- Write the subject-specific .slurm file 
    template.write_queue_for_prediction()
     
