"""
This script is used to generate .submit files for HTCondor on Juseless.
It parses command-line arguments to specify the type of model for prediction
modeling of rsfMRI features.

The script reads configuration settings from a YAML file to determine
various paths, settings, and parameters required for the analysis. It then
generates HTCondor submit files based on these settings, which are used to
submit jobs for prediction modeling on a computing cluster.

Usage example:
    python Step9_create_job_files_for_HTCondor.py --model_type <model>

Arguments:
    --model_type: Type of the prediction model. Valid options include
        'ridge_regressor', 'ridge_classifier', 'rf', and 'heuristic_svm'.

Example:
    python Step9_create_job_files_for_HTCondor.py --model_type ridge_regressor

Note:
    This script assumes the availability of a configuration file specified in
    the YAML format.
     
Written by: Amir Omidvarnia, PhD
Email: amir.omidvarnia@gmail.com
"""

import os, yaml, sys, argparse
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

# Parse the command-line arguments
args = parser.parse_args()
model_type = args.model_type

assert model_type in ['ridge', 'rf', 'svm'], \
    "model_type must be one of the following: ridge, rf, svm"
    
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
executable_file = \
    config['juseless_paths']['executable_file_prediction_ConfOnly']
if not os.path.isfile(executable_file):
    raise ValueError(f"File {executable_file} does not exist!")

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
target_names = prediction_params['target_names']
target_labels = prediction_params['target_labels']
problem_types = prediction_params['problem_types']

# target_names = target_names[:-1]
# target_labels = target_labels[:-1]
# problem_types = problem_types[:-1]

target_names = ['31-0.0']
target_labels = ['Sex']
problem_types = ['binary_classification']

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
# from ptpython.repl import embed
# print("Stop: Step6_create_job_files_for_HTCondor.py")
# embed(globals(), locals())
for n_batch in range(N_batches):

    # ----- Subjects list for each batch
    batch_start = batch_subjects[0]
    batch_end = batch_subjects[n_batch+1]

    subjects_list = full_list[0, batch_start:batch_end]
    subjects_list = np.concatenate(subjects_list)
    subjects_list = subjects_list.tolist()
    subjects_list = \
        list({x.replace('sub-', '') for x in subjects_list})
    N_batch_size = len(subjects_list)

    # --------------------------------------------------------------------
    # Generate the slurm file with multiple sruns
    # --------------------------------------------------------------------
    job_name = f"model_{model_type}_Batch{n_batch+1}"

    submit_file = Path(
        f"{submit_files_folder}/ConfOnly_{job_name}.submit"
    )

    # ----- Input parameters for Spatiotemporal_RangeEn.py
    general_args = [
        model_type,
        N_batch_size
    ]

    # ----- Create an instance from the 'HTCondorSubmissionTemplate' class
    template = HTCondorSubmissionTemplate(
        initial_dir = initial_dir,
        executable_file = executable_file,
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
    template.write_queue_for_prediction_ConfOnly()
     
