"""
This script generates HTCondor submission files for performing fingerprinting
analysis of rsfMRI complexity measures.

It takes command-line arguments for the type of model and confound removal
strategy, loads configurations from a YAML file, and sets up folder paths.
The script generates HTCondor submission files for each batch of subjects
specified in the configuration. It creates multiple submission files with
different configurations based on the model type and confound removal strategy.

Arguments:
    model_type (str): Type of the model, one of "ridge_regressor,"
        "ridge_classifier," "rf," or "heuristic_svm."
    confound_removal (int): Confound removal strategy: 0 for no removal, 1 for
        confound removal, 2 for features + confounds.

Note:
    This script assumes the availability of a configuration file specified in
    the YAML format and the HTCondor submission template class.

Usage example:
    python script_name.py --model_type ridge_regressor --confound_removal 1
    
Written by: Amir Omidvarnia, PhD
Email: amir.omidvarnia@gmail.com
"""
import os, yaml, sys, argparse
import numpy as np
import scipy.io as sio
from pathlib import Path

sys.path.append((Path(__file__).parent.parent / "common").as_posix())
from htcondor_submit_file_generation import HTCondorSubmissionTemplate

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
model_type = args.model_type
confound_removal = args.confound_removal
atlas_name = "schaefer"     # Atlas name in nilearn

assert model_type in ['ridge', 'rf', 'svm'], \
    "model_type must be one of the following: ridge, rf, svm"

assert atlas_name in ['schaefer', 'glasser'], 'Atlas name is not correct.'

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

# ----- Results folder
fingerprinting_results_folder = \
    config['juseless_paths'][f'fingerprinting_results_folder_{atlas_name}']

# ----- Executable file
executable_file = \
    config['juseless_paths']['executable_file_fingerprinting']
if not os.path.isfile(executable_file):
    raise ValueError(f"File {executable_file} does not exist!")

# ----- Subjects list
subjects_list_file = config['juseless_paths']['subjects_list_file']
if not os.path.isfile(subjects_list_file):
    raise ValueError(f"File {subjects_list_file} does not exist!")
    
full_list = sio.loadmat(subjects_list_file)
full_list = full_list['subs_fitting']

# ----- htcondor files will be stored here.
submit_files_folder = config['juseless_paths'][f'fingerprinting_submit_files_folder_{atlas_name}']
if not os.path.isdir(submit_files_folder):
    os.makedirs(submit_files_folder)
    
if not os.path.isdir(submit_files_folder):
    os.mkdir(submit_files_folder)

# ----- Array of slurm files. Each file includes multiple sruns.
batch_subjects = config['prediction']['subjects_span']['batch_subjects']
N_batches = len(batch_subjects) - 1 #  Number of batch jobs 

# ----- Generate slurm files for submission to slurm
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
    job_name = f"model_{model_type}_{atlas_name}_Batch{n_batch+1}"

    if(confound_removal == 0):
        submit_file = os.path.join(
            submit_files_folder,
            f"{job_name}_noConfRem.submit"
        )
    elif(confound_removal == 1):
        submit_file = os.path.join(
            submit_files_folder,
            f"{job_name}_ConfRem.submit"
        )
    elif(confound_removal == 2):
        submit_file = os.path.join(
            submit_files_folder,
            f"{job_name}_CombinedFeature.submit"
        )
    else:
        print(f'confound_removal must be either 0 (no confound removal), 1 (confound removal) or 2 (combined feature) for Step 5.')
        exit()

    # ----- Input parameters for the executable file
    general_args = [
        model_type,
        N_batch_size,
        confound_removal
    ]

    # ----- Create an instance from the 'HTCondorSubmissionTemplate' class
    template = HTCondorSubmissionTemplate(
        initial_dir = initial_dir,
        executable_file = executable_file,
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
    template.write_queue_for_fingerptinting()
      
