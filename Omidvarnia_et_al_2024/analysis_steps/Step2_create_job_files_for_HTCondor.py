"""This script is used to generate .submit files for HTCondor on Juseless.

The script is used to generate .submit files for the complexity analysis
of the UKB dataset. It is based on the HTCondorSubmissionTemplate class in the 
'common' folder.

These input arguments need to be defined at the beginning of the script:
1. analysis_name: 
    name of the analysis
2. subjects_list: 
    list of UKB subject IDs to be included in the htcondor files
3. initial_dir: 
    initial directory for the HTCondor job
4. executable_file_schaefer: 
    name of file to be executed by condor (including file extension)
5. mail_user: 
    An email addreess for receiving different notices from juseless
6. logs_folder: 
    A folder name where the htcondor log files will be saved.
7. request_cpus: 
    Number of requested CPUs (def = 1)
8. request_memory: 
    Number of cpus per task (def = 32)
9. request_disk: 
    Number of requested disk space (def = 100)
10. submit_file: 
    name of submit file (including directory) to generate for Juseless,
    e.g., "submit_file.submit"
11. Arguments: 
    list of additional arguments to be passed to the executable file 

The script can be run from the command line as follows:
>> python3 Step2_create_job_files_for_HTCondor.py

Note:
    This script assumes the availability of a configuration file specified in
    the YAML format.
    
Written by: Amir Omidvarnia, PhD
Email: amir.omidvarnia@gmail.com
"""
import os, sys
from pathlib import Path
import yaml
import numpy as np
import scipy.io as sio
sys.path.append((Path(__file__).parent.parent / "common").as_posix())
from htcondor_submit_file_generation import HTCondorSubmissionTemplate

# from ptpython.repl import embed
# print("Stop: Step2_create_job_files_for_HTCondor.py")
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

# # ----- FSL params
# fsl_applywarp_interp = \
#     config['jureca_software_settings']['fsl_params']['fsl_applywarp_interp']
# assert fsl_applywarp_interp in ['spline', 'trilinear', 'nn'], \
#     "Please provide a string for fsl_applywarp_interp!"

# ----- Parcellation and graph theory analysis parameters
# Spatial resolution of the parcellation atlas
resolution_mm = config['feature_extraction'][f'brain_atlas_{atlas_name}']['resolution_mm']
# Number of networks in the parcellation atlas
N_network = config['feature_extraction'][f'brain_atlas_{atlas_name}']['N_network']
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

# ----- Define initial parameters for Multiscale entropy analysis
N_r = config['feature_extraction']['entropy_params']['N_r']
emb_dim = config['feature_extraction']['entropy_params']['emb_dim']
r_MSE = config['feature_extraction']['entropy_params']['r_MSE']
N_scale = config['feature_extraction']['entropy_params']['N_scale']

assert isinstance(N_r, int), \
    "Please provide an integer for N_r!"
assert isinstance(emb_dim, int), \
    "Please provide an integer for emb_dim!"
assert isinstance(r_MSE, float), \
    "Please provide a float between 0 and 1 for r_MSE!"

# ----- Folder of the main analysis
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
executable_file_feature_extraction = config['juseless_paths']['executable_file_feature_extraction']
if not os.path.isfile(executable_file_feature_extraction):
    raise ValueError(f"File {executable_file_feature_extraction} does not exist!")

# ----- Subjects list
subjects_list_file = config['juseless_paths']['subjects_list_file']
if not os.path.isfile(subjects_list_file):
    raise ValueError(f"File {subjects_list_file} does not exist!")
    
full_list = sio.loadmat(subjects_list_file)
full_list = full_list['subs_fitting']

# ----- htcondor files will be stored here.
submit_files_folder = config['juseless_paths'][f'submit_files_folder_{atlas_name}']
if(not os.path.isdir(submit_files_folder)):
    raise ValueError(
        f"Folder {submit_files_folder} does not exist! Please run Step1 first!"
    )

# ----- Full list of good subjects in UKB (courtesy of Jan Kasper)
subj_start = config['feature_extraction']['subjects_span']['subj_start']
# Max number of subjects: 30044
subj_end = config['feature_extraction']['subjects_span']['subj_end']
# Number of subjects in each batch job (.submit submission)
batch_size = config['feature_extraction']['subjects_span']['batch_size']

assert subj_end <= full_list.shape[1], \
    f"subj_end ({subj_end}) is larger than the number of subjects in the list ({full_list.shape[1]})"
assert subj_start < subj_end, \
    f"subj_start ({subj_start}) is larger than subj_end ({subj_end})"
assert batch_size > 0, \
    f"batch_size ({batch_size}) must be larger than 0"

# Array of batch jobs. Each batch will analyse 'batch_size' subjects
batch_subjects = np.arange(
    subj_start, 
    subj_end+batch_size, 
    batch_size, 
    dtype=int
)
N_batches = len(batch_subjects) #  Number of batch jobs

# ----- Generate batch job files for submission to htcondor
for n_batch in range(N_batches-1):

    # ----- Subjects list for each batch
    batch_start = batch_subjects[n_batch]
    batch_end = batch_subjects[n_batch+1]
    
    print(f"Batch {n_batch+1} of {N_batches} (subjects {batch_start} to {batch_end})")

    subjects_list = full_list[0, batch_start:batch_end]
    subjects_list = np.concatenate(subjects_list)
    subjects_list = subjects_list.tolist()
    subjects_list = \
        list({x.replace('sub-', '') for x in subjects_list})
    N_batch_size = len(subjects_list)

    # ----- Check if the number of subjects in each batch is equal to the
    # requested number
    if(not N_batch_size==batch_size):
        raise ValueError(
                ("Number of subjects in each batch is not equal "
                "to the requested number. Please double check!")
            )

    # ----- Job name and submit file
    job_name = f"feature_extraction_Batch{n_batch+1}"
    submit_file = Path(f"{submit_files_folder}/{job_name}.submit")

    # ----- Input parameters for the executable file
    general_args = [
        atlas_name,
        N_network,
        resolution_mm,
        N_ROI,
        N_r,
        emb_dim, 
        r_MSE,
        N_scale
    ]

    # ----- Create an instance from the 'HTCondorSubmissionTemplate' class
    template = HTCondorSubmissionTemplate(
        initial_dir = initial_dir,
        executable_file = executable_file_feature_extraction,
        mail_user = mail_user,
        logs_folder = logs_folder,
        submit_file = submit_file,
        args = general_args,
        subjects_list = subjects_list,
        request_cpus = request_cpus,
        request_memory = request_memory,
        request_disk = request_disk
    )

    # ----- Store the generated .submit files
    template.write_queue_for_feature_extraction()
    
    print(f"Batch {n_batch+1} of {N_batches-1} is generated!")
