#!/home/aomidvarnia/miniconda3/envs/fMRI_complexity/bin/python
"""
This executable file performs first-level feature extraction from rsfMRI data
using different brain atlases (Schaefer or Glasser). It takes the subject ID
as the first command-line argument and extracts features including fALFF
(fractional amplitude of low-frequency fluctuations), LCOR (local correlation),
and GCOR (global correlation) from pre-processed rsfMRI data. The file paths
for input data and output folders are loaded from a YAML configuration file.

Arguments:
    subj_ID (str): Subject ID for which the features will be extracted.

Usage example:
    python fmri_feature_extraction_executable_schaefer.py 1246275

Notes:
    - This script assumes the availability of a configuration file specified
        in the YAML format.
    - The extracted features are stored in a database file and parcellated
        data file, as specified in the configuration.
    - The feature extraction process is performed using the
        'feature_extraction_first_level' function from the
        'feature_extraction' module.
    - Before executing the script, ensure that it has executable permissions.
        The script can be made executable by running the following command:
        >> chmod +x fmri_feature_extraction_executable.py

         
Written by: Amir Omidvarnia, PhD
Email: amir.omidvarnia@gmail.com
"""
import numpy as np
import os
import time
import sys
import yaml
from pathlib import Path

sys.path.append((Path(__file__).parent.parent / "common").as_posix())
import feature_extraction as fce

# ----- Input arguments
yaml_file = Path(
    ("/home/aomidvarnia/GIT_repositories/Comm_Biol_manuscript/analysis_codes"
     "/common/config_HCP_icbm152_mask.yaml")
)
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)

# -------------------------------------
# Extract input parameters
# -------------------------------------
# Example usage: 
# >> python fmri_feature_extraction_executable_schaefer.py 1246275 complexity jureca spline schaefer 17 2 400 10 2 .5 10

subj_ID = sys.argv[1]
atlas_name = config['analysis_name']['atlas_name']
fALFF_db_folder = config[f'juseless_paths'][f'fALFF_db_folder_{atlas_name}']
parcellated_data_folder = config['juseless_paths'][f'parcellated_data_folder_{atlas_name}']

if atlas_name == 'schaefer':
    fmri_parcellated_filename = os.path.join(
        parcellated_data_folder,
        f'ID{subj_ID}_schaefer.npz'
    )
elif atlas_name == 'glasser':
    fmri_parcellated_filename = os.path.join(
        parcellated_data_folder,
        f'sub-{subj_ID}_ses-2_Glasser_HCP_v1.0_FIX-cleaned_time-series.npz'
    )
else:
    
    raise ValueError('Atlas name is not correct.')

if atlas_name == 'schaefer':
    fALFF_db_file = os.path.join(
        fALFF_db_folder,
        'fALFF_GCOR_LCOR.db')
elif atlas_name == 'glasser':
    fALFF_db_file = os.path.join(
        fALFF_db_folder,
        f'element_{subj_ID}_ukb_glasser.hdf5')
else:
    raise ValueError('Atlas name is not correct.')

# -------------------------------------
# Load/compute the first level fMRI features using Glasser atlas
# Parcellated fALFF/LCOR/GCOR features from Fede's storage
# Storage location on juseless:  /data/group/appliedml/fraimondo/for_amir
# -------------------------------------
t1 = time.time()
Output_flag1 = fce.feature_extraction_first_level(
    config,
    subj_ID,
    fALFF_db_file,
    fmri_parcellated_filename,
    atlas_name
)

t2 = time.time()
print(
    f'Extraction of fALFF, LCOR, GCOR: {Output_flag1}. Elapsed time for subject {subj_ID}: {t2-t1} sec'
)

