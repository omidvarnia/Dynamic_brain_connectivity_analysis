"""
This script generates the mean tSNR (temporal signal-to-noise ratio) map from
subject-specific rsfMRI data using the Glasser or Schaefer brain atlas.

Note:
    - Make sure to have the YAML configuration file located at the
        specified path.
    - Set the 'atlas_name' variable to either 'schaefer' or 'glasser'
        depending on the brain atlas being used.
    - Verify that the directories for subject-specific .npz files 
        ('npzfiles_folder') and parcellation atlas ('parcel_atlas_folder')
        exist and are correctly specified in the YAML configuration file.
    - Specify the 'atlas_filename' variable to point to the appropriate 
        parcellation atlas file (Schaefer or Glasser).
    - Ensure that the output directory for saving figures ('figures_folder') 
        exists or create it if not.

Note:
    - Ensure that the YAML configuration file contains the necessary
        parameters and paths.
        
Written by: Amir Omidvarnia, PhD
Email: amir.omidvarnia@gmail.com
"""
import numpy as np
from pathlib import Path
import os, sys, yaml
import warnings
warnings.simplefilter("ignore")

sys.path.append((Path(__file__).parent.parent / "common").as_posix())
from complexity import complexity_analysis
from vec2nii import vec2image

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

if atlas_name == 'schaefer':
    atlas_filename = os.path.join(
        parcel_atlas_folder, 'schaefer_2018', 'Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'
    )
elif atlas_name == 'glasser':
    atlas_filename = os.path.join(
        parcel_atlas_folder, 'glasser', 'MNI_Glasser_HCP_v1.0.nii.gz'
    )
else:
    raise ValueError('Atlas name is not correct!')

# ----- Folder of the output figures
figures_folder = os.path.join(main_folder, f'Figures_{atlas_name}')
figures_folder = os.path.join(figures_folder, 'Mean_tSNR_map')   
os.makedirs(figures_folder, exist_ok=True)

fig_filename = os.path.join(figures_folder,
                f'Mean_tSNR_map_{atlas_name}.png')
nii_filename = os.path.join(figures_folder,
                f'Mean_tSNR_map_{atlas_name}.nii.gz')

# A npy file containing a dictionary of dataframes for the concatenated 
# brain maps of all subjects for all complexity measures
brain_maps_all_file = os.path.join(
            npzfiles_folder, 'complex_measures_all_df.npy')

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

# -------------------------------------------------
# Extract the tSNR thresholding map
# -------------------------------------------------
brain_map_df_tSNR = dict_of_brain_maps_df['tSNR_fMRI']
brain_map_df_tSNR['eid'] = brain_map_df_tSNR['eid'].astype(int)
tSNR_brain_map_df, tSNR_map_df_mean_ROI_normalized = \
    complexity_analysis.tSNR_MinMaxScaler(brain_map_df_tSNR)
    
# -------------------------------------------------
# Save the mean tSNR map as a nifti file
# -------------------------------------------------
# from ptpython.repl import embed
# print('Stop: plot_mean_tSNR_nifti.py')
# embed(globals(), locals())

vec2image(
    tSNR_map_df_mean_ROI_normalized, 
    atlas_filename,
    nii_filename
)

print(f'* The mean tSNR map was saved as: {nii_filename}.')

