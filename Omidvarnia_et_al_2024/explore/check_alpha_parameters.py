"""
This script performs prediction modeling of rsfMRI features on Juseless. It
extracts the best alpha parameter for ridge regression from different scenarios: 
1. No confound removal
2. Confound removal
3. Combined features and confounds
4. Confounds only.

Arguments:
    --atlas_name (str): Type of the brain atlas: 'schaefer' or 'glasser'.

Note:
    - Ensure that the YAML configuration file contains the necessary
        parameters and paths.

Written by: Amir Omidvarnia, PhD
Email: amir.omidvarnia@gmail.com
"""
import numpy as np
import os, sys, yaml, argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

import warnings
warnings.simplefilter("ignore")


# ----- Input arguments
parser = argparse.ArgumentParser(description="Preform prediction modelling of "
                                             "rsfMRI features on Juseless. ")

parser.add_argument(
    "--atlas_name", 
    required=True, 
    type=str, 
    help="Type of the brain atlas: schaefer, glasser"
)

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
model_type = 'ridge'
atlas_name = args.atlas_name

# ----- Input arguments
yaml_file = Path(
    ("/home/aomidvarnia/GIT_repositories/Comm_Biol_manuscript/analysis_codes"
     "/common/config_HCP_icbm152_mask.yaml")
)
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)

# Atlas name
assert atlas_name in ['schaefer', 'glasser'], 'Atlas name is not correct.'

# ----- Parcellation and graph params
N_network = config['feature_extraction'][f'brain_atlas_{atlas_name}']['N_network']
N_ROI = config['feature_extraction'][f'brain_atlas_{atlas_name}']['N_ROI']

# ------ Main folder
main_folder = config['juseless_paths']['main_folder']
if not os.path.isdir(main_folder):
    raise ValueError(f"Folder {main_folder} does not exist!")

# ----- Folder of the output figures
figures_folder = os.path.join(main_folder, f'Figures_{atlas_name}')   
figures_folder = os.path.join(figures_folder, f'{model_type}_check_alpha_param')
os.makedirs(figures_folder, exist_ok=True)

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

# from ptpython.repl import embed
# print("Stop: Step5_perform_prediction_juseless_tsnr0.py")
# embed(globals(), locals())

# ----- Input parameters
prediction_params = config['prediction']
target_labels = config['prediction']['target_labels']
target_labels.remove('TIV')
N_targets = len(target_labels) #  Number of targets
feature_names = prediction_params['feature_names']
N_features = len(feature_names)

# Exclude tSNR from the measures (complexity features)
feature_names = feature_names[1:]
complex_measure_indices = np.arange(1, N_features)
N_features = N_features - 1

# Choose the desired measures (complexity features)
N_subj_vec1 = np.arange(100, 2000, 50)
N_subj_vec2 = np.arange(2000, 20500, 500)
N_subj_vec = np.concatenate([N_subj_vec1, N_subj_vec2])

N_scaling = len(N_subj_vec)

# ------------------------------------
# ------------------------------------
# ------------------------------------
# ------------------------------------
# ------------------------------------
# ------------------------------------
# ------------------------------------
# Extract the best alpha param
# for ridge regression 
# ------------------------------------
# feature_names.append('Confounds')
best_alpha_subjs1 = np.zeros((N_targets, N_features, N_scaling))
best_alpha_subjs2 = np.zeros((N_targets, N_features, N_scaling))
best_alpha_subjs3 = np.zeros((N_targets, N_features, N_scaling))
best_alpha_subjs4 = np.zeros((N_targets, N_scaling))
    
for target_no in range(N_targets):

    target_label = target_labels[target_no]

    ### Loop over N_subjs (scaling)
    n_sample_size = 0
    for n_scale in N_subj_vec:

        ### Loop over complexity measures
        for n_feature in range(N_features):

            # print('OKKKK: Plot prediction accuracies')
            # embed(globals(), locals())
            feature_name = feature_names[n_feature]

            # ----------------------
            # No Conf Removed
            # ----------------------
            accuracy_out_filename_noconfoundRemoved = os.path.join(
                prediction_results_folder,
                (f'accuracy_{model_type}_{feature_name}_Nsubj{n_scale}'
                f'_{target_label}_NoConfoundRemoved.npz')
            )
            
            try:
                out1 = np.load(accuracy_out_filename_noconfoundRemoved, 
                    allow_pickle=True)
                # accuracy_all1 = out1['accuracy_all'][0][0]
                best_alpha_all1 = out1['best_alpha_all'][0][0]
            except:
                # accuracy_all1 = None
                best_alpha_all1 = None
            
            best_alpha_subjs1[target_no, n_feature, n_sample_size] = best_alpha_all1

            # ----------------------
            # Conf Removed
            # ----------------------
            accuracy_out_filename_confoundRemoved = os.path.join(
                prediction_results_folder,
                (f'accuracy_{model_type}_{feature_name}_Nsubj{n_scale}'
                f'_{target_label}_confoundRemoved.npz')
            )

            try:
                out2 = np.load(accuracy_out_filename_confoundRemoved, 
                    allow_pickle=True)
                # accuracy_all2 = out2['accuracy_all'][0][0]
                best_alpha_all2 = out2['best_alpha_all'][0][0]
            except:
                # accuracy_all2 = None
                best_alpha_all2 = None

            best_alpha_subjs2[target_no, n_feature, n_sample_size] = best_alpha_all2

            # ----------------------
            # Combined features
            # ----------------------
            accuracy_out_filename_CombinedFeature = os.path.join(
                prediction_results_folder,
                (f'accuracy_{model_type}_{feature_name}_Nsubj{n_scale}'
                f'_{target_label}_CombinedFeature.npz')
            )

            try:
                out3 = np.load(accuracy_out_filename_CombinedFeature, 
                    allow_pickle=True)
                # accuracy_all3 = out3['accuracy_all'][0][0]
                best_alpha_all3 = out3['best_alpha_all'][0][0]
            except:
                # accuracy_all3 = None
                best_alpha_all3 = None

            best_alpha_subjs3[target_no, n_feature, n_sample_size] = best_alpha_all3

        # ----------------------
        # Confounds only
        # ----------------------
        # from ptpython.repl import embed
        # print('OKKKK: Plot prediction accuracies')
        # embed(globals(), locals())
        accuracy_out_filename_confoundsOnly = os.path.join(
            prediction_results_folder,
            (f'accuracy_{model_type}_Nsubj{n_scale}'
            f'_{target_label}_ConfoundsOnly.npz')
        )

        try:
            out4 = np.load(accuracy_out_filename_confoundsOnly, 
                allow_pickle=True)
            # accuracy_all4 = float(out4['accuracy_all'])
            best_alpha_all4 = float(out4['best_alpha_all'])
        except:
            # accuracy_all4 = None
            best_alpha_all4 = None

        best_alpha_subjs4[target_no, n_sample_size] = best_alpha_all4
        
        n_sample_size = n_sample_size + 1
   
# ------------------------------------
# ------------------------------------
# ------------------------------------
# ------------------------------------
# ------------------------------------
# ------------------------------------
# ------------------------------------
# Plot the histogram of best alpha param
# for ridge regression 

fig_filename = os.path.join(
    figures_folder,
    f'best_alpha_histograms_{model_type}_{atlas_name}.png'
)

# Flatten the vectors
offset = 1e-1
flattened1 = best_alpha_subjs1.flatten() + offset
flattened2 = best_alpha_subjs2.flatten()
flattened2 = flattened2 + np.random.uniform(offset-1e-5, offset+1e-5, size=flattened2.shape)

flattened3 = best_alpha_subjs3.flatten() + offset
flattened4 = best_alpha_subjs4.flatten() + offset
 
# Create a figure with 1x4 subplots
# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
fig = plt.figure(1, figsize=(80, 80))
axs = [fig.add_subplot(221 + i) for i in range(4)]

# Set x-axis to logarithmic scale
for ax in axs:
    ax.set_xscale('log')

# Create KDE plots for each vector with frequency on y-axis
sns.kdeplot(flattened1, fill=True, color='blue', alpha=0.7, ax=axs[0], common_norm=False, cut=0)
axs[0].set_title(f'Scenario 1: No confound removal\n', fontweight='bold', fontsize=130)
axs[0].set_xlabel(f'\nBest \u03B1\n', fontweight='bold', fontsize=80)
axs[0].set_ylabel(f'\nHistogram\n', fontweight='bold', fontsize=80)
axs[0].set_xticks([0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000])
axs[0].set_xticklabels(['', '0', '1', '10', '100', '1k', '10k', '100k', '1m'], fontweight='bold', fontsize=80)
for label in axs[0].get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(80)

# Draw a thick box around the plot
box = Rectangle((0, 0), 1, 1, transform=axs[0].transAxes, linewidth=15, edgecolor='black', facecolor='none')
axs[0].add_patch(box)


sns.kdeplot(flattened2, fill=True, color='green', alpha=0.7, ax=axs[1], common_norm=False, cut=0)
axs[1].set_title(f'Scenario 2: Confound removal\n', fontweight='bold', fontsize=130)
axs[1].set_xlabel(f'\nBest \u03B1\n', fontweight='bold', fontsize=80)
axs[1].set_ylabel(f'\nHistogram\n', fontweight='bold', fontsize=80)
axs[1].set_xticks([0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000])
axs[1].set_xticklabels(['', '0', '1', '10', '100', '1k', '10k', '100k', '1m'], fontweight='bold', fontsize=80)
for label in axs[1].get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(80)
    
# Draw a thick box around the plot
box = Rectangle((0, 0), 1, 1, transform=axs[1].transAxes, linewidth=15, edgecolor='black', facecolor='none')
axs[1].add_patch(box)


sns.kdeplot(flattened3, fill=True, color='orange', alpha=0.7, ax=axs[2], common_norm=False, cut=0)
axs[2].set_title(f'\nScenario 3: Features and confounds\n', fontweight='bold', fontsize=130)
axs[2].set_xlabel(f'\nBest \u03B1\n', fontweight='bold', fontsize=80)
axs[2].set_ylabel(f'\nHistogram\n', fontweight='bold', fontsize=80)
axs[2].set_xticks([0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000])
axs[2].set_xticklabels(['', '0', '1', '10', '100', '1k', '10k', '100k', '1m'], fontweight='bold', fontsize=80)
for label in axs[2].get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(80)
    
# Draw a thick box around the plot
box = Rectangle((0, 0), 1, 1, transform=axs[2].transAxes, linewidth=15, edgecolor='black', facecolor='none')
axs[2].add_patch(box)


sns.kdeplot(flattened4, fill=True, color='red', alpha=0.7, ax=axs[3], common_norm=False, cut=0)
axs[3].set_title(f'\nScenario 4: Confounds only\n', fontweight='bold', fontsize=130)
axs[3].set_xlabel(f'\nBest \u03B1\n', fontweight='bold', fontsize=80)
axs[3].set_ylabel(f'\nHistogram\n', fontweight='bold', fontsize=80)
axs[3].set_xticks([0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000])
axs[3].set_xticklabels(['', '0', '1', '10', '100', '1k', '10k', '100k', '1m'], fontweight='bold', fontsize=80)
for label in axs[3].get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(80)
    
# Draw a thick box around the plot
box = Rectangle((0, 0), 1, 1, transform=axs[3].transAxes, linewidth=15, edgecolor='black', facecolor='none')
axs[3].add_patch(box)


# Adjust layout
plt.tight_layout()
fig.subplots_adjust(top=0.8)
if atlas_name=='glasser':
    plt.suptitle(f'(B) Glasser brain atlas', fontsize=150, fontweight='bold')
else:
    plt.suptitle(f'(A) Schaefer brain atlas', fontsize=150, fontweight='bold')

# Show the plot
plt.show()
plt.savefig(fig_filename, dpi=300)
plt.close()
