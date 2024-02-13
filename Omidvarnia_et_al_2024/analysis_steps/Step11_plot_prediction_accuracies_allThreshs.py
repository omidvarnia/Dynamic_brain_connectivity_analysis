"""
This script loads subject-specific results and plots group-mean maps
for all tSNR tresholds (2D maps of prediction accuracy across all tSNR
thresholds and population sizes).

It requires numpy, os, sys, yaml, argparse, and matplotlib.pyplot libraries.

Usage example:
    python Step11_plot_prediction_accuracies_allThreshs.py \
        --model_type <model_type> --confound_removal <confound_removal> \
        --atlas_name <atlas_name>

Arguments:
    --model_type: 
        Type of the predictive model (e.g., ridge, rf, svm).
    --confound_removal: 
        Indicates the type of confound removal (0 for no removal, 1 for
        confound removal, 2 for features+confounds).
    --atlas_name: 
        Type of the brain atlas (e.g., schaefer, glasser).

Note:
    This script assumes the availability of a configuration file specified in
    the YAML format.
           
Written by: Amir Omidvarnia, PhD
Email: amir.omidvarnia@gmail.com
"""
import numpy as np
import os, sys, yaml, argparse
from pathlib import Path
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore")


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
    "--confound_removal", 
    required=True, 
    type=int, 
    help="0 for no remove, 1 for confound removal, 2 for features+confounds"
)

parser.add_argument(
    "--atlas_name", 
    required=True, 
    type=str, 
    help="Type of the brain atlas: schaefer, glasser"
)

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
model_type = args.model_type
confound_removal = int(args.confound_removal)
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
figures_folder = os.path.join(figures_folder, 'ThreshAll', f'{model_type}_prediction')   
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

# ----- Input parameters
prediction_params = config['prediction']
target_names = config['prediction']['target_names']
target_labels = config['prediction']['target_labels']
target_labels.remove('TIV')
target_labels2 = target_labels.copy()
target_labels2 = ['Fluid intelligence' if item == 'FI_score' else item for item in target_labels2]
target_labels2 = ['Processing speed' if item == 'reaction_time' else item for item in target_labels2]
target_labels2 = ['Visual memory' if item == 'visual_memory' else item for item in target_labels2]
target_labels2 = ['Numeric memory' if item == 'numeric_memory' else item for item in target_labels2]
target_labels2 = ['Fish consumption' if item == 'fish_consumer' else item for item in target_labels2]
target_labels2 = ['Age at scan' if item == 'AgeAtScan' else item for item in target_labels2]

N_targets = len(target_labels) #  Number of targets
feature_names = prediction_params['feature_names']
N_features = len(feature_names)

# Exclude tSNR from the measures (complexity features)
feature_names = feature_names[1:]
complex_measure_indices = np.arange(1, N_features)
N_features = N_features - 1

feature_names2 = feature_names.copy()
feature_names2 = ['RangeEn' if item == 'RangeEnB_AUC_fMRI' else item for item in feature_names2]
feature_names2 = ['MSE' if item == 'MSE_AUC_fMRI' else item for item in feature_names2]
feature_names2 = ['wPE' if item == 'wPE_fMRI' else item for item in feature_names2]
feature_names2 = ['Hurst' if item == 'RSA_hurst_fMRI' else item for item in feature_names2]
feature_names2 = ['fALFF' if item == 'fALFF_fMRI' else item for item in feature_names2]
feature_names2 = ['LCOR' if item == 'LCOR_fMRI' else item for item in feature_names2]
feature_names2 = ['GCOR' if item == 'GCOR_fMRI' else item for item in feature_names2]
feature_names2 = ['EC' if item == 'eig_cent' else item for item in feature_names2]
feature_names2 = ['wCC' if item == 'wCC' else item for item in feature_names2]

thresh_vec = config['prediction']['thresh_vec']
N_thresh = len(thresh_vec)


# Choose the desired measures (complexity features)
N_subj_vec1 = np.arange(100, 2000, 50)
N_subj_vec2 = np.arange(2000, 20500, 500)
N_subj_vec = np.concatenate([N_subj_vec1, N_subj_vec2])

N_scaling = len(N_subj_vec)

# -------------------------------------
# Plot 2D accuracy images of N_subj vs. N_ROI
# for each target_label and each complexity measure
# after confound removal
# -------------------------------------
for target_no in range(N_targets):

    target_label = target_labels[target_no]
    target_label2 = target_labels2[target_no]

    ### Loop over complexity measures
    for n_feature in range(N_features):

        accuracy_subjs_vs_ROIs = np.zeros((N_thresh, N_scaling))
        feature_name = feature_names[n_feature]
        feature_name2 = feature_names2[n_feature]

        ### Loop over N_subjs (scaling)
        n_sample_size = 0
        for n_scale in N_subj_vec:

            # print('OKKKK: Plot prediction accuracies')
            # embed(globals(), locals())

            if(confound_removal==0):
                conf_stat = 'noConfoundRemoved'
                accuracy_out_filename = os.path.join(
                    prediction_results_folder,
                    (f'accuracy_{model_type}_{feature_name}_Nsubj{n_scale}'
                    f'_{target_label}_NoConfoundRemoved_tsnrAll.npz')
                )

            elif(confound_removal==1):
                conf_stat = 'ConfoundRemoved'
                accuracy_out_filename = os.path.join(
                    prediction_results_folder,
                    (f'accuracy_{model_type}_{feature_name}_Nsubj{n_scale}'
                    f'_{target_label}_confoundRemoved_tsnrAll.npz')
                )

            else: # confound_removal=2
                conf_stat = 'CombinedFeature'
                accuracy_out_filename = os.path.join(
                    prediction_results_folder,
                    (f'accuracy_{model_type}_{feature_name}_Nsubj{n_scale}'
                    f'_{target_label}_CombinedFeature_tsnrAll.npz')
                )

            # from ptpython.repl import embed
            # print("Stop: Step11_plot_prediction_accuracies_allThreshs.py")
            # embed(globals(), locals())
    
            try:
                out = np.load(accuracy_out_filename, allow_pickle=True)
                accuracy_all = out['accuracy_all']
                # best_alpha_all = out['best_alpha_all']
                accuracy_subjs_vs_ROIs[:, n_sample_size] = accuracy_all.flatten()
            except:
                accuracy_subjs_vs_ROIs[:, n_sample_size] = np.nan
                
            n_sample_size = n_sample_size + 1
        
        try:
            N_ROI_supraThresh_all = [out['N_ROI_supraThresh_all'][xx][0] for xx in range(N_thresh)]
        except:
            N_ROI_supraThresh_all = thresh_vec

        # from ptpython.repl import embed
        # print("Stop: Step11_plot_prediction_accuracies_allThreshs.py")
        # embed(globals(), locals())
        
        if conf_stat == 'noConfoundRemoved':
            fig_filename = os.path.join(figures_folder,
                f'Figure2_target_{target_label}' + \
                f'_feature_{feature_names[n_feature]}_{conf_stat}.png')
        elif conf_stat == 'ConfoundRemoved':
            fig_filename = os.path.join(figures_folder,
                f'Figure3_target_{target_label}' + \
                f'_feature_{feature_names[n_feature]}_{conf_stat}.png')
        else:
            fig_filename = os.path.join(figures_folder,
                f'Figure4_target_{target_label}' + \
                f'_feature_{feature_names[n_feature]}_{conf_stat}.png')
            
        fig = plt.figure(1, figsize=(25, 25))
        ax = fig.add_subplot(111)
        im = ax.imshow(accuracy_subjs_vs_ROIs, 
                extent=[N_subj_vec[0],N_subj_vec[-1], 
                N_ROI_supraThresh_all[-1], N_ROI_supraThresh_all[0]], 
                aspect=46, origin='upper')
        
        # Set the colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.3)
        cbar.set_label(label=' ', weight='bold', size=80)
        cbar.ax.tick_params(labelsize=80)
    
        # Define min and max tick labels
        min_val, max_val = np.min(accuracy_subjs_vs_ROIs.flatten()), np.max(accuracy_subjs_vs_ROIs.flatten())
        cbar.set_ticks([min_val, max_val])
        cbar.set_ticklabels([f'{min_val:.2f}', f'{max_val:.2f}'])

        # Set tick label font weight directly
        for tick_label in cbar.ax.get_yticklabels():
            tick_label.set_fontweight('bold')
            
        # Draw a box around the colorbar
        import matplotlib.patches as patches
        box = patches.Rectangle((0, 0), 1, 1, linewidth=15, edgecolor='black', facecolor='none')
        cbar.ax.add_patch(box)
            
        # ax.set_ylabel('N_ROI')
        # ax.set_yticks(N_ROI_supraThresh_all)
        # ax.set_xlabel('No of subjects')
        ax.set_title(f'{feature_name2}\n', fontweight='bold', fontsize=130)
    
        # Set the xticks and xticklabels
        xticks = [2000, 5000, 10000, 15000, 20000]
        xticklabels = ['2k', '5k', '10k', '15k', '20k']
        plt.xticks(xticks, xticklabels)

        if atlas_name == 'glasser':
            yticks = [50, 150, 250, 360]
        else:
            yticks = [50, 150, 250, 350, 400]
            
        yticklabels = [f'{int(number)}' for number in yticks]
        plt.yticks(yticks, yticklabels)
            
        # Make the xticks and yticks bold
        plt.xticks(weight='bold', fontsize=80)
        plt.yticks(weight='bold', fontsize=80)
        
        # Draw a thick box around the plot
        from matplotlib.patches import Rectangle
        box = Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=15, edgecolor='black', facecolor='none')
        ax.add_patch(box)
    
        plt.tight_layout()
        plt.show()
        plt.savefig(fig_filename, dpi=300)
        plt.close()

        print(f'{target_labels[target_no]}: {feature_names[n_feature]}')
