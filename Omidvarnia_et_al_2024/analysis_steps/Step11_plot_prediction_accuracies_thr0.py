"""
This script loads subject-specific results and plots group-mean curves of
the zero tSNR treshold (i.e., all ROIs in the feature vectors) and at the 
maximum population size.

Usage example:
    python Step11_plot_prediction_accuracies_thr0.py \
        --model_type <model_type> --confound_removal <confound_removal> \
        --atlas_name <atlas_name>

Arguments:
    --model_type: 
        Type of the predictive model (e.g., ridge, rf, svm).
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
    "--atlas_name", 
    required=True, 
    type=str, 
    help="Type of the brain atlas: schaefer, glasser"
)

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
model_type = args.model_type
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
figures_folder = os.path.join(figures_folder, 'Thresh0', f'{model_type}_prediction')
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
# Make Figure2 of the manuscript
# ------------------------------------
for target_no in range(N_targets):

    target_label = target_labels[target_no]
    target_label2 = target_labels2[target_no]
    accuracy_subjs_vs_ROIs1 = np.zeros((N_features, N_scaling))
    accuracy_subjs_vs_ROIs2 = np.zeros((N_features, N_scaling))
    accuracy_subjs_vs_ROIs3 = np.zeros((N_features, N_scaling))
    accuracy_subjs_vs_ROIs4 = np.zeros((N_scaling, 1))

    ### Loop over N_subjs (scaling)
    n_sample_size = 0
    for n_scale in N_subj_vec:

        ### Loop over complexity measures
        for n_feature in range(N_features):

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
                accuracy_all1 = out1['accuracy_all'][0][0]
                # best_alpha_all1 = out1['best_alpha_all'][0][0]
            except:
                accuracy_all1 = None
                # best_alpha_all1 = None
            
            accuracy_subjs_vs_ROIs1[n_feature, n_sample_size] = accuracy_all1

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
                accuracy_all2 = out2['accuracy_all'][0][0]
                # best_alpha_all2 = out2['best_alpha_all'][0][0]
            except:
                accuracy_all2 = None
                # best_alpha_all2 = None

            accuracy_subjs_vs_ROIs2[n_feature, n_sample_size] = accuracy_all2

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
                accuracy_all3 = out3['accuracy_all'][0][0]
                # best_alpha_all3 = out3['best_alpha_all'][0][0]
            except:
                accuracy_all3 = None
                # best_alpha_all3 = None

            accuracy_subjs_vs_ROIs3[n_feature, n_sample_size] = accuracy_all3

        # ----------------------
        # Confounds only
        # ----------------------
        accuracy_out_filename_confoundsOnly = os.path.join(
            prediction_results_folder,
            (f'accuracy_{model_type}_Nsubj{n_scale}'
            f'_{target_label}_ConfoundsOnly.npz')
        )

        try:
            out4 = np.load(accuracy_out_filename_confoundsOnly, 
                allow_pickle=True)
            accuracy_all4 = float(out4['accuracy_all'])
            # best_alpha_all4 = float(out4['best_alpha_all'])
        except:
            accuracy_all4 = None
            # best_alpha_all4 = None

        accuracy_subjs_vs_ROIs4[n_sample_size] = accuracy_all4
        
        n_sample_size = n_sample_size + 1

    # ---------------------------
    # Figure 2: No Confound removal
    # ---------------------------
    fig_filename = os.path.join(figures_folder,
        f'Figure2_NoConfoundRemoved_target{target_no}.png')

    fig = plt.figure(1, figsize=(25, 25))
    ax = fig.add_subplot(111)
    im = ax.plot(N_subj_vec, accuracy_subjs_vs_ROIs1.T, linewidth=20)
    im = ax.plot(N_subj_vec, accuracy_subjs_vs_ROIs4, 'k', linewidth=20)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=80)
    ax.set_xlabel('No of subjects', fontweight='bold', fontsize=80)
    ax.set_title(f'{target_labels2[target_no]}\n', fontweight='bold', fontsize=130)
    # ax.legend(feature_names, loc='center left', bbox_to_anchor=(1,.5))
    # ax.set_aspect('equal')
    
    ### Set clim for imshow
    target_label = target_labels2[target_no]
    ytick_positions, ytick_labels = plt.yticks()
    yticks = np.linspace(ytick_positions.min(), ytick_positions.max(), 5)
    yticklabels = [f'{number:.2f}' for number in yticks]
    
    if(target_label=='Sex'):
        yticklabels = [f'{int(100*number)}%' for number in yticks]
        
    if(target_label=='Fish consumption'):
        yticks = [0.45, 0.47, 0.50, 0.53, 0.55]
        yticklabels = [f'{int(100*number)}%' for number in yticks]
        
    # Draw dashed horizontal lines at each ytick
    plt.yticks(yticks, yticklabels)
    for ytick in yticks:
        plt.hlines(
            ytick, 
            xmin=N_subj_vec.min(), 
            xmax=N_subj_vec.max(), 
            linestyle='dashed', 
            color='gray', 
            alpha=0.8,
            linewidth=12
        )
        
    # Set the xticks and xticklabels
    xticks = [2000, 5000, 10000, 15000, 20000]
    xticklabels = ['2k', '5k', '10k', '15k', '20k']
    plt.xticks(xticks, xticklabels)

    # Make the xticks and yticks bold
    plt.xticks(weight='bold', fontsize=80)
    plt.yticks(weight='bold', fontsize=80)
    
    # Make xlabels and ylabels bold
    # plt.xlabel('Number of Subjects', weight='bold')
    # plt.ylabel('Accuracy', weight='bold')    

    # Draw a thick box around the plot
    from matplotlib.patches import Rectangle
    box = Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=15, edgecolor='black', facecolor='none')
    ax.add_patch(box)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(fig_filename, dpi=300)
    plt.close()

    # ---------------------------
    # Figure3: Confound removal
    # ---------------------------
    fig_filename = os.path.join(figures_folder,
        f'Figure3_ConfoundRemoved_target{target_no}.png')
        
    fig = plt.figure(1, figsize=(25, 25))
    ax = fig.add_subplot(111)
    im = ax.plot(N_subj_vec, accuracy_subjs_vs_ROIs2.T, linewidth=20)
    im = ax.plot(N_subj_vec, accuracy_subjs_vs_ROIs4, 'k', linewidth=20)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=80)
    ax.set_xlabel('No of subjects', fontweight='bold', fontsize=80)
    ax.set_title(f'{target_labels2[target_no]}\n', fontweight='bold', fontsize=130)
    # ax.legend(feature_names, loc='center left', bbox_to_anchor=(1,.5))
    # plt.tight_layout()
    # plt.gca().set_aspect('equal', adjustable='box')
    
    ### Set clim for imshow
    target_label = target_labels2[target_no]
    ytick_positions, ytick_labels = plt.yticks()
    yticks = np.linspace(ytick_positions.min(), ytick_positions.max(), 5)
    yticklabels = [f'{number:.2f}' for number in yticks]
    
    if(target_label=='Sex'):
        yticklabels = [f'{int(100*number)}%' for number in yticks]
        
    if(target_label=='Fish consumption'):
        yticks = [0.45, 0.47, 0.50, 0.53, 0.55]
        yticklabels = [f'{int(100*number)}%' for number in yticks]
        
    # Draw dashed horizontal lines at each ytick
    plt.yticks(yticks, yticklabels)
    for ytick in yticks:
        plt.hlines(
            ytick, 
            xmin=N_subj_vec.min(), 
            xmax=N_subj_vec.max(), 
            linestyle='dashed', 
            color='gray', 
            alpha=0.8,
            linewidth=12
        )
        
    # Set the xticks and xticklabels
    xticks = [2000, 5000, 10000, 15000, 20000]
    xticklabels = ['2k', '5k', '10k', '15k', '20k']
    plt.xticks(xticks, xticklabels)

    # Make the xticks and yticks bold
    plt.xticks(weight='bold', fontsize=80)
    plt.yticks(weight='bold', fontsize=80)
    
    # Make xlabels and ylabels bold
    # plt.xlabel('Number of Subjects', weight='bold')
    # plt.ylabel('Accuracy', weight='bold')

    # Draw a thick box around the plot
    from matplotlib.patches import Rectangle
    box = Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=15, edgecolor='black', facecolor='none')
    ax.add_patch(box)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(fig_filename, dpi=300)
    plt.close()

    # ---------------------------
    # Figure 4: Combined features with confounds
    # ---------------------------
    fig_filename = os.path.join(figures_folder,
        f'Figure4_CombinedFeatures_target{target_no}.png')

    fig = plt.figure(1, figsize=(25, 25))
    ax = fig.add_subplot(111)
    im = ax.plot(N_subj_vec, accuracy_subjs_vs_ROIs3.T, linewidth=20)
    im = ax.plot(N_subj_vec, accuracy_subjs_vs_ROIs4, 'k', linewidth=20)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=80)
    ax.set_xlabel('No of subjects', fontweight='bold', fontsize=80)
    ax.set_title(f'{target_labels2[target_no]}\n', fontweight='bold', fontsize=130)
    # ax.legend(feature_names, loc='center left', bbox_to_anchor=(1,.5))
    # plt.tight_layout()
    # plt.gca().set_aspect('equal', adjustable='box')
    
    ### Set clim for imshow
    target_label = target_labels2[target_no]
    ytick_positions, ytick_labels = plt.yticks()
    yticks = np.linspace(ytick_positions.min(), ytick_positions.max(), 5)
    yticklabels = [f'{number:.2f}' for number in yticks]
    
    if(target_label=='Sex'):
        yticklabels = [f'{int(100*number)}%' for number in yticks]
        
    if(target_label=='Fish consumption'):
        yticks = [0.45, 0.47, 0.50, 0.53, 0.55]
        yticklabels = [f'{int(100*number)}%' for number in yticks]

    # Draw dashed horizontal lines at each ytick
    plt.yticks(yticks, yticklabels)
    for ytick in yticks:
        plt.hlines(
            ytick, 
            xmin=N_subj_vec.min(), 
            xmax=N_subj_vec.max(), 
            linestyle='dashed', 
            color='gray', 
            alpha=0.8,
            linewidth=12
        )
        
    # Set the xticks and xticklabels
    xticks = [2000, 5000, 10000, 15000, 20000]
    xticklabels = ['2k', '5k', '10k', '15k', '20k']
    plt.xticks(xticks, xticklabels)
    
    # Make the xticks and yticks bold
    plt.xticks(weight='bold', fontsize=80)
    plt.yticks(weight='bold', fontsize=80)

    # Make xlabels and ylabels bold
    # plt.xlabel('Number of Subjects', weight='bold')
    # plt.ylabel('Accuracy', weight='bold')

    # Draw a thick box around the plot
    from matplotlib.patches import Rectangle
    box = Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=15, edgecolor='black', facecolor='none')
    ax.add_patch(box)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(fig_filename, dpi=300)
    plt.close()

    print(f'{target_label}: Figures 2, 3, 4 were created.')
