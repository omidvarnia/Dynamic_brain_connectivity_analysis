import numpy as np
from pathlib import Path
import os, yaml, sys, argparse
import warnings
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
sys.path.append((Path(__file__).parent.parent / "common").as_posix())

warnings.simplefilter("ignore")

# Access the values of the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--confound_removal", required=True, type=int)

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

args = parser.parse_args()
confound_removal = args.confound_removal
atlas_name = args.atlas_name
model_type = args.model_type

# from ptpython.repl import embed
# print("Stop: Step4_check_tsnr_and_nroi_load_results.py")
# embed(globals(), locals())

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
assert atlas_name in ['schaefer', 'glasser'], 'Atlas name is not correct.'

# ------ Main folder
main_folder = config['juseless_paths']['main_folder']
if not os.path.isdir(main_folder):
    raise ValueError(f"Folder {main_folder} does not exist!")

figures_folder = os.path.join(main_folder, f'Figures_{atlas_name}')
figures_folder = os.path.join(figures_folder, f'{model_type}_accuracy_vs_tsnr_20k')
os.makedirs(figures_folder, exist_ok=True)
 
# ----- Folder of the mean maps and nifti files
prediction_results_folder = \
    config['juseless_paths'][f'prediction_results_folder_{atlas_name}']
prediction_results_folder = os.path.join(
    prediction_results_folder, 
    f'tsnr_analysis_{model_type}'
)
if not os.path.isdir(main_folder):
    raise ValueError(f"Folder {main_folder} does not exist!")

# A npy file containing a dictionary of dataframes for the concatenated 
# brain maps of all subjects for all complexity measures
prediction_params = config['prediction']
feature_names = prediction_params['feature_names']
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
N_measures = len(feature_names)

N_ROI = config['feature_extraction']['brain_atlas_schaefer']['N_ROI']
nroi_vec = np.arange(20, N_ROI+1, 20)
N_thresh = len(nroi_vec)

# from ptpython.repl import embed
# print("Stop: Step4_check_tsnr_and_nroi_load_results.py")
# embed(globals(), locals())
    
# -------------------------------------
# Plot 2D accuracy images of N_subj vs. N_ROI
# for each target_label and each complexity measure
# after confound removal
# -------------------------------------
for target_no in range(N_targets):

    target_label = target_labels[target_no]
    target_label2 = target_labels2[target_no]
    
    # print('OKKKK: Plot prediction accuracies')
    # embed(globals(), locals())

    if(confound_removal==0):
        plot_filename = os.path.join(figures_folder,
            f'Figure2_NoConfoundRemoved_target_{target_label}.png')
    elif(confound_removal==1):
        plot_filename = os.path.join(figures_folder,
            f'Figure3_ConfoundRemoved_target_{target_label}.png')
    else: # confound_removal=2
        plot_filename = os.path.join(figures_folder,
            f'Figure4_CombinedFeatures_target_{target_label}.png')

    ### Loop over complexity measures
    fig = plt.figure(1, figsize=(25, 25))
    ax = fig.add_subplot(111)
    for n_feature in range(1, N_measures): # start from 1 to skip tSNR

        accuracy_subjs_vs_ROIs = np.zeros((N_thresh, 1))
        feature_name = feature_names[n_feature]
        
        if(confound_removal==0):
            accuracy_out_filename = os.path.join(
                prediction_results_folder,
                (f'tsnr_nroi_check_accuracy_{model_type}_{feature_name}_Nsubj20000'
                f'_{target_label}_NoConfoundRemoved.npz')
            )

        elif(confound_removal==1):
            accuracy_out_filename = os.path.join(
            prediction_results_folder,
            (f'tsnr_nroi_check_accuracy_{model_type}_{feature_name}_Nsubj20000_'
            f'{target_label}_confoundRemoved.npz')
            )

        else: # confound_removal=2
            accuracy_out_filename = os.path.join(
            prediction_results_folder,
            (f'tsnr_nroi_check_accuracy_{model_type}_{feature_name}_Nsubj20000_'
            f'{target_label}_CombinedFeature.npz')
            )

        out = np.load(accuracy_out_filename, allow_pickle=True)
        accuracy_all = out['accuracy_all']
        N_ROI_supraThresh_all = out['N_ROI_supraThresh_all']
        
        # Convert the list of arrays to a list of numbers
        N_ROI_supraThresh_all = \
            [(arr[0]) for arr in N_ROI_supraThresh_all]

        accuracy_subjs_vs_ROIs = list(accuracy_all.flatten())

        # from ptpython.repl import embed
        # print("Stop: Step4_plot_auuracy_vs_tsnr_N20k.py")
        # embed(globals(), locals())

        # -------------------------------------
        # N_ROI_supraThresh_all = N_ROI_supraThresh_all[0:-14]
        # accuracy_subjs_vs_ROIs = accuracy_subjs_vs_ROIs[0:-14]
        im = ax.plot(
            N_ROI_supraThresh_all, 
            accuracy_subjs_vs_ROIs, 
            label=feature_name,
            linewidth=20)
    
    # from ptpython.repl import embed
    # print("Stop: Step4_plot_auuracy_vs_tsnr_N20k.py")
    # embed(globals(), locals())
        
    ax.set_title(f'{target_label2}\n', fontweight='bold', fontsize=130)
    ax.set_xlabel('tSNR threshold (%)', fontweight='bold', fontsize=80)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=80)
    
    # Set the xticks and xticklabels
    xticks = [0.05, 0.25, 0.45, 0.65, 0.85]
    xticklabels = ['5%', '25%', '45%', '65%', '85%']
    plt.xticks(xticks, xticklabels)
    
    # Get y-axis ticks
    ytick_positions, ytick_labels = plt.yticks()
    yticks = np.linspace(ytick_positions.min(), ytick_positions.max(), 5)
    yticklabels = [f'{number:.2f}' for number in yticks]
    
    # Draw dashed horizontal lines at each ytick
    plt.yticks(yticks, yticklabels)
    for ytick in yticks:
        plt.hlines(
            ytick, 
            xmin=0, 
            xmax=1, 
            linestyle='dashed', 
            color='gray', 
            alpha=0.8,
            linewidth=12
        )

    # Make the xticks and yticks bold
    plt.xticks(weight='bold', fontsize=80)
    plt.yticks(weight='bold', fontsize=80)
    plt.tight_layout()

    print(f'* Target: {target_label2} --> completed!')
    
    # Draw a thick box around the plot
    box = Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=15, edgecolor='black', facecolor='none')
    ax.add_patch(box)

    plt.show()
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

