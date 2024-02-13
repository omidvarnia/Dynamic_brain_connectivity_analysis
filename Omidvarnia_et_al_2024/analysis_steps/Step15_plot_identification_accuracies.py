"""
This script analyzes and visualizes the results of fingerprinting analysis of
rsfMRI measures. It takes command-line arguments for the model type and
confound removal strategy, loads configurations from a YAML file, and sets up
folder paths. The script loads results from previous analyses, processes the
data, and generates plots to visualize the identification accuracy of different 
complexity measure pairs across different population sizes.

Arguments:
    model_type (str): Type of the model used in the analysis, one of 
        "ridge_regressor," "ridge_classifier," "rf," or "heuristic_svm."
    confound_removal (int): Confound removal strategy: 0 for no removal, 1
        for confound removal, 2 for features + confounds.

Usage example:
    python Step15_plot_identification_accuracies.py --model_type\
        ridge_regressor --confound_removal 1

Note:
    This script assumes the availability of a configuration file specified in
    the YAML format and the existence of result files from previous analyses.
          
Written by: Amir Omidvarnia, PhD
Email: amir.omidvarnia@gmail.com
"""
import numpy as np
import os, yaml, argparse
from pathlib import Path
import scipy.io as sio

import warnings
warnings.simplefilter("ignore")

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

# Access the values of the arguments
model_type = args.model_type
confound_removal = args.confound_removal
atlas_name = "schaefer"     # Atlas name in nilearn

# ----- Array of slurm files. Each file includes multiple sruns.
batch_subjects = config['prediction']['subjects_span']['batch_subjects']
N_batches = len(batch_subjects) - 1 #  Number of batch jobs 

# ------ Main folder
main_folder = config['juseless_paths']['main_folder']
if not os.path.isdir(main_folder):
    raise ValueError(f"Folder {main_folder} does not exist!")

# ----- Results folder
fingerprinting_results_folder = \
    config['juseless_paths'][f'fingerprinting_results_folder_{atlas_name}']

# ----- Folder of the output figures
figures_folder = os.path.join(main_folder, f'Figures_{atlas_name}')   
figures_folder = os.path.join(figures_folder, f'{model_type}_fingerprinting')
os.makedirs(figures_folder, exist_ok=True)

# ----- Subjects list
subjects_list_file = config['juseless_paths']['subjects_list_file']
if not os.path.isfile(subjects_list_file):
    raise ValueError(f"File {subjects_list_file} does not exist!")
    
full_list = sio.loadmat(subjects_list_file)
full_list = full_list['subs_fitting']

# ----- Feature names
prediction_params = config['prediction']
feature_names = prediction_params['feature_names']
N_features = len(feature_names)

feature_labels = [
    'RangeEn', 
    'MSE', 
    'wPE', 
    'Hurst', 
    'fALFF', 
    'LCOR', 
    'GCOR',
    'eigCent', 
    'wCC'
]

# Exclude tSNR from the measures (complexity features)
feature_names = feature_names[1:]
N_features = N_features - 1

# Choose the desired measures (complexity features)
N_subj_vec1 = np.arange(100, 2000, 50)
N_subj_vec2 = np.arange(2000, 20500, 500)
N_subj_vec = np.concatenate([N_subj_vec1, N_subj_vec2])

N_scaling = len(N_subj_vec)

# ---------------------------
# Load the results
# ---------------------------
interesting_Idiff_pairs = np.zeros((10, N_batches))

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
    
    if(confound_removal==0):
        output_file = os.path.join(fingerprinting_results_folder, 
            f'Ideff_accu_ishealthy_noConfRemoved_N{N_batch_size}.npz')

        fig_file = os.path.join(figures_folder,
            f'Ideff_accu_ishealthy_noConfRemoved_N{N_batch_size}.png')

        fig_title = f'Before removing\nage, gender, and TIV\n'

    elif(confound_removal==1):
        output_file = os.path.join(fingerprinting_results_folder, 
            f'Ideff_accu_ishealthy_ConfRemoved_N{N_batch_size}.npz')

        fig_file = os.path.join(figures_folder,
            f'Ideff_accu_ishealthy_ConfRemoved_N{N_batch_size}.png')

        fig_title = f'After removing\nage, gender, and TIV\n'

    else:
        output_file = os.path.join(fingerprinting_results_folder, 
            f'Ideff_accu_ishealthy_CombinedFeature_N{N_batch_size}.npz')

        fig_file = os.path.join(figures_folder,
            f'Ideff_accu_ishealthy_CombinedFeature_N{N_batch_size}.png')

        fig_title = f'Combined with\nage, gender, and TIV\n'

    out = np.load(output_file, allow_pickle=True)
    Ideff_accu = out['Ideff_accu']

    # -----------------------
    # Save the non-zero Ideff pairs
    # -----------------------
    interesting_Idiff_pairs[0, n_batch] = Ideff_accu[0, 2] # RangeEn/wPE
    interesting_Idiff_pairs[1, n_batch] = Ideff_accu[1, 3] # MSE/Hurst
    interesting_Idiff_pairs[2, n_batch] = Ideff_accu[4, 5] # fALFF/LCOR
    interesting_Idiff_pairs[3, n_batch] = Ideff_accu[4, 6] # fALFF/GCOR
    interesting_Idiff_pairs[4, n_batch] = Ideff_accu[5, 6] # LCOR/GCOR
    interesting_Idiff_pairs[5, n_batch] = Ideff_accu[1, 7] # MSE/EC
    interesting_Idiff_pairs[6, n_batch] = Ideff_accu[1, 8] # MSE/wCC
    interesting_Idiff_pairs[7, n_batch] = Ideff_accu[3, 7] # Hurst/EC
    interesting_Idiff_pairs[8, n_batch] = Ideff_accu[3, 8] # Hurst/wCC
    interesting_Idiff_pairs[9, n_batch] = Ideff_accu[7, 8] # EC/wCC
    Ideff_pair_labels = ['RangeEn/wPE', 'MSE/Hurst', 
                         'fALFF/LCOR', 'fALFF/GCOR', 'LCOR/GCOR',
                         'MSE/EC', 'MSE/wCC', 'Hurst/EC',
                         'Hurst/wCC', 'EC/wCC']

    # -----------------------
    # Plot
    # -----------------------
    if N_batch_size == 20000:
        import matplotlib.pyplot as plt
        
        fig = plt.figure(1, figsize=(25, 25))
        ax = fig.add_subplot(111)

        img = ax.imshow(Ideff_accu, cmap=plt.get_cmap('hot'), vmin=0, vmax=1)
        ax.set_xticks(range(N_features))
        ax.set_xticklabels(feature_labels, fontsize=80, fontweight='bold', rotation=90)

        ax.set_yticks(range(N_features))
        ax.set_yticklabels(feature_labels, fontsize=80, fontweight='bold')
        # ax.set_title(fig_title, fontweight='bold')

        for (j,i),label in np.ndenumerate(Ideff_accu):
            if(label>0.01):
                ax.text(i,j,f"{label:.2f}",ha='center',va='center', 
                color='g', fontsize=60, fontweight='bold')

        # fig.colorbar(img)
        plt.tight_layout()
        plt.savefig(fig_file, dpi=300)
        plt.close()

        print(f'The figure for N_subj = {N_batch_size} was saved!')

# ------------------
# Plot the non-zero Ideff pairs
# ------------------
if(confound_removal==0):
    fig_file = os.path.join(figures_folder, 
        f'Ideff_accu_ishealthy_noConfRemoved_{model_type}.png')
    fig_title = f'(B) Before removing\nage, gender, and TIV\n'

elif(confound_removal==1):
    fig_file = os.path.join(figures_folder, 
        f'Ideff_accu_ishealthy_ConfRemoved_{model_type}.png')
    fig_title = f'(C) After removing\nage, gender, and TIV\n'

else:
    fig_file = os.path.join(figures_folder, 
        f'Ideff_accu_ishealthy_CombinedFeature_{model_type}.png')
    fig_title = f'(D) Combined with\nage, gender, and TIV\n'

import matplotlib.pyplot as plt
fig = plt.figure(1, figsize=(25, 25))
ax = fig.add_subplot(111)
img = ax.plot(N_subj_vec, interesting_Idiff_pairs.T, linewidth=20)

ytick_positions, ytick_labels = plt.yticks()
yticks = np.linspace(ytick_positions.min(), ytick_positions.max(), 7)
yticklabels = [f'{number:.2f}' for number in yticks]

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
    
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
# ax.legend(Ideff_pair_labels, loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel('No of subjects', fontweight='bold', fontsize=80)
ax.set_ylabel('Identification accuracy', fontweight='bold', fontsize=80)
ax.set_ylim([0, 1])
ax.set_title(fig_title, fontweight='bold', fontsize=120)

# Draw a thick box around the plot
from matplotlib.patches import Rectangle
box = Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=15, edgecolor='black', facecolor='none')
ax.add_patch(box)
    
plt.tight_layout()
plt.show()
plt.savefig(fig_file, dpi=300)
plt.close()
