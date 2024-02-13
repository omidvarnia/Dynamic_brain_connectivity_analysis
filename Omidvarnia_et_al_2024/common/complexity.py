'''
Module Name: julearn_utils

This module contains utility functions and classes for performing prediction
and analysis using Julearn.

Written by: Amir Omidvarnia, PhD
Email: amir.omidvarnia@gmail.com
'''
import numpy as np
import pandas as pd
from pathlib import Path
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import nibabel as nib
import glob
from julearn.pipeline import PipelineCreator
from julearn.pipeline import TargetPipelineCreator 
from sklearn import preprocessing
from julearn import run_cross_validation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import make_scorer
from julearn.scoring import register_scorer
from julearn.utils import configure_logging, logger

sys.path.append((Path(__file__).parent.parent / "common").as_posix())
from feature_extraction import spear_corr

configure_logging(level="INFO")

spearman_scorer = make_scorer(spear_corr)
register_scorer("spear_corr", spearman_scorer)
class complexity_analysis:
    """
    A class for conducting complexity analysis on brain maps and motion data.

    Arguments:
    ----------
    log_filename : str, optional
        Name of the log file, by default "complexity_analysis.txt".
    """
    def __init__(
        self,
        log_filename="complexity_analysis.txt"
    ):

        self.log_filename = log_filename

    def log(self, log_str):
        """
        Write a string to the input log file.

        Arguments:
        ---------
        log_str : str
            String to be written in the log file.

        Returns:
        --------
        None
        """

        assert isinstance(log_str, str), (
            "Please provide input to the logging function as a string!"
        )
        try:
            with open(self.log_filename, "a+") as f:
                f.write(log_str)
        except FileNotFoundError:
            print("Logfile does not exist!")
        finally:
            print(log_str) 

    def vec2image(
    brain_maps_2_nii, 
    atlas_filename,
    output_filename
    ):
        """
        Accepts an ROI-wise vector associated with a brain atlas and generates
        a 3D nifti image based on that.

        Arguments:
        -----------
        brain_maps_2_nii: array of size N_ROI x N_measure
            A 2d array or brain maps for multiple graph measures.
        atlas_filename: str or path
            Path to a parcellated brain atlas with N_ROI regions.
        output_filename: str or path
            Filename of the output file.

        Returns:
        --------
        brain_surf_map: str or path
            Filename of the converted 3d brain surface nifti file.
        """

        N_ROI, N_measure = brain_maps_2_nii.shape

        # Read the brain atlas image
        atlas_img = nib.load(atlas_filename)
        affine_mat = atlas_img.affine
        atlas_img = atlas_img.get_fdata()
        N_x, N_y, N_z = np.shape(atlas_img)
        
        brain_maps_img = np.zeros((N_x, N_y, N_z, N_measure))

        for n_roi in range(N_ROI):

            ind = np.where(atlas_img==n_roi+1)
            ix = ind[0]
            iy = ind[1]
            iz = ind[2]

            for n_meas in range(N_measure):
                brain_maps_img[ix, iy, iz, n_meas] = brain_maps_2_nii[n_roi, n_meas]

        brain_map_nii = nib.Nifti1Image(brain_maps_img, affine=affine_mat)
        brain_map_nii.to_filename(output_filename)

        return atlas_filename

    def within_RSN_mean(
        RSN_labels,
        N_network,
        brain_map):

        """
        Converts the ROI-wise input brain map into a RSN-wise map
        by averaging ROIs within each RSN according to the RSN labels.

        Parameters
        -----------
        RSN_labels: 1d vector of size N_ROI x 1
              Vector of RSN labels for each ROI
        N_network: int
              Number of RSNs in RSN_labels (maximum value in RSN_labels)
        brain_map: array 
              ROI-wise Brain map for which the RSN-wise means should be 
              calculated. It could be a single map (1d) or multiple maps (2d) 
              such as MSE and RangeEn 

        Returns
        --------
        within_RSN_m: N_RSN x 1 vector
            RSN-wise brain map

        """
        N_dim = brain_map.ndim

        if(N_dim==1):
            N_maps = 1
        elif(N_dim==2):
            _, N_maps = brain_map.shape
        else:
            N_maps = None

        within_RSN_m = np.zeros((N_network, N_maps))
        for n_rsn in range(N_network):

            # -----------------------------------------------------
            # -- Extract within-RSN mean (e.g., mean RangeEn for each RSN)
            # ----------------------------------------------------- 
            RSN_labels_within_n = np.where(RSN_labels==n_rsn+1) # is tuple
            RSN_labels_within_n = np.array(RSN_labels_within_n)
            RSN_labels_within_n = RSN_labels_within_n.flatten()

            within_tmp = brain_map[RSN_labels_within_n, :]
            within_tmp = np.median(within_tmp, axis=0)

            within_RSN_m[n_rsn, :] = within_tmp
            del within_tmp

        return within_RSN_m

    def brainmap2df(
    feature_names,
    npzfiles_folder,
    output_file,
    N_ROI
    ):
        """
        Accepts an ROI-wise vector associated with a brain atlas and generates
        a 3D nifti image based on that.

        Arguments:
        -----------
        measure_name: str or path
            Name of the complexity/tSNR measure in the subject-specific
            outfile files (.npz).  
        npzfiles_folder: str or path
            Folder address of the subject-specific .npz files from complexity
            analysis.
        output_file: str or path
            File name of the output measures.
        N_ROI: int
            Number of ROIs.

        Returns:
        --------
        dict_of_brain_maps_df: dictionary of dataframes
            A dictionary of brain map dataframes for all complexity measures.
        """
        # ---------------------------------------
        # Initialize a dictionary of dataframes 
        # for all complexity measures
        # ---------------------------------------
        dict_of_brain_maps_df = {}
        for measure_name in feature_names:

            dict_of_brain_maps_df[measure_name] = pd.DataFrame()

        # Make the column names for the output df.
        # Each ROI is presented by 'x'
        df_col_names = list()
        for n_roi in range(N_ROI):
            df_col_names.append(f'x{n_roi}')

        # ---------------------------------------
        # Load the brain maps and append in a dataframe
        # ---------------------------------------
        npz_files = glob.glob(os.path.join(npzfiles_folder,'ID*.npz'))
        N_subj = len(npz_files)
        n_subj_good = 0

        for output_file_subj in npz_files:

            subj_ID = os.path.basename(output_file_subj)
            subj_ID = subj_ID[2:9]

            # -----------------------------------
            # Load the subject-specific .npz file
            # -----------------------------------
            out = np.load(output_file_subj, allow_pickle=True)
            Output_flag = out['Output_flag']

            # -----------------------------------
            # Make a dataframe of brain maps for each
            # complexity measures and put it in the dictionary
            # -----------------------------------
            if Output_flag == 'Success':

                # -----------------------------------
                # Extract the measures from the .npz file
                # -----------------------------------
                for measure_name in feature_names:

                    measure_ROI_subj = out[measure_name]
                    measure_ROI_subj = np.reshape(
                        measure_ROI_subj, (N_ROI, 1)
                        )

                    # -----------------------------------
                    # Store the measures in the dataframe
                    # -----------------------------------
                    # print('OKKKK: Load results')
                    # embed(globals(), locals())
                    brain_maps_df = dict_of_brain_maps_df[measure_name]
                    measure_ROI_subj_df = pd.DataFrame(
                        measure_ROI_subj.T, 
                        columns=df_col_names, 
                        index=[n_subj_good]
                        )
                    measure_ROI_subj_df.insert(loc=0, column='eid', value=subj_ID)
                    brain_maps_df = brain_maps_df.append(measure_ROI_subj_df)

                    # Update 'dict_of_brain_maps_df'
                    dict_of_brain_maps_df[measure_name] = brain_maps_df

                    del measure_ROI_subj, brain_maps_df

                n_subj_good = n_subj_good + 1
                print(f'n_subj_good = {n_subj_good}')
                del out

        np.save(output_file, dict_of_brain_maps_df)
        print(f'Number of good subjects: {n_subj_good} out of {N_subj}\n')

        return dict_of_brain_maps_df

    def motion2df(
    measure_name,
    motion_type,
    output_folder,
    output_file
    ):
        """ 
        Load the complexity measures of motion files.

        Parameters
        -----------
        measure_name: str
            name of the complexity/tSNR measure in the subject-specific
            outfile files (.npz)
        motion_type: str
            type of the measured motion ('rmsrel' or 'rmsabs') 
        output_folder: str or path
            folder address of the subject-specific .npz files from complexity
            analysis
        output_file: str or path
            file name of the output measures

        Returns
        --------
        brain_surf_map: str or path
            filename of the converted 3d brain surface nifti file

        """
        measure_name = f'{measure_name}_{motion_type}'
        npz_files = glob.glob(os.path.join(output_folder,'ID*.npz'))

        # Load the brain maps and append in a dataframe
        n_subj_good = 0
        n_subj_bad = 0
        motion_df = pd.DataFrame()

        for output_file_subj in npz_files:

            subj_ID = os.path.basename(output_file_subj)
            subj_ID = subj_ID[2:9]

            try:

                # -----------------------------------
                # Load the subject-specific .npz file
                # -----------------------------------
                out = np.load(output_file_subj)
                measure_ROI_subj = float(out[measure_name])

                del out

                # -----------------------------------
                # Store the measures in the dataframe
                # -----------------------------------
                measure_ROI_subj_df = pd.DataFrame(
                    measure_ROI_subj, 
                    columns=['x'], 
                    index=[n_subj_good]
                    )
                measure_ROI_subj_df.insert(loc=0, column='eid', value=subj_ID)
                motion_df = motion_df.append(measure_ROI_subj_df)
                del measure_ROI_subj

                n_subj_good = n_subj_good + 1
                print(f'n_subj_good = {n_subj_good}')

            except Exception as e:

                n_subj_bad = n_subj_bad + 1
                print(f'Error message: {e}')

        if(not motion_df.empty):
            motion_df.to_csv(output_file, index=False, sep=',')

        print(f'Number of good subjects: {n_subj_good}\n')
        print(f'Number of bad subjects: {n_subj_bad}\n')

        return motion_df

    def plot_complexity(
        output_nii_folder,
        tSNR_map_ROI,
        brain_map_df_mean_ROI,
        measure_name,
        Networks,
        RSN_labels
    ):
        """ Plots a mean complexity map versus mean tSNR map, both RSN-wise
        and ROI-wise 

        Parameters
        -----------
        output_nii_folder: str or pth
            folder path where the figures are saved.
        tSNR_map_ROI: array of size N_ROIx1
            ROI-wise mean tSNR map
        brain_map_df_mean_ROI: array of size N_ROIx1
            ROI-wise mean complexity map
        measure_name: str
            complexity measure name used in the output .npz files
        Networks: list of strings
            list of RSN names
        RSN_labels: array of size N_ROIx1
            ROI-wise vector of RSN labels (from 1 to N_RSN)

        Returns
        --------
        None

        """
        # ------------------------------------------
        # Plot the mean complexity map vs. 
        # mean tSNR map (ROI-wise)
        # ------------------------------------------
        plt.figure()        
        plt.plot(tSNR_map_ROI, brain_map_df_mean_ROI, '.')
        plt.xlabel('tSNR')
        plt.ylabel(measure_name)
        plt.show()
        plt.savefig(os.path.join(
            output_nii_folder,
            f'{measure_name}_vs_tSNR_ROI.png')
        )
        plt.close()

        # ------------------------------------------
        # Plot the mean complexity map vs. 
        # mean tSNR map (RSN-wise)
        # ------------------------------------------
        plt.figure()
        N_network = len(Networks)
        fig, axs = plt.subplots(1, N_network, figsize=(9, 3), sharey=True)
        
        fig.suptitle('Categorical Plotting')
        for n_rsn in range(N_network):

            # -----------------------------------------------------
            # Extract within-RSN mean (e.g., mean RangeEn) for each RSN
            # ----------------------------------------------------- 
            RSN_labels_within_n = np.where(RSN_labels==n_rsn+1) # is tuple
            RSN_labels_within_n = np.array(RSN_labels_within_n)
            RSN_labels_within_n = RSN_labels_within_n.flatten()

            tmp_x = tSNR_map_ROI[RSN_labels_within_n]
            tmp_y = brain_map_df_mean_ROI[RSN_labels_within_n]

            axs[n_rsn].plot(tmp_x, tmp_y, '.', label=Networks[n_rsn])
            axs[n_rsn].set_xlabel('tSNR')
            axs[n_rsn].set_ylabel(measure_name)

        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(
            output_nii_folder,
            f'{measure_name}_vs_tSNR_RSN.png')
        )
        plt.close()

    def merge_ukbb_complexity(
        img_subs_only,
        brain_map_df,
        TIV_column
    ):

        """ Merge two dataframes from the ukbb_parser and the complexity 
        maps of all subjects 

        Parameters
        -----------
        img_subs_only: dataframe
            output of ukbb_parser for UKB subjects with fMRI data
        brain_map_df: dataframe
            dataframe of the subject-specific complexity maps
        TIV_column: 1D array
            Array of TIVs for all UKB subjects with fMRI data

        Returns
        --------
        merged_df: dataframe
            merged dataframe

        """
        # Merge the two dataframes according to their list of subject IDs
        merged_df = pd.merge(
            right=img_subs_only, 
            left=brain_map_df, 
            how='inner', 
            left_on='eid', 
            right_on='eid'
        )

        # Add the corresponding TIV values to the merged dataframe 
        merged_df = pd.merge(
            right=TIV_column, 
            left=merged_df, 
            how='inner', 
            left_on='eid', 
            right_on='eid'
        )

        return merged_df

    def tSNR_MinMaxScaler(
        tSNR_brain_map_df
    ):
        """ Scales and translates each subject-specific tSNR map such that it 
        is in the given range of zero and one.

        Parameters
        -----------
        tSNR_brain_map_df: dataframe
            dataframe of the tSNR maps for all subjects

        Returns
        --------
        tSNR_map_df_normalized_ROI: dataframe
            normalized tSNR maps of all subjects
        tSNR_map_df_mean_ROI: array of size N_ROIx1
            mean map of tSNR over all subjects

        """
        # Fit and transform MinMaxScaler on the tSNR maps of all subjects
        tSNR_map_df_normalized_ROI = tSNR_brain_map_df
        tmp = tSNR_map_df_normalized_ROI.iloc[:, 1:]
        tmp = tmp.T
        N_ROI = tmp.shape[0]
        scaler = MinMaxScaler()

        scaler.fit(tmp)
        scaled = scaler.fit_transform(tmp)
        scaled_df = pd.DataFrame(scaled, columns=tmp.columns)

        tSNR_map_df_normalized_ROI.iloc[:, 1:] = scaled_df.T

        # ----------------------------------------
        # Subject to data leakage across subjects?
        # ----------------------------------------
        # Compute the mean map over all subjects
        tSNR_map_df_mean_ROI = tSNR_map_df_normalized_ROI.iloc[:,1:].mean(axis=0)
        tSNR_map_df_mean_ROI = np.array(tSNR_map_df_mean_ROI)
        tSNR_map_df_mean_ROI = tSNR_map_df_mean_ROI.reshape(N_ROI,1)

        return tSNR_map_df_normalized_ROI, tSNR_map_df_mean_ROI

    def thresh_by_tSNR_at_groupLevel(
        tSNR_map_ROI_normalized,
        brain_map_df,
        thresh_level=0
    ):
        """ Plots a mean complexity map versus mean tSNR map, both RSN-wise
        and ROI-wise 
        Parameters
        -----------
        tSNR_map_ROI_normalized: array of size N_ROIx1
            normalized vector of mean tSNR map
        brain_map_df: dataframe
            dataframe of the subject-specific complexity maps
        thresh_level: float 
            a number between 0 and 1 for thresholding of the tSNR map
        Returns
        --------
        brain_map_df_thresh: dataframe
            thresholded complexity maps of all subjects
        """
        # Threshold the ROI-wise measure using the tSNR map and average the 
        # suprathreshold ROIs
        N_row, N_ROI = brain_map_df.shape
        N_ROI = N_ROI - 1

        # Make the column names for the output df.
        # Each ROI is presented by 'x'
        df_col_names = list()
        for n_roi in range(N_ROI):
            df_col_names.append(f'x{n_roi}')

        thresh = tSNR_map_ROI_normalized>thresh_level
        tmp = np.repeat(thresh, N_row, axis=1).T
        tmp = tmp.astype(int)
        tmp = pd.DataFrame(
            tmp, 
            columns=df_col_names
            )

        # Multiply the thresh df and the brain maps of all subjects
        tmp2 = brain_map_df.iloc[:,1:]
        df = tmp.mul(tmp2)

        # Exclude the zero columns (ROIs)
        eid_column = brain_map_df.iloc[:, 0].astype(int)
        df = df.loc[:, (df != 0).any(axis=0)]
        df.insert(loc=0, column='eid', value=eid_column)

        brain_map_df_thresh = df

        return brain_map_df_thresh

    def thresh_by_tSNR_at_subjLevel(
        tSNR_brain_map_df,
        brain_map_df,
        thresh_level
    ):
        """ Plots a mean complexity map versus mean tSNR map, both RSN-wise
        and ROI-wise 

        Parameters
        -----------
        tSNR_brain_map_df: dataframe
            dataframe of the normalized tSNR maps of all subjects
        brain_map_df: dataframe
            dataframe of the subject-specific complexity maps
        thresh_level: float 
            a number between 0 and 1 for thresholding of the tSNR map

        Returns
        --------
        brain_map_df_thresh: dataframe
            thresholded complexity maps of all subjects

        """
        # Threshold the ROI-wise measure using the tSNR map and average the 
        # suprathreshold ROIs
        brain_map_df_thresh = brain_map_df
        tmp = tSNR_brain_map_df.iloc[:,1:]

        _, N_ROI = brain_map_df.shape
        N_ROI = N_ROI - 1

        # Make the column names for the output df.
        # Each ROI is presented by 'x'
        df_col_names = list()
        for n_roi in range(N_ROI):
            df_col_names.append(f'x{n_roi}')

        # Threshold the brain maps according to the tSNR maps
        thresh = tmp>thresh_level
        thresh = thresh.astype(int)

        # Multiply the thresh df and the brain maps of all subjects
        df = thresh.mul(tmp)
        brain_map_df_thresh.iloc[:, 1:] = df

        return brain_map_df_thresh

    def make_balanced_classes(
        df_julearn,
        target_name,
        N_subj_males_or_females
    ):
        """ Make a balanced input for julearn with equal number of 
        males and females

        Parameters
        -----------
        df_julearn: dataframe
            brain maps and the target measure from ukbb_parser
        target_name: str 
            name of the desired data field 
        N_subj_males_or_females: int
            requested number of males and females in the output dataframe.
            The total number of subjects in df_julearn will be:
            2 x N_subj_males_or_females
        N_roi: int
            Number of suprathreshold ROIs in the thresholded brain map

        Returns
        --------
        df_julearn: dataframe
            updated input dataframe after making two balanced classes with
            equal number of males and females. Compatible with julearn
            (run_cross_validation)

        """
        # Exclude the IDs with no rsID data 
        if(not pd.isnull(target_name)):
            df_julearn = df_julearn.dropna(subset = [target_name])
            df_julearn[target_name] = pd.to_numeric(df_julearn[target_name])

        #### Make balanced data
        males = df_julearn.loc[df_julearn['31-0.0'] == 1] # 31-0.0 -> Sex-0.0
        males = males.iloc[0:N_subj_males_or_females, :]
        mean_age_males = np.mean(np.array(males['AgeAtScan']))

        females = df_julearn.loc[df_julearn['31-0.0'] == 0] # 31-0.0-> Sex-0.0
        females = females.iloc[0:N_subj_males_or_females, :]
        mean_age_females = np.mean(np.array(females['AgeAtScan']))
        
        df_julearn = pd.concat([males, females])

        return df_julearn, mean_age_males, mean_age_females


    def labelEncoder(
        merged_df,
        problem_type,
        target_name
    ):
        """ Prepare numerical target labels for classification using
        sklearn's LabelEncoder

        Parameters
        -----------
        merged_df: dataframe
            brain maps and the target measure from ukbb_parser
        problem_type: str
            classification/regression type, compatible with Julearn
        target_name: str 
            name of the desired data field 

        Returns
        --------
        merged_df: dataframe
            updated input dataframe after changing the target labels, 
            if needed.

        """
        if(problem_type=='multiclass_classification' or
                problem_type=='binary_classification'
            ):
                le = preprocessing.LabelEncoder()
                le.fit(merged_df[target_name])
                y = le.transform(merged_df[target_name])
                merged_df[target_name] = y

        return merged_df

    def julearn_prediction(
        problem_type,
        target_name,
        df_julearn,
        model_grid,
        model_type,
        confound_removal=0,
        confounds=None
    ):
        """
        Perform prediction and cross-validation using Julearn.

        Parameters
        ----------
        problem_type : str
            Type of problem, either "classification" or "regression".
        target_name : str
            Name of the target variable.
        df_julearn : DataFrame
            DataFrame containing brain maps and the target measure.
        model_grid : dict
            Dictionary of model parameters.
        model_type : str
            Type of model to use, either "ridge", "rf" (random forest),
            or "svm" (support vector machine).
        confound_removal : int, optional
            Type of confound removal:
            - 0: No confound removal
            - 1: Confound removal at feature level
            - -1: Confound removal at target level
            - 2: Input features + confound features
            Defaults to 0.
        confounds : list, optional
            List of confounding variables. Defaults to None.

        Returns
        -------
        model_trained : object
            Trained model object.
        scores : dict
            Dictionary containing cross-validation scores.
        scores_summary : dict
            Summary statistics of cross-validation scores.
        inspector : object
            Inspector object containing additional information.
        mean_score : float
            Mean score obtained from cross-validation.
        best_param : object or None
            Best parameter found during cross-validation, or None if not
            applicable.
        class_counts : array-like
            Class counts if problem type is classification, otherwise an
            empty list.
        """
        # Extraxt the feature names
        X_labels = list(df_julearn.filter(regex=r'x\d+', axis=1).columns)
        N_ROI = len(X_labels)
        
        # Extract the target name
        y_label = target_name

        # Assert model_type
        assert model_type in ["ridge", "rf", "svm"], \
            "model_type should be either ridge, rf, or svm"

        # Extract the regression model name and its parameters
        # Ref of the RF params:
        # https://cran.r-project.org/web/packages/randomForest/randomForest.pdf
        if(model_type == "ridge" and problem_type=='regression'):
            model_name = model_grid["ridge_regressor"]["model_name"]
            alpha = model_grid["ridge_regressor"]['ridge__alpha']
        elif(model_type == "ridge" and problem_type=='binary_classification'):
            model_name = model_grid["ridge_classifier"]["model_name"]
            alpha = model_grid["ridge_classifier"]['ridge__alpha']
        elif(model_type == "svm" and problem_type=='binary_classification'):
            model_name = model_grid["hueristic_svc"]["model_name"]
        elif(model_type == "svm" and problem_type=='regression'):
            model_name = model_grid["hueristic_svr"]["model_name"]
        elif(model_type == "rf"):
            model_name = model_grid["rf"]["model_name"]
            n_estimators = model_grid["rf"]['rf__n_estimators']
            max_depth = model_grid["rf"]['rf__max_depth']
            min_samples_leaf = model_grid["rf"]['rf__min_samples_leaf']
            min_samples_split = model_grid["rf"]['rf__min_samples_split']
            max_samples = model_grid["rf"]['rf__max_samples'] # sampsize
            max_leaf_nodes = model_grid["rf"]['rf__max_leaf_nodes'] # maxnodes
            bootstrap = model_grid["rf"]['rf__bootstrap']
        else:
            raise ValueError("model_type not recognized")


        if(problem_type == 'binary_classification'): # Binary classification

            # Check if the classes are balanced
            class_counts = df_julearn[y_label].value_counts()
            
            # Ref: page 18 - 
            # https://cran.r-project.org/web/packages/randomForest/randomForest.pdf
            max_features = int(np.fix(N_ROI/3))

            # Run cross validation using Julearn 
            if(confound_removal==1):

                # Create a pipeline with confound removal at the feature level
                creator = PipelineCreator(
                    problem_type='classification',
                    apply_to="features"
                )
                creator.add("confound_removal", confounds="confounds")
                # cv = KFold(n_splits=5, shuffle=True, random_state=200)
                cv_outer = RepeatedStratifiedKFold(
                    n_splits=5, n_repeats=5, random_state=200
                )
                cv_inner = StratifiedKFold(
                    n_splits=5, random_state=0, shuffle=True
                )
                search_params = {
                    "kind": "grid",
                    "cv": cv_inner
                }

                if(model_type=="ridge"):
                    creator.add(model_name, alpha=alpha)
                elif(model_type=="rf"):
                    creator.add(
                        model_name, 
                        n_estimators=n_estimators, 
                        max_depth=max_depth,
                        max_samples = max_samples,
                        max_leaf_nodes = max_leaf_nodes,
                        min_samples_split = min_samples_split,
                        min_samples_leaf = min_samples_leaf,
                        max_features = max_features,
                        bootstrap = bootstrap
                    )
                elif(model_type=="svm"):
                    eval(f'creator.add({model_name}())')

                # Predict using Julearn
                scores, model_trained, inspector = run_cross_validation(
                X=X_labels+confounds, y=y_label, data=df_julearn, 
                X_types={"features": X_labels, "confounds": confounds},
                model=creator,
                return_estimator='all',
                return_inspector=True,
                return_train_score=True,
                cv=cv_outer,
                search_params=search_params,
                scoring=['accuracy', 'balanced_accuracy'])

            elif(confound_removal==0):

                # Create a pipeline with confound removal at the feature level
                creator = PipelineCreator(
                    problem_type='classification', 
                    apply_to="features",
                )
                creator.add("zscore")
                # cv = KFold(n_splits=5, shuffle=True, random_state=200)
                cv_outer = RepeatedStratifiedKFold(
                    n_splits=5, n_repeats=5, random_state=200
                )
                cv_inner = StratifiedKFold(
                    n_splits=5, random_state=0, shuffle=True
                )
                search_params = {
                    "kind": "grid",
                    "cv": cv_inner
                }

                if(model_type=="ridge"):
                    creator.add(model_name, alpha=alpha)
                elif(model_type=="rf"):
                    creator.add(
                        model_name, 
                        n_estimators=n_estimators, 
                        max_depth=max_depth,
                        max_samples = max_samples,
                        max_leaf_nodes = max_leaf_nodes,
                        min_samples_split = min_samples_split,
                        min_samples_leaf = min_samples_leaf,
                        max_features = max_features,
                        bootstrap = bootstrap
                    )
                elif(model_type=="svm"):
                    eval(f'creator.add({model_name}())')

                scores, model_trained, inspector = run_cross_validation(
                X=X_labels, y=y_label, data=df_julearn,
                X_types={"features": X_labels},
                model=creator,
                return_estimator='all',
                return_inspector=True,
                return_train_score=True,
                cv=cv_outer,
                search_params=search_params,
                scoring=['accuracy', 'balanced_accuracy'])

            elif(confound_removal==-1): # Confounds only as input features
                
                # Create a pipeline with confound removal at the feature level
                creator = PipelineCreator(
                    problem_type='classification', 
                    apply_to="features"
                )
                creator.add("zscore")
                # cv = KFold(n_splits=5, shuffle=True, random_state=200)
                cv_outer = RepeatedStratifiedKFold(
                    n_splits=5, n_repeats=5, random_state=200
                )
                cv_inner = StratifiedKFold(
                    n_splits=5, random_state=0, shuffle=True
                )
                search_params = {
                    "kind": "grid",
                    "cv": cv_inner
                }
                
                if(model_type=="ridge"):
                    creator.add(model_name, alpha=alpha)
                elif(model_type=="rf"):
                    creator.add(
                        model_name, 
                        n_estimators=n_estimators, 
                        max_depth=max_depth,
                        max_samples = max_samples,
                        max_leaf_nodes = max_leaf_nodes,
                        min_samples_split = min_samples_split,
                        min_samples_leaf = min_samples_leaf,
                        max_features = max_features,
                        bootstrap = bootstrap
                    )
                elif(model_type=="svm"):
                    eval(f'creator.add({model_name}())')

                scores, model_trained, inspector = run_cross_validation(
                X=confounds, y=y_label, data=df_julearn,
                X_types={"features": confounds},
                model=creator,
                return_estimator='all',
                return_inspector=True,
                return_train_score=True,
                cv=cv_outer,
                search_params=search_params,
                scoring=['accuracy', 'balanced_accuracy'])

            elif(confound_removal==2): # (input features + confound features)

                # Create a pipeline with confound removal at the feature level
                creator = PipelineCreator(
                    problem_type='classification', 
                    apply_to="features"
                )
                creator.add("zscore")
                # cv = KFold(n_splits=5, shuffle=True, random_state=200)
                cv_outer = RepeatedStratifiedKFold(
                    n_splits=5, n_repeats=5, random_state=200
                )
                cv_inner = StratifiedKFold(
                    n_splits=5, random_state=0, shuffle=True
                )
                search_params = {
                    "kind": "grid",
                    "cv": cv_inner
                }
                
                if(model_type=="ridge"):
                    creator.add(model_name, alpha=alpha)
                elif(model_type=="rf"):
                    creator.add(
                        model_name, 
                        n_estimators=n_estimators, 
                        max_depth=max_depth,
                        max_samples = max_samples,
                        max_leaf_nodes = max_leaf_nodes,
                        min_samples_split = min_samples_split,
                        min_samples_leaf = min_samples_leaf,
                        max_features = max_features,
                        bootstrap = bootstrap
                    )
                elif(model_type=="svm"):
                    eval(f'creator.add({model_name}())')

                X_labels2 = X_labels + confounds

                scores, model_trained, inspector = run_cross_validation(
                X=X_labels2, y=y_label, data=df_julearn,
                X_types={"features": X_labels2},
                model=creator,
                return_estimator='all',
                return_inspector=True,
                return_train_score=True,
                cv=cv_outer,
                search_params=search_params,
                scoring=['accuracy', 'balanced_accuracy'])

            else:
                raise ValueError('Invalid confound_removal')

            # Compue classification score means
            mean_test_accuracy = scores['test_accuracy'].mean()
            mean_test_balanced_accuracy = \
                scores['test_balanced_accuracy'].mean()

            scores_summary = {
                'mean_test_accuracy':mean_test_accuracy,
                'mean_test_balanced_accuracy':mean_test_balanced_accuracy
            }

            mean_score = mean_test_balanced_accuracy
            if(model_type=="ridge"):
                best_param = model_trained.best_params_
                if 'ridge__alpha' in best_param:
                    best_param = best_param['ridge__alpha']
                elif 'ridge_target_transform__model__ridge__alpha' in best_param:
                    best_param = best_param[
                        'ridge_target_transform__model__ridge__alpha'
                    ]
                else:
                    best_param = None
            elif(model_type=="rf" or model_type=="svm"):
                best_param = None

        elif(problem_type == 'regression'):

            # There is no class imbalance in regression
            class_counts = []

            # Ref: page 18 - 
            # https://cran.r-project.org/web/packages/randomForest/randomForest.pdf
            max_features = int(np.sqrt(N_ROI))
            
            # Run cross validation using Julearn
            if(confound_removal==1):
                               
                # Create a pipeline for confound removal at the target level
                target_pipeline_creator = TargetPipelineCreator()
                target_pipeline_creator.add(
                    "confound_removal", 
                    confounds="confounds"
                )

                # Create a pipeline for the main model
                creator = PipelineCreator(
                    problem_type='regression', 
                    apply_to="features"
                )
                creator.add("zscore")
                creator.add(target_pipeline_creator, apply_to="target")

                if(model_type=="ridge"):
                    creator.add(model_name, alpha=alpha)
                elif(model_type=="rf"):
                    creator.add(
                        model_name, 
                        n_estimators=n_estimators, 
                        max_depth=max_depth,
                        max_samples = max_samples,
                        max_leaf_nodes = max_leaf_nodes,
                        min_samples_split = min_samples_split,
                        min_samples_leaf = min_samples_leaf,
                        max_features = max_features,
                        bootstrap = bootstrap
                    )
                elif(model_type=="svm"):
                    eval(f'creator.add({model_name}())')

                scores, model_trained, inspector = run_cross_validation(
                X=X_labels+confounds, y=y_label, data=df_julearn, 
                X_types={"features": X_labels, "confounds": confounds},
                model=creator,
                return_estimator='all',
                return_inspector=True,
                return_train_score=True,
                scoring=["spear_corr", "r2"])

            elif(confound_removal==0):

                # Create a pipeline with no confound removal
                creator = PipelineCreator(
                    problem_type='regression', 
                    apply_to="features"
                )
                creator.add("zscore")
                if(model_type=="ridge"):
                    creator.add(model_name, alpha=alpha)
                elif(model_type=="rf"):
                    creator.add(
                        model_name, 
                        n_estimators=n_estimators, 
                        max_depth=max_depth,
                        max_samples = max_samples,
                        max_leaf_nodes = max_leaf_nodes,
                        min_samples_split = min_samples_split,
                        min_samples_leaf = min_samples_leaf,
                        max_features = max_features,
                        bootstrap = bootstrap
                    )
                elif(model_type=="svm"):
                    eval(f'creator.add({model_name}())')

                scores, model_trained, inspector = run_cross_validation(
                X=X_labels, y=y_label, data=df_julearn,
                X_types={"features": X_labels},
                model=creator,
                return_estimator='all',
                return_inspector=True,
                return_train_score=True,
                scoring=["spear_corr", "r2"])

            elif(confound_removal==-1): # confounds only

                # Create a pipeline with confounds as input features
                creator = PipelineCreator(
                    problem_type='regression', 
                    apply_to="features"
                )
                creator.add("zscore")
                if(model_type=="ridge"):
                    creator.add(model_name, alpha=alpha)
                elif(model_type=="rf"):
                    creator.add(
                        model_name, 
                        n_estimators=n_estimators, 
                        max_depth=max_depth,
                        max_samples = max_samples,
                        max_leaf_nodes = max_leaf_nodes,
                        min_samples_split = min_samples_split,
                        min_samples_leaf = min_samples_leaf,
                        max_features = max_features,
                        bootstrap = bootstrap
                    )
                elif(model_type=="svm"):
                    eval(f'creator.add({model_name}())')

                scores, model_trained, inspector = run_cross_validation(
                X=confounds, y=y_label, data=df_julearn,
                X_types={"features": confounds},
                model=creator,
                return_estimator='all',
                return_inspector=True,
                return_train_score=True,
                scoring=["spear_corr", "r2"])

            elif(confound_removal==2): # (input features + confound features)

                X_labels2 = X_labels + confounds

                # Create a pipeline with confounds as input features
                creator = PipelineCreator(
                    problem_type='regression', 
                    apply_to="features"
                )
                creator.add("zscore")
                if(model_type=="ridge"):
                    creator.add(model_name, alpha=alpha)
                elif(model_type=="rf"):
                    creator.add(
                        model_name, 
                        n_estimators=n_estimators, 
                        max_depth=max_depth,
                        max_samples = max_samples,
                        max_leaf_nodes = max_leaf_nodes,
                        min_samples_split = min_samples_split,
                        min_samples_leaf = min_samples_leaf,
                        max_features = max_features,
                        bootstrap = bootstrap
                    )
                elif(model_type=="svm"):
                    eval(f'creator.add({model_name}())')

                scores, model_trained, inspector = run_cross_validation(
                X=X_labels2, y=y_label, data=df_julearn,
                X_types={"features": X_labels2},
                model=creator,
                return_estimator='all',
                return_inspector=True,
                return_train_score=True,
                scoring=["spear_corr", "r2"])

            else:
                    
                raise ValueError(
                    "The 'confound_removal' input input is unknown!"
                )

            # Compue classification score means
            mean_test_spear_corr = scores['test_spear_corr'].mean()
                
            scores_summary = {
                'mean_test_spear_corr':mean_test_spear_corr
            }

            mean_score = mean_test_spear_corr
            
            if(model_type=="ridge"):
                best_param = model_trained.best_params_
                if 'ridge__alpha' in best_param:
                    best_param = best_param['ridge__alpha']
                elif 'ridge_target_transform__model__ridge__alpha' in best_param:
                    best_param = best_param[
                        'ridge_target_transform__model__ridge__alpha'
                    ]
                else:
                    best_param = None
            elif(model_type=="rf" or model_type=="svm"):
                best_param = None

        else:

            raise ValueError(
                    "The 'study_type' input input is unknown!"
                )

        return model_trained, scores, scores_summary,\
               inspector, mean_score, best_param, class_counts

    def compare_male_female(
        main_folder,
        df_julearn,
        target_names,
        target_labels,
        measure_name,
        N_roi
    ):
        """ Compare gruop mean complexity values between males and females 

        Parameters
        -----------
        df_julearn: dataframe
            brain maps and the target measure from ukbb_parser
        target_name: str 
            name of the desired data field 
        N_roi: int
            Number of suprathreshold ROIs in the thresholded brain map

        Returns
        --------
        out: dataframe
            Statistics of the complexity maps between males and females

        """
        import seaborn as sns
        sns.set_theme(style="white", palette=None)

        Figs_folder = os.path.join(main_folder, 'Figures', 'Step4')
        if(not os.path.isdir(os.path.join(main_folder, 'Figures'))):
            os.mkdir(os.path.join(main_folder, 'Figures'))
        if(not os.path.isdir(Figs_folder)):
            os.mkdir(Figs_folder)

        ss = 0
        for target_name in target_names:

            target_label = target_labels[ss]

            try:
                # Exclude the IDs with no rsID data 
                df = df_julearn.dropna(subset = [target_name])

                #### Make balanced data
                males = df.loc[df['31-0.0'] == 1] # 31-0.0 -> Sex-0.0
                males = males.iloc[0:10000, :]
                males['average'] = males.iloc[:,1:N_roi+1].mean(axis=1)
                males[target_name] = pd.to_numeric(males[target_name])
                cov = np.cov(males.iloc[:,1:N_roi+1])
                ev , eig = np.linalg.eig(cov)
                males['PC1'] = eig[:,0]
                del cov, ev, eig

                females = df.loc[df['31-0.0'] == 0] # 31-0.0-> Sex-0.0
                females = females.iloc[0:10000, :]
                females['average'] = females.iloc[:,1:N_roi+1].mean(axis=1)
                females[target_name] = pd.to_numeric(females[target_name])
                cov = np.cov(females.iloc[:,1:N_roi+1])
                ev , eig = np.linalg.eig(cov)
                females['PC1'] = eig[:,0]
                del cov, ev, eig

                # --------------------------------------------
                # Check progress over time
                # --------------------------------------------
                p1 = np.polyfit(males[target_name], males['PC1'], 1)
                males_vs_age = np.polyval(p1, pd.to_numeric(males[target_name]))
                males['average_line'] = males_vs_age

                p2 = np.polyfit(females[target_name], females['PC1'], 1)
                females_vs_age = np.polyval(p2, pd.to_numeric(females[target_name]))
                females['average_line'] = females_vs_age

                tmp = pd.concat([males, females], axis=0)

                # --------------------------------------------
                # Plot 1 using seaborn
                # --------------------------------------------
                plot_filename1 = os.path.join(Figs_folder,
                    f'Step4_{measure_name}_{target_label}_1.png')

                graph1 = sns.jointplot(
                    data=tmp,
                    x=target_name, y="PC1", hue="31-0.0",
                    cmap="Reds", kind="kde",
                    shade=True, shade_lowest=False, alpha=.5
                )
                ax = plt.gca()
                ax.legend(["M", "F"])
                graph1.set_axis_labels(target_name, f'Mean {measure_name}')
                fig = graph1.fig

                graph1.ax_joint.plot(males[target_name], males['average_line'], 
                'r-', linewidth = 6)
                graph1.ax_joint.plot(
                    females[target_name], females['average_line'], 
                    'b-', linewidth = 6
                )

                fig = fig.savefig(plot_filename1)

                print(
                    f'Step4_{measure_name}_{target_label}_1.png: completed ({ss})!'
                )
                ss = ss + 1

            except:

                print(
                    f'Step4_{measure_name}_{target_label}_1.png: Problematic!'
                )
