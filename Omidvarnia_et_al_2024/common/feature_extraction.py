#!/home/aomidvarnia/miniconda3/envs/fMRI_complexity/bin/python
"""
This script contains functions for feature extraction and analysis of
fMRI data.

Written by: Amir Omidvarnia, PhD
Email: amir.omidvarnia@gmail.com
"""

import numpy as np
import os, datetime
import neurokit2 as nk
from pyentrp import entropy as ent
from nilearn.connectome import ConnectivityMeasure
import bct
from scipy.stats import spearmanr
from junifer.storage import HDF5FeatureStorage, SQLiteFeatureStorage

def spear_corr(y_true, y_pred):
    """
    Calculate Spearman correlation coefficient between true and predicted
    regression targets.

    Arguments:
    -----------
    y_true : array_like
        True regression targets.
    y_pred : array_like
        Predicted regression targets.

    Returns:
    --------
    r : float
        Spearman correlation coefficient.
    """
    r = spearmanr(y_true, y_pred)[0]
    return r

def dist_range(x, y):
    """Compute range distance between two vectors.

    Arguments:
    -----------
    x : array_like
        First vector.
    y : array_like
        Second vector.

    Returns:
    --------
    dist : float
        Range distance between x and y.
    """
    dist = (np.max(np.abs(x - y), axis=1) - np.min(np.abs(x - y), axis=1)) \
           / (np.max(np.abs(x - y), axis=1) + np.min(np.abs(x - y), axis=1))
    return dist

def RangeEn_B(x, emb_dim=2, tolerance=0.5, dist=dist_range):
    """
    Compute RangeEn_B complexity measure.

    Arguments:
    -----------
    x : array_like
        Time series.
    emb_dim : int, optional
        Embedding dimension (default is 2).
    tolerance : float, optional
        Tolerance value (default is 0.5).
    dist : function, optional
        Distance function (default is dist_range).

    Returns:
    --------
    RangeEn_B : float
        RangeEn_B complexity measure.
    """
    n = np.shape(x)[0]
    tVecs = np.zeros((n - emb_dim, emb_dim + 1))
    for i in range(tVecs.shape[0]):
        tVecs[i, :] = x[i:i + tVecs.shape[1]]
    counts = []
    for m in [emb_dim, emb_dim + 1]:
        counts.append(0)
        tVecsM = tVecs[:n - m + 1, :m]
        for i in range(len(tVecsM)):
            dsts = dist(tVecsM, tVecsM[i])
            dsts = np.delete(dsts, i, axis=0)
            counts[-1] += np.sum(dsts < tolerance) / (n - m - 1)
    if counts[1] == 0:
        RangeEn_B = np.nan
    else:
        RangeEn_B = -np.log(1.0 * counts[1] / counts[0])
    return RangeEn_B

def feature_extraction_first_level(
    config,
    subj_ID,
    fALFF_db_file,
    fmri_parcellated_filename,
    atlas_name
    ):
    """
    Perform first-level feature extraction from fMRI data.

    Arguments:
    -----------
    config : dict
        Dictionary containing configuration information.
    subj_ID : str
        Subject ID.
    fALFF_db_file : str
        File path to the fALFF database.
    fmri_parcellated_filename : str
        File path to the fmri parcellated data.
    atlas_name : str
        Name of the brain atlas used.

    Returns:
    --------
    Output_flag : str
        Status of feature extraction.
    """
    
    # from ptpython.repl import embed
    # print("OKKKK: feature_extraction_first_level")
    # embed(globals(), locals()) # --> In order to put a break point
    
    # saved features folder    
    npzfiles_folder = config['juseless_paths'][f'npzfiles_folder_{atlas_name}']
    if not os.path.isdir(npzfiles_folder):
        raise ValueError(
            f"** The folder {npzfiles_folder} does not exist!"
        )
    
    output_file = os.path.join(
        npzfiles_folder, 
        f'ID{subj_ID}_{atlas_name}.npz'
    )

    # ----- Define initial parameters for Multiscale entropy analysis
    N_r = config['feature_extraction']['entropy_params']['N_r']
    emb_dim = config['feature_extraction']['entropy_params']['emb_dim']
    r_MSE = config['feature_extraction']['entropy_params']['r_MSE']
    r_span = np.arange(1/N_r, 1 + 1/N_r, 1/N_r)
    N_scale = config['feature_extraction']['entropy_params']['N_scale']
    N_ROI = \
        config[f'feature_extraction'][f'brain_atlas_{atlas_name}']['N_ROI']
        
    # ----- Load the fMRI data
    if(not os.path.isfile(output_file)):

        # ----- Initialize ROI-wise complexity measures
        ShannonEn = np.zeros((N_ROI, 1))
        MSE_fMRI = np.zeros((N_ROI, N_scale))
        MSE_AUC = np.zeros((N_ROI, 1))
        RangeEnB_fMRI = np.zeros((N_ROI, N_r))
        RangeEnB_AUC = np.zeros((N_ROI, 1))
        PE = np.zeros((N_ROI, 1)) # Permutation entropy
        wPE = np.zeros((N_ROI, 1)) # weighted Permutation entropy
        RSA_hurst = np.zeros((N_ROI, 1)) # Hurst exponent through DFA
        fALFF = np.zeros((N_ROI, 1)) # fALFF from Felix's datalad dataset
        LCOR = np.zeros((N_ROI, 1)) # LCOR from Felix's datalad dataset
        GCOR = np.zeros((N_ROI, 1)) # GCOR from Felix's datalad dataset
        
        # -----load parcel time series (size: N_ROI x N_time)
        fmri_parcellated = np.load(fmri_parcellated_filename)
        fmri_parcellated = fmri_parcellated['fmri_parcellated']
        
        if atlas_name == 'glasser':
            fmri_parcellated = fmri_parcellated.T
        
        try:

            # -----------------------------------------------------
            # -- Extract the fALFF/LCOR/GCOR files from the db file
            # ----------------------------------------------------- 
            if atlas_name == 'glasser':
                
                storage = HDF5FeatureStorage(
                    uri=fALFF_db_file, 
                    single_output=True)

                feature_name = 'fALFF_Glasser_Mean'
                df = storage.read_df(feature_name=feature_name)
                fALFF = df.to_numpy()[0]
                del df

                feature_name = 'LCOR_Glasser_Mean'
                df = storage.read_df(feature_name=feature_name)
                LCOR = df.to_numpy()[0]
                del df

                feature_name = 'GCOR_Glasser_Mean'
                df = storage.read_df(feature_name=feature_name)
                GCOR = df.to_numpy()[0]
                del df
                
            elif atlas_name == 'schaefer':

                storage = SQLiteFeatureStorage(
                fALFF_db_file, 
                single_output=True)

                feature_name = f'fALFF_Schaefer{N_ROI}x7_Mean'
                df = storage.read_df(feature_name=feature_name)
                df = df.reset_index() # Make 'subject' a regular column rather than index column
                fALFF = df[df['subject']==subj_ID]
                fALFF = fALFF.to_numpy()
                fALFF = fALFF.T
                fALFF = fALFF[2:]
                del df

                feature_name = f'LCOR_Schaefer{N_ROI}x7_Mean'
                df = storage.read_df(feature_name=feature_name)
                df = df.reset_index() # Make 'subject' a regular column rather than index column
                LCOR = df[df['subject']==subj_ID]
                LCOR = LCOR.to_numpy()
                LCOR = LCOR.T
                LCOR = LCOR[2:]
                del df

                feature_name = f'GCOR_Schaefer{N_ROI}x7_Mean'
                df = storage.read_df(feature_name=feature_name)
                df = df.reset_index() # Make 'subject' a regular column rather than index column
                GCOR = df[df['subject']==subj_ID]
                GCOR = GCOR.to_numpy()
                GCOR = GCOR.T
                GCOR = GCOR[2:]
                del df
                
            else:
                
                raise ValueError('** Unknown atlas name!')

            # -------------------------------------
            # Extract connectivity measures
            # -------------------------------------
            correlation_measure = ConnectivityMeasure(kind='correlation')
            correlation_matrix = correlation_measure.fit_transform(
                [fmri_parcellated.T]
            )[0]
            np.fill_diagonal(correlation_matrix, 0)
            correlation_matrix = np.abs(correlation_matrix)

            # -------------------------------------
            # Extract the ROI-wise graph measures as 
            # brain maps from the FC matrix
            # -------------------------------------
            # Eigenvector centrality brain map
            eig_cent = bct.eigenvector_centrality_und(correlation_matrix)

            # Weighted clustering coefficient maps
            wCC = bct.clustering_coef_wu(correlation_matrix)

            # -------------------------------------
            # Extract ROI-wise complexity
            # -------------------------------------
            for n_roi in range(N_ROI):
                
                # Shannon entropy
                ShannonEn[n_roi] = \
                    ent.shannon_entropy(fmri_parcellated[n_roi,:])

                # MSE and AUC of MSE
                r_corrected = r_MSE*np.std(fmri_parcellated[n_roi,:])
                MSE_fMRI[n_roi, :] = ent.multiscale_entropy(
                    fmri_parcellated[n_roi,:],
                    sample_length=emb_dim,
                    tolerance=r_corrected,
                    maxscale=N_scale)

                MSE_fMRI[n_roi, :] = np.nan_to_num(MSE_fMRI[n_roi, :])
                MSE_AUC[n_roi] = np.trapz(MSE_fMRI[n_roi, :])
                MSE_AUC[n_roi] = MSE_AUC[n_roi]/N_scale

                # Permutation Entropy and weighted Permutation entropy
                # ref: 
                # https://www.frontiersin.org/articles/10.3389/fnagi.2017.00378/full
                PE[n_roi] = ent.permutation_entropy(
                    fmri_parcellated[n_roi,:],
                    order = 4,
                    delay = 1,
                    normalize = True
                )

                wPE[n_roi] = ent.weighted_permutation_entropy(
                    fmri_parcellated[n_roi,:],
                    order = 4,
                    delay = 1,
                    normalize = True
                )

                # Hurst exponent through Rescaled Range (R/S) Analysis
                RSA_hurst[n_roi], _ = \
                    nk.fractal_hurst(fmri_parcellated[n_roi,:])
                    
                # RangeEn and AUC of RangeEn
                for n_r in range(N_r):

                    tolerance = r_span[n_r]

                    RangeEnB_fMRI[n_roi, n_r] = RangeEn_B(
                        fmri_parcellated[n_roi, :], 
                        emb_dim=emb_dim, 
                        tolerance=tolerance, 
                        dist=dist_range
                    )
                
                    RangeEnB_fMRI[n_roi, :] = \
                        np.nan_to_num(RangeEnB_fMRI[n_roi, :])
                    RangeEnB_AUC[n_roi] = np.trapz(RangeEnB_fMRI[n_roi, :])
                    RangeEnB_AUC[n_roi] = RangeEnB_AUC[n_roi]/N_r
                    
                print(f'ROI {n_roi+1}/{N_ROI} done!')

            # -----------------------------------------------------
            # -- Compute ROI-wise temporal SNR (mean/std)
            # ----------------------------------------------------- 
            tSNR = np.zeros((N_ROI,1))
            for n_roi in range(N_ROI):
                m = np.mean(fmri_parcellated[n_roi, :])
                std = np.std(fmri_parcellated[n_roi, :])
                tSNR[n_roi] = m/std

            # -----------------------------------------------------
            # Save the subject-specific entropy/connectivity results
            # -----------------------------------------------------
            # print('OKKKK: Spatiotemporal complexity')
            # embed(globals(), locals()) 
            Output_flag = 'Success'
            np.savez(
                output_file, 
                fmri_parcellated=fmri_parcellated,
                tSNR_fMRI=tSNR,
                N_r=N_r,
                eig_cent=eig_cent,
                wCC=wCC,
                ShannonEn_fMRI=ShannonEn,
                MSE_fMRI=MSE_fMRI,
                MSE_AUC_fMRI=MSE_AUC,
                r_MSE=r_MSE,
                PE_fMRI=PE,
                wPE_fMRI=wPE,
                RSA_hurst_fMRI=RSA_hurst,
                r_span=r_span, 
                emb_dim=emb_dim,
                RangeEnB_fMRI=RangeEnB_fMRI,
                RangeEnB_AUC_fMRI=RangeEnB_AUC,
                fALFF_fMRI=fALFF,
                LCOR_fMRI=LCOR,
                GCOR_fMRI=GCOR,
                Output_flag=Output_flag)
            
            print(f"The output file {output_file} was saved successfully!") 

        except Exception as e:

            Output_flag = 'Fail'
            if(os.path.isfile(output_file)):
                os.remove(output_file)
            np.savez(
                output_file,
                Output_flag=Output_flag)
        
            print(f"The output file {output_file} was failed! {e}")          

    else:

        out = np.load(output_file, allow_pickle=True)
        Output_flag = out['Output_flag']
        print(f"The output file {output_file} already exists!")

    return Output_flag
