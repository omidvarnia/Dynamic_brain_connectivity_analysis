"""
This script provides functions for neuroimaging data processing and analysis.

Written by: Amir Omidvarnia
Contact: amir.omidvarnia@gmail.com
"""

import nibabel as nib
import numpy as np

def vec2image(
brain_maps_2_nii, 
atlas_filename,
output_filename
):
    """Converts ROI-wise brain maps to a 3D NIfTI image.

    Arguments:
    -----------
    brain_maps_2_nii: array of size N_ROI x N_measure
        2D array containing brain maps for multiple graph measures.
    atlas_filename: str or path
        Path to a parcellated brain atlas with N_ROI regions.
    output_filename: str or path
        Filename of the output file.

    Returns:
    --------
    brain_surf_map: str or path
        Filename of the converted 3D brain surface NIfTI file.
    """
    N_ROI, N_measure = brain_maps_2_nii.shape

    # Read the brain atlas image
    atlas_img = nib.load(atlas_filename)
    affine_mat = atlas_img.affine
    atlas_img = atlas_img.get_fdata()
    N_x, N_y, N_z = np.shape(atlas_img)
    
    unique_labels = np.unique(atlas_img.flatten())
    unique_labels = unique_labels[unique_labels != 0]
    
    brain_maps_img = np.zeros((N_x, N_y, N_z, N_measure))

    for n_roi in range(N_ROI):

        ind = np.where(atlas_img==unique_labels[n_roi])
        ix = ind[0]
        iy = ind[1]
        iz = ind[2]

        for n_meas in range(N_measure):
            brain_maps_img[ix, iy, iz, n_meas] = brain_maps_2_nii[n_roi, n_meas]

    brain_map_nii = nib.Nifti1Image(brain_maps_img, affine=affine_mat)
    brain_map_nii.to_filename(output_filename)

    return atlas_filename
