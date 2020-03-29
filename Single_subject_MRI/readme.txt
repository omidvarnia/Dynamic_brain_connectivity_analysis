Statistical Inference of Individual-Level MRI data (MATLAB program - Mangor Pedersen, University of Melbourne 2020)

-----------

Add all files to your MATLAB path, and type "subject_level_MRI_stats" to start.

Below is a description of each command button in the program:

Button 1. The is presented with an option to calculate local connectivity (ReHo). If this option is selected then choose a filtered 4D fMRI .nii file from a single patient with epilepsy. If this option is not chosen, then you can choose a 3D file of your own choice (this can be a functional or structural MRI file with your own chosen metric).

Button 2. The user is again presented with an option to calculate local connectivity (ReHo). If this option is chosen, then you need to select a folder containing all your control MRI .nii data (local connectivity = filtered 4D data; own metric = 3D data).

Button 3. Select a binary .nii mask for statistical evaluation

Button 4. Smooth all input data (6-8 mm is often a good option, however, this depends on voxel-size).

Button 5. Voxel-wise z-score between the single subject (button 1) and the control group (button 2). Here, the user can employ parallel computing to speed up the process, and add a patient identifier to the output files.

Button 6. Random Field Theory thresholding of voxel-wise z-score .nii file.

Button 7. False Discovery Rate thresholding of voxel-wise z-score .nii file