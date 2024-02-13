#!/home/aomidvarnia/miniconda3/envs/fMRI_complexity/bin/python

SUBMIT_FILES="/data/project/SPP2041/results/aomidvarnia/UKB_analysis_manuscript/prediction_glasser_submit_files/ConfOnly_*"

for submit_file in $SUBMIT_FILES
do

  # cat "$f"
  condor_submit $submit_file
  echo "$submit_file was submitted successfully!"

done