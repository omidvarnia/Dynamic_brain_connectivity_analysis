"""
The script generates .submit files for the complexity analysis of the UKB
dataset. It is based on the HTCondorSubmissionTemplate class in
the 'common' folder.

The script can be run from the command line as follows:
>> python3 Step2_create_job_files_for_HTCondor.py

Written by: Amir Omidvarnia
Email: amir.omidvarnia@gmail.com
"""
class HTCondorSubmissionTemplate:
    """A class to provide a template/convenience functions for interfacing
    with HTCondor on Juseless.
    
    Parameters
    ----------
    subjects_list : list
        List of UKB subject IDs to be included in the htcondor file.
    initial_dir : str or path
        Initial directory.
    executable_file : str
        Name of the file to be executed by condor (including file extension).
    target_names : list, optional
        List of target names, by default [].
    target_labels : list, optional
        List of target labels, by default [].
    HPC_name : str, optional
        Name of the HPC, by default 'juseless'.
    mail_user : str, optional
        An email address for receiving different notices from juseless,
        by default 'a.omidvarnia@fz-juelich.de'.
    logs_folder : str or path, optional
        A folder name where the htcondor log files will be saved,
        by default " ".
    request_cpus : int, optional
        Number of requested CPUs, by default 1.
    request_memory : int, optional
        Number of CPUs per task, by default 20.
    request_disk : int, optional
        Requested disk space in GB, by default 100.
    submit_file : str or path, optional
        Name of the submit file (including directory) to generate for Juseless,
        e.g., "submit_file.submit", by default " ".
    args : list, optional
        Additional arguments, by default [].
    """

    def __init__(
        self,
        subjects_list,
        initial_dir,
        executable_file,
        target_names=[],
        target_labels=[],
        HPC_name='juseless',
        mail_user='a.omidvarnia@fz-juelich.de',
        logs_folder=" ",
        request_cpus=1,
        request_memory=20,
        request_disk=100,
        submit_file=" ",
        args=[],
    ):

        self.HPC_name = HPC_name
        self.subjects_list = subjects_list
        self.executable_file = executable_file
        self.target_names = target_names
        self.target_labels = target_labels
        self.mail_user = mail_user
        self.submit_file = submit_file
        self.request_cpus = request_cpus
        self.request_memory = request_memory
        self.request_disk = request_disk
        self.initial_dir = initial_dir
        self.logs_folder = logs_folder
        self.args=args
        self.subjects_list=subjects_list

    def make_htcondor_general_options(self):

        """
        Generates a multi-line string of general options for the htcondor
        file.
        """


        general_options = (

f"""#!/bin/bash

# The environment
universe = vanilla
getenv = True

# Resources
request_cpus = {self.request_cpus}
request_memory = {self.request_memory}G
request_disk = {self.request_disk}G


# Execution
initial_dir          = {self.initial_dir}
executable           = {self.executable_file}
transfer_executable  = False

# E-mail option
Notify_user          = {self.mail_user}
Notification         = never

"""
        )

        return general_options

    def write_queue_for_feature_extraction(self):

        """
        Adds the subject-specific srun lines to the htcondor file with 
        already added general options.
        """

        general_options = self.make_htcondor_general_options()
        N_subj = len(self.subjects_list)

        with open(self.submit_file, 'w') as f:
            f.write(general_options)
            
            for n_subj in range(N_subj):

                subj_ID = self.subjects_list[n_subj]

                executable_cmd = (
                    f"# Subject no {n_subj + 1} out of {N_subj}: {subj_ID}\n"
                    f"log    = {self.logs_folder}/$(ClusterId)_S{subj_ID}.log\n"
                    f"output = {self.logs_folder}/$(ClusterId)_S{subj_ID}.output\n"
                    f"error  = {self.logs_folder}/$(ClusterId)_S{subj_ID}.err\n"
                )
                f.write(executable_cmd)

                cmd = f"arguments = {subj_ID}"
                cmd = f"{cmd}\nqueue\n\n"

                f.write(cmd)

        f.close()

    def write_queue_for_prediction(self):

        """
        Adds the subject-specific srun lines to the htcondor file with 
        already added general options.
        """

        general_options = self.make_htcondor_general_options()

        with open(self.submit_file, 'w') as f:
            f.write(general_options)

            N_targets = len(self.target_names)
            for n_target in range(N_targets):

                target_name = self.target_names[n_target]
                target_label = self.target_labels[n_target]

                executable_cmd = (
                    f"# Target {target_name}: ({target_label})\n"
                    f"log    = {self.logs_folder}/$(ClusterId)_{target_name}.log\n"
                    f"output = {self.logs_folder}/$(ClusterId)_{target_name}.output\n"
                    f"error  = {self.logs_folder}/$(ClusterId)_{target_name}.err\n"
                )
                f.write(executable_cmd)

                cmd = self.args[0]
                cmd = f"arguments = --model_type {cmd}"
                cmd = f"{cmd} --target {target_name}"
                cmd = f"{cmd} --feature_name {self.args[1]}"
                cmd = f"{cmd} --N_subj_all {self.args[2]}"
                
                try:
                    cmd = f"{cmd} --confound_removal {self.args[3]}"
                except:
                    pass
                
                cmd = f"{cmd}\nqueue\n\n"

                f.write(cmd)

        f.close()
        
    def write_queue_for_prediction_ConfOnly(self):
        
        """
        Adds the subject-specific srun lines to the htcondor file with 
        already added general options.
        """

        general_options = self.make_htcondor_general_options()

        with open(self.submit_file, 'w') as f:
            f.write(general_options)

            N_targets = len(self.target_names)
            for n_target in range(N_targets):

                target_name = self.target_names[n_target]
                target_label = self.target_labels[n_target]

                executable_cmd = (
                    f"# Target {target_name}: ({target_label})\n"
                    f"log    = {self.logs_folder}/$(ClusterId)_{target_name}.log\n"
                    f"output = {self.logs_folder}/$(ClusterId)_{target_name}.output\n"
                    f"error  = {self.logs_folder}/$(ClusterId)_{target_name}.err\n"
                )
                f.write(executable_cmd)

                cmd = self.args[0]
                cmd = f"arguments = --model_type {cmd}"
                cmd = f"{cmd} --target {target_name}"
                cmd = f"{cmd} --N_subj_all {self.args[1]}"
                
                try:
                    cmd = f"{cmd} --confound_removal {self.args[2]}"
                except:
                    pass
                
                cmd = f"{cmd}\nqueue\n\n"

                f.write(cmd)

        f.close()
        
    def write_queue_for_fingerptinting(self):
       
        """
        Adds the subject-specific srun lines to the htcondor file with 
        already added general options.
        """

        general_options = self.make_htcondor_general_options()
        
        model_type = self.args[0]
        N_subj_all = self.args[1]
        confound_removal = self.args[2]

        with open(self.submit_file, 'w') as f:
            
            f.write(general_options)

            executable_cmd = (
                f"# Fingerprinting analysis\n"
                f"log    = {self.logs_folder}/$(ClusterId)_fingerprinting_N{N_subj_all}.log\n"
                f"output = {self.logs_folder}/$(ClusterId)_fingerprinting_N{N_subj_all}.output\n"
                f"error  = {self.logs_folder}/$(ClusterId)_fingerprinting_N{N_subj_all}.err\n"
            )
            f.write(executable_cmd)
            
            cmd = f"arguments = --model_type {model_type}"
            cmd = f"{cmd} --N_subj_all {N_subj_all}"
            cmd = f"{cmd} --confound_removal {confound_removal}"

            cmd = f"{cmd}\nqueue\n\n"

            f.write(cmd)

        f.close()
        