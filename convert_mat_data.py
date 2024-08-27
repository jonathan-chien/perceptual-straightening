"""
Script to convert MATLAB files to PyTorch pt tensor files.
"""

import os
import re
import scipy
import subprocess
import torch


def convert_mat_data(dir='data/mat'):
    """
    This function takes as argument a path name and descends recursively until 
    it finds .mat data files, whose contents (`response_mat`/`responseMatrix`) 
    are converted to PyTorch tensors and saved in a directory structure matching 
    that of the original.

    Arguments
    ---------
    dir : (str | Default = 'data/mat') A path name. For top level usage, provide 
          path to directory that contains data or subdirectories with data.

    Author: Jonathan Chien. 2024/02. Last update 2024/03/25.
    """

    for file in os.listdir(dir): 
        # Build current item name inside current directory.
        dir_file = dir + '/' + file

        # If current item is a directory, recursively call function till file is found.
        if file != "." and file != ".." and os.path.isdir(dir_file):
            print(f"Going into {dir_file} ...")   
            convert_mat_data(dir_file)

        # Once/if file is found, convert response matrix to tensor and save.
        elif re.search(r'/.+_.*\.mat', dir_file) and not re.search(r'/\._', dir_file):
            print(f"Converting {dir_file} ...")
            
            # Rename for clarity.
            mat_file = dir_file
            pt_file = re.sub('mat', 'pt', mat_file)

            if re.search('corey', pt_file):
                key = 'responseMatrix'
            else: 
                key = 'response_matrix'

            data = scipy.io.loadmat(mat_file)
            response_mat = torch.from_numpy(data[key])

            # Iff directory does not already exist, create it. 
            subprocess.run(["mkdir", "-p", os.path.dirname(pt_file)])

            torch.save(response_mat, pt_file)


# --------------------------------------------------
if __name__ == "__main__":
    convert_mat_data()
    print("Done.")
