import os
import tarfile
import shutil
import argparse


def compress_subdirectory(subroot_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    
    for subdir in os.listdir(subroot_dir):
        subdir_path = os.path.join(subroot_dir, subdir)
        
        if os.path.isdir(subdir_path):
            # create the tar file path in the destination directory
            tar_file_name = os.path.join(dest_dir, f"{subdir}.tar")
            with tarfile.open(tar_file_name, "w") as tar:
                tar.add(subdir_path, arcname=subdir)
            print(f"Compressed {subdir_path} into {tar_file_name}")
            
            # remove the original subdirectory after compressing (optional)
            # shutil.rmtree(subdir_path)
            # print(f"Removed original directory {subdir_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default="../resources/example_file_structures/tvad_raw/", type=str)
    parser.add_argument('--save_dir', default="../resources/example_file_structures/tvad/", type=str)
    args = parser.parse_args()

    for subroot_name in os.listdir(args.root_dir):
        compress_subdirectory(os.path.join(args.root_dir, subroot_name), os.path.join(args.save_dir, subroot_name))
