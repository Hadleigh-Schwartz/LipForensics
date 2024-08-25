"""
Ensure all output from prep_vids_lf.py is valid before running LipForensics 
preprocessing and evaluation
"""

import glob
import pandas as pd
import os


def exceeds_hops(nums, max_hop):
    for i in range(len(nums) - 1):
        n1 = int(nums[i].split("/")[-1].split(".")[0])
        n2 = int(nums[i + 1].split("/")[-1].split(".")[0])
        if n2 - n1 > max_hop:
            return True
    return False

def check_lf_input(folder_name):
    """
    Ensure all images and landmarks folders for deepfaked videos contain sufficient, validly ordered frames
    """

    images_path = f"data/datasets/CelebDF/{folder_name}/images"
    landmarks_path = f"data/datasets/CelebDF/{folder_name}/landmarks"
    mapping_path = f"{folder_name}_video_mapping.csv"
    failures_path = f"{folder_name}_lip_forensics_failures.csv"
    failures_file = open(failures_path, "a")

    df = pd.read_csv(mapping_path)
    dirs = glob.glob(images_path + "/*")
    empty = 0
    failed = 0
    max_hop = 5
    tot_checked = 0
    for im_dir in dirs:
        ims = glob.glob(im_dir + "/*")
        ims = sorted(ims, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        too_short = len(ims) < 30
        max_hop_exceeded =  exceeds_hops(ims, max_hop)
        if len(ims) == 0 or too_short or max_hop_exceeded:
            vid_num = int(im_dir.split("/")[-1])
            path = df.loc[df['ID'] == vid_num]["path"].values[0]

            print(im_dir, path)
            if os.path.exists(path):
                failed += 1
                reason = ""
                if max_hop_exceeded:
                    reason += "Max hop exceeded. "
                if too_short:
                    reason += "Too short. "
                if len(ims) == 0:
                    reason += "Empty. "
                    empty += 1
                print("Failure: ", path, " ", vid_num, " ", reason)
                failures_file.write(f"{path},{reason}\n")
            
            os.system(f"rm -r {im_dir}")
            zfill_num = str(vid_num).zfill(4)
            # print("landmarks at ", f"{landmarks_path}/{zfill_num}")
            os.system(f"rm -r {landmarks_path}/{zfill_num}")
        tot_checked += 1
    failures_file.close()
    print("-------------------")
    print(f"Empty: {empty}/{tot_checked}")
    print(f"Failed: {failed}/{tot_checked}")

def check_og():
    """
    Ensure all images folders for original videos contain sufficient, validly ordered frames
    """
    images_path = f"LipForensics/data/datasets/CelebDF/RealCelebDF/images"
    df = pd.read_csv("og_video_mapping.csv")
    og_failures_file = open("og_lip_forensics_failures.csv", "a")
    max_hop = 5
    dirs = glob.glob(images_path + "/*")
    failures = 0
    for im_dir in dirs:
        ims = glob.glob(im_dir + "/*")
        ims = sorted(ims, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        too_short = len(ims) < 30
        max_hop_exceeded =  exceeds_hops(ims, max_hop)
        if len(ims) == 0 or too_short or max_hop_exceeded:
            vid_num = int(im_dir.split("/")[-1])
            target_row = df.loc[df['ID'] == vid_num]["path"].values
            if len(target_row) == 0:
                print(vid_num)
            else:
                path = target_row[0]
            failures += 1
            reason = ""
            if max_hop_exceeded:
                reason += "Max hop exceeded. "
            if too_short:
                reason += "Too short. "
            if len(ims) == 0:
                reason += "Empty. "
            print(im_dir, reason)
            og_failures_file.write(f"{path},{reason}\n")
            os.system(f"rm -r -f {im_dir}")
    og_failures_file.close()
    print("OG failures: ", failures)

def remove_landmarks(folder_name):
    """
    Remove any landmark files that dont have images folder with same index
    """
    landmarks_path = f"data/datasets/CelebDF/{folder_name}/landmarks"
    landmarks_files = glob.glob(landmarks_path + "/*")
    removed_count = 0
    for lm_file in landmarks_files:
        vid_num = lm_file.split("/")[-1].split(".")[0]
        images_path = f"data/datasets/CelebDF/{folder_name}/images/{vid_num}"
        if not os.path.exists(images_path):
            removed_count += 1
            os.system(f"rm -r -f {lm_file}")
            print("Removed: ", lm_file)


    print("Removed: ", removed_count)

check_lf_input("real_dyn")