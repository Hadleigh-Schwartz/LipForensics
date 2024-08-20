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

def check_fake():
    """
    Ensure all images folders for deepfaked videos contain sufficient, validly ordered frames
    """

    model = "dagan"
    images_path = f"LipForensics/data/datasets/CelebDF/{model}/images"
    mapping_path = f"{model}_video_mapping.csv"
    failures_path = f"{model}_lip_forensics_failures.csv"
    failures_file = open(failures_path, "a")

    df = pd.read_csv(mapping_path)
    dirs = glob.glob(images_path + "/*")
    empty = 0
    failed = 0
    max_hop = 5
    for im_dir in dirs:
        ims = glob.glob(im_dir + "/*")
        ims = sorted(ims, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        too_short = len(ims) < 30
        max_hop_exceeded =  exceeds_hops(ims, max_hop)
        if len(ims) == 0 or too_short or max_hop_exceeded:
            vid_num = int(im_dir.split("/")[-1])
            if model == "dagan":
                hadleigh_mac_path = df.loc[df['ID'] == vid_num]["path"].values[0]
                path = "/media/admin/E380-1E91/Deepfake/End_To_End/deepfakes_may24/" + hadleigh_mac_path.split("df_paragraphs_may27")[-1]
            else:
                path = df.loc[df['ID'] == vid_num]["path"].values[0].replace("raunak", "admin")

            print(im_dir, path)
            if os.path.exists(path):
                failed += 1
                print("Failure: ", path)
                reason = ""
                if max_hop_exceeded:
                    reason += "Max hop exceeded. "
                if too_short:
                    reason += "Too short. "
                if len(ims) == 0:
                    reason += "Empty. "
                    empty += 1
                print(reason)
                failures_file.write(f"{path},{reason}\n")
            
          
            os.system(f"rm -r {im_dir}")
    print("-------------------")
    print("Empty: ", empty)
    print("Failures: ", failed)


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

def remove_landmarks():
    """
    Remove any landmark files that dont have images folder with same index
    """
    model = "RealCelebDF"
    landmarks_path = f"LipForensics/data/datasets/CelebDF/{model}/landmarks"
    landmarks_files = glob.glob(landmarks_path + "/*")
    removed_count = 0
    for lm_file in landmarks_files:
        vid_num = lm_file.split("/")[-1].split(".")[0]
        images_path = f"LipForensics/data/datasets/CelebDF/{model}/images/{vid_num}"
        if not os.path.exists(images_path):
            removed_count += 1
            # os.system(f"rm -r -f {lm_file}")
            print("Removed: ", lm_file)


    print("Removed: ", removed_count)

remove_landmarks()

