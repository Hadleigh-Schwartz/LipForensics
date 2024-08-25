import cv2
import face_alignment
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import time
import sys
import pickle
from colorama import Fore, Style

# initialize face detector and landmark detector
if sys.platform == "linux":
    device = "cuda"
else:
    device = "mps"
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device = device, face_detector='blazeface')

def prep_video(video_path, vid_num, lip_forensics_absolute_path, folder_name, visualize_landmarks = False):
    """
    Parameters:
    video_path: str
        path to video
    vid_num: int
        Unique integer identifer for the video within the dataset that will be written to LipForensics/data/datasets/CelebDF/FakeCelebDF
        or LipForensics/data/datasets/CelebDF/RealCelebDF
    lip_forensics_absolute_path: str
        Absolute path to LipForensics repo
    folder_name: str
        Images and landmarks will be saved to LipForensics/data/datasets/CelebDF/{folder_name}
    visualize_landmarks: bool
        Whether to display visualization of the landmarks on the face frame
    
    Returns:
    None. Saves face frames and landmarks to LipForensics/data/datasets/CelebDF/{folder_name} in
    the requirement format/naming convention for CelebDF, as specified in the LipForensics README.md. We arbitrarily choose CelebDF as the dataset
    we are imitating in order to get results for our own dataset.
    """
    if not os.path.exists(video_path):
        print(f"Video {video_path} does not exist")
        return

    print(f"Prepping {video_path}")
    start = time.time()
   
    vid_num = str(vid_num).zfill(4)
    frames_output_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/{folder_name}/images/{vid_num}/"
    landmarks_output_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/{folder_name}/landmarks/{vid_num}/"
  
    if not os.path.exists(frames_output_path):
        os.makedirs(frames_output_path)
    else:
        raise Exception(f"Path {frames_output_path} already exists")

    if not os.path.exists(landmarks_output_path):
        os.makedirs(landmarks_output_path)
    else:
        raise Exception(f"Path {landmarks_output_path} already exists")
    
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # get landmarks and face bbox
        input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lms, _, face_bbox = fa.get_landmarks(input, return_bboxes=True)
        if face_bbox is None or lms is None:
            frame_num += 1
            continue
        lms = lms[0]
        face_bbox = face_bbox[0]
        face_frame = frame[int(face_bbox[1]):int(face_bbox[3]), int(face_bbox[0]):int(face_bbox[2])]
        if np.sum(face_frame) == 0:
            frame_num += 1
            continue

        # save face frame
        cv2.imwrite(frames_output_path + f"{frame_num}.png", face_frame)
        
        # format landmarks as (68, 2) and save
        lms = lms.reshape(68, 2)
        lms[:, 0] -= face_bbox[0]
        lms[:, 1] -= face_bbox[1]
        np.save(landmarks_output_path + f"{frame_num}.npy", lms)

        if visualize_landmarks:
            plt.imshow(face_frame)
            plt.scatter(lms[:,0], lms[:,1], 2)
            plt.show()

        frame_num += 1
    cap.release()
    end = time.time()
    print(f"Time taken to prep {frame_num} frames of {video_path}: {end - start}")


def aggregate_models(lip_forensics_absolute_path):
    """
    Take face frames, landmarks, and cropped mouths of each deepfake model and combine into one aggregated deepfake folder at FakeCelebDF.

    Parameters:
    lip_forensics_absolute_path: str
        Absolute path to LipForensics repo
    
    Returns:
    None. Saves aggregated face frames, landmarks, and cropped mouths to LipForensics/data/datasets/CelebDF/FakeCelebDF
    """
    frames_output_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/FakeCelebDF/images/"
    landmarks_output_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/FakeCelebDF/landmarks/"
    cropped_mouths_output_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/FakeCelebDF/cropped_mouths/"
    # if not os.path.exists(landmarks_output_path):
    #     os.makedirs(landmarks_output_path)
    # else:
    #     raise Exception(f"Path {landmarks_output_path} already exists")
    # if not os.path.exists(cropped_mouths_output_path):
    #     os.makedirs(cropped_mouths_output_path)
    # else:
    #     raise Exception(f"Path {cropped_mouths_output_path} already exists")
    # if not os.path.exists(frames_output_path):
    #     os.makedirs(frames_output_path)
    # else:   
    #     raise Exception(f"Path {frames_output_path} already exists")
    
    models = ["dagan", "faceswap", "first", "sadtalker", "talklip"]
    overall_vid_num = 0
    for model in models:
        model_landmarks_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/{model}/landmarks/"
        model_cropped_mouths_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/{model}/cropped_mouths/"
        model_frames_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/{model}/images/"
        model_landmark_folders = glob.glob(model_landmarks_path + "*")
        model_cropped_mouth_folders = glob.glob(model_cropped_mouths_path + "*")
        model_frame_folders = glob.glob(model_frames_path + "*")
        # sort folders by number
        model_landmark_folders.sort(key = lambda x: int(x.split("/")[-1]))
        model_cropped_mouth_folders.sort(key = lambda x: int(x.split("/")[-1]))
        model_frame_folders.sort(key = lambda x: int(x.split("/")[-1]))
        for i in range(len(model_landmark_folders)):
            if os.path.exists(f"{cropped_mouths_output_path}{str(overall_vid_num).zfill(4)}"):
                overall_vid_num += 1
                continue
            print(model_landmark_folders[i], model_cropped_mouth_folders[i], model_frame_folders[i], f"cp -r {model_landmark_folders[i]} {landmarks_output_path}{str(overall_vid_num).zfill(4)}")
            os.system(f"cp -r {model_landmark_folders[i]} {landmarks_output_path}{str(overall_vid_num).zfill(4)}")
            os.system(f"cp -r {model_cropped_mouth_folders[i]} {cropped_mouths_output_path}{str(overall_vid_num).zfill(4)}")
            os.system(f"cp -r {model_frame_folders[i]} {frames_output_path}{str(overall_vid_num).zfill(4)}")
            overall_vid_num += 1

def remove_model_from_aggregate(lip_forensics_absolute_path, model_to_remove):
    """
    Assumes you have just run aggregate_models and want to remove the folders corresponding to a certain df model from the aggregated folder metdata. 

    Parameters:
    lip_forensics_absolute_path: str
        Absolute path to LipForensics repo
    model_to_remove: str
        Name of the deepfake model to remove from the aggregated folder. Must be one of "dagan", "faceswap", "first", "sadtalker", "talklip"
    
    Returns:
    None. Removes the folders corresponding to the deepfake model from the aggregated folder metadata
    """
    frames_output_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/FakeCelebDF/images/"
    landmarks_output_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/FakeCelebDF/landmarks/"
    cropped_mouths_output_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/FakeCelebDF/cropped_mouths/"

    models = ["dagan", "faceswap", "first", "sadtalker", "talklip"]
    overall_vid_num = 0
    for model in models:
        model_landmarks_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/{model}/landmarks/"
        model_cropped_mouths_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/{model}/cropped_mouths/"
        model_frames_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/{model}/images/"
        model_landmark_folders = glob.glob(model_landmarks_path + "*")
        model_cropped_mouth_folders = glob.glob(model_cropped_mouths_path + "*")
        model_frame_folders = glob.glob(model_frames_path + "*")
        # sort folders by number
        model_landmark_folders.sort(key = lambda x: int(x.split("/")[-1]))
        model_cropped_mouth_folders.sort(key = lambda x: int(x.split("/")[-1]))
        model_frame_folders.sort(key = lambda x: int(x.split("/")[-1]))
        for i in range(len(model_landmark_folders)):
            if model == model_to_remove:
                print(f"Removing {landmarks_output_path}{str(overall_vid_num).zfill(4)}, {cropped_mouths_output_path}{str(overall_vid_num).zfill(4)}, {frames_output_path}{str(overall_vid_num).zfill(4)}")
                os.system(f"rm -r {landmarks_output_path}{str(overall_vid_num).zfill(4)}")
                os.system(f"rm -r {cropped_mouths_output_path}{str(overall_vid_num).zfill(4)}")
                os.system(f"rm -r {frames_output_path}{str(overall_vid_num).zfill(4)}")
            overall_vid_num += 1


def prep_e2e_fake_videos():
    participants = ["yuval", "colman", "toma", "phyllis", "hadleigh", "xiaofeng", "keylen", "rundi",  "tao", "dan", "qijia", "charlie", "ruoshi", "saeyoung", "kathryn", "naz", "honglin", "lisa", "abhinav",  "kahlil"]
    devices = ["googlepixel", "webcam", "canon", "iphone"]
    paragraph_nums = [1, 2, 3, 4, 5, 6]
    model = "faceswap"


    vid_file = open("faceswap_video_mapping.csv", "w")
    vid_file.write("Real/Fake,ID,path\n")
    df_vid_count = 0
    for participant in participants:
        for device in devices:
            for p in paragraph_nums:
                vid_path = f"/media/admin/E380-1E91/Deepfake/End_To_End/deepfakes_may24/{participant}/{model}/{device}/p{p}_df.mp4"
                if not os.path.exists(vid_path):
                    continue
                prep_video(vid_path, df_vid_count, "/home/hadleigh/deepfake_detection/system/evaluation/passive_detection/LipForensics", fake = True)
                vid_file.write(f"Fake,{df_vid_count},{vid_path}\n")
                df_vid_count += 1
    vid_file.close()
 

def generate_dyn_micro_fake_clips():

    with open("/home/xiaofeng/deepfake_detection/system/evaluation/dynamic_features/df_ratio_dict.pkl", "rb") as f:
        df_ratio_dict = pickle.load(f)

    # loop through all paricipants, paragraphs, mod percentages for 4.5s win 
    mod_percs = ["25", "50", "75"]
    paragraph_nums = [0, 1, 2, 3]
    participants = ["charlie", "colman", "hadleigh", "xiaofeng", "lisa", "qi", "qijia", "clementine"]
    bin_vals = {"0_10": [], "10_20": [], "20_30": [], "30_40": [], "40_50": [], "50_60": [], "60_70": [], "70_80": [], "80_90": [], "90_100": []}
    bin_targets = {"10_20": [], "20_30": [], "30_40": [], "40_50": []}
    for participant in participants:
        for m in mod_percs:
            for paragraph in paragraph_nums:
                ratios = df_ratio_dict[participant][m][paragraph][4.5]
                print(participant, paragraph, m, ratios)
                for win_num, r in enumerate(ratios):
                    if r < 0.1:
                        bin_vals["0_10"].append(r)
                    elif r > 0.1 and r < 0.2:
                        bin_vals["10_20"].append(r)
                        bin_targets["10_20"].append([participant, paragraph, m, win_num])
                    elif r > 0.2 and r < 0.3:
                        bin_vals["20_30"].append(r)
                        bin_targets["20_30"].append([participant, paragraph, m, win_num ])
                    elif r > 0.3 and r < 0.4:
                        bin_vals["30_40"].append(r)
                        bin_targets["30_40"].append([participant, paragraph, m, win_num])
                    elif r > 0.4 and r < 0.5:
                        bin_vals["40_50"].append(r)
                        bin_targets["40_50"].append([participant, paragraph, m, win_num])
                    elif r > 0.5 and r < 0.6:
                        bin_vals["50_60"].append(r)
                    elif r > 0.6 and r < 0.7:
                        bin_vals["60_70"].append(r)
                    elif r > 0.7 and r < 0.8:
                        bin_vals["70_80"].append(r)
                    elif r > 0.8 and r < 0.9:
                        bin_vals["80_90"].append(r)
                    elif r > 0.9:
                        bin_vals["90_100"].append(r)


    for mod_group, targets in bin_targets.items():
        for t in targets:
            participant, paragraph, m, win_num = t
            print(Fore.MAGENTA  + f"{mod_group} " + Fore.BLUE + f"Participant: {participant}, Paragraph: {paragraph}, Mod: {m}, Win: {win_num}" + Style.RESET_ALL)
            for model in ["dagan", "first", "sadtalker", "talklip"]:
                output_folder_path = f"/home/hadleigh/deepfake_data/mod_clips/{mod_group}/{model}"
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)
                for cam in ["45_close", "45_far", "front_far", "60_close", "60_far", "front_close"]:
                    vid_path = f"/media/admin/E380-1E91/Deepfake/Dynamic_Micros/df_paragraphs/{participant}/{model}/{m}/{cam}/{participant}_df_p{paragraph}.mp4"
                    output_path = f"{output_folder_path}/{participant}_{m}_{cam}_{paragraph}_{win_num}.mp4"
                    print(f"Processing {vid_path} to {output_path}")
                    os.system(f"ffmpeg -loglevel error -i {vid_path} -ss {win_num * 4.5} -t 4.5 -c copy {output_path}")

def generate_dyn_micro_real_clips():
    participants = ["charlie", "colman", "hadleigh", "xiaofeng", "lisa", "qi", "qijia", "clementine"]
    paragraph_nums = [0, 1, 2, 3]
    for participant in participants:
        for paragraph in paragraph_nums:
            for cam in ["45_close", "45_far", "front_far", "60_close", "60_far", "front_close"]:
                vid_path = f"/media/admin/E380-1E91/Deepfake/Dynamic_Micros/og_paragraphs/{participant}/{cam}/{participant}_og_p{paragraph}.mp4"
                # get duration of video
                cap = cv2.VideoCapture(vid_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                num_wins = int(duration / 4.5)
                for n in range(num_wins):
                    output_path = f"/home/hadleigh/deepfake_data/real_clips/{participant}_{paragraph}_{cam}_{n}.mp4"
                    print(f"Processing {vid_path} to {output_path}")
                    os.system(f"ffmpeg -loglevel error -i {vid_path} -ss {n * 4.5} -t 4.5 -c copy {output_path}")
               
def prep_fake_dyn_clips(mod_groups):
    models = ["dagan", "first", "sadtalker", "talklip"]
    for mod_group in mod_groups:
        for model in models:
            vid_file = open(f"{mod_group}_{model}_video_mapping.csv", "w")
            vid_file.write("Real/Fake,ID,path\n")
            video_paths = glob.glob(f"/home/hadleigh/deepfake_data/mod_clips/{mod_group}/{model}/*")
            vid_num = 0
            for vid_path in video_paths:
                prep_video(vid_path, vid_num, "/home/hadleigh/deepfake_detection/system/evaluation/passive_detection/LipForensics", f"{mod_group}_{model}")
                vid_file.write(f"Fake,{vid_num},{vid_path}\n")
                vid_num += 1
            vid_file.close()


def prep_real_dyn_clips():
    vid_file = open("real_dyn_video_mapping.csv", "w")
    vid_file.write("Real/Fake,ID,path\n")
    video_paths = glob.glob("/home/hadleigh/deepfake_data/real_clips/*")
    vid_num = 0
    for vid_path in video_paths:
        prep_video(vid_path, vid_num, "/home/hadleigh/deepfake_detection/system/evaluation/passive_detection/LipForensics", "real_dyn")
        vid_file.write(f"Real,{vid_num},{vid_path}\n")
        vid_num += 1
