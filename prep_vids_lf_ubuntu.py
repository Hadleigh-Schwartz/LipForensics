import cv2
import face_alignment
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import time
import sys

# initialize face detector and landmark detector
if sys.platform == "linux":
    device = "cuda"
else:
    device = "mps"
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device = device, face_detector='blazeface')

def prep_video(video_path, vid_num, lip_forensics_absolute_path, fake = False, visualize_landmarks = False):
    """
    Parameters:
    video_path: str
        path to video
    vid_num: int
        Unique integer identifer for the video within the dataset that will be written to LipForensics/data/datasets/CelebDF/FakeCelebDF
        or LipForensics/data/datasets/CelebDF/RealCelebDF
    lip_forensics_absolute_path: str
        Absolute path to LipForensics repo
    fake: bool
        Whether the video is a deepfake or real video. Important because it determines whether the images and landmarks are saved to
        LipForensics/data/datasets/CelebDF/FakeCelebDF or LipForensics/data/datasets/CelebDF/RealCelebDF
    visualize_landmarks: bool
        Whether to display visualization of the landmarks on the face frame
    
    Returns:
    None. Saves face frames and landmarks to LipForensics/data/datasets/CelebDF/FakeCelebDF or LipForensics/data/datasets/CelebDF/RealCelebDF in
    the requirement format/naming convention for CelebDF, as specified in the LipForensics README.md. We arbitrarily choose CelebDF as the dataset
    we are imitating in order to get results for our own dataset.
    """
    if not os.path.exists(video_path):
        print(f"Video {video_path} does not exist")
        return

    print(f"Prepping {video_path}")
    start = time.time()
   
    if fake:
        dataset_name = "test"#"FakeCelebDF"
    else:
        dataset_name = "RealCelebDF"

    vid_num = str(vid_num).zfill(4)
    frames_output_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/{dataset_name}/images/{vid_num}/"
    landmarks_output_path = f"{lip_forensics_absolute_path}/data/datasets/CelebDF/{dataset_name}/landmarks/{vid_num}/"
  
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
            print("bad")
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

# participants = ["yuval", "colman", "toma", "phyllis", "hadleigh", "xiaofeng", "keylen", "rundi",  "tao", "dan", "qijia", "charlie", "ruoshi", "saeyoung", "kathryn", "naz", "honglin", "lisa", "abhinav",  "kahlil"]
# devices = ["googlepixel", "webcam", "canon", "iphone"]
# paragraph_nums = [1, 2, 3, 4, 5, 6]
# model = "faceswap"


# vid_file = open("faceswap_video_mapping.csv", "w")
# vid_file.write("Real/Fake,ID,path\n")
# df_vid_count = 0
# for participant in participants:
#     for device in devices:
#         for p in paragraph_nums:
#             vid_path = f"/media/admin/E380-1E91/Deepfake/End_To_End/deepfakes_may24/{participant}/{model}/{device}/p{p}_df.mp4"
#             if not os.path.exists(vid_path):
#                 continue
#             prep_video(vid_path, df_vid_count, "/home/hadleigh/deepfake_detection/system/evaluation/passive_detection/LipForensics", fake = True)
#             vid_file.write(f"Fake,{df_vid_count},{vid_path}\n")
#             df_vid_count += 1
# vid_file.close()

# aggregate_models("/home/hadleigh/deepfake_detection/system/evaluation/passive_detection/LipForensics")
remove_model_from_aggregate("/home/hadleigh/deepfake_detection/system/evaluation/passive_detection/LipForensics", "faceswap")