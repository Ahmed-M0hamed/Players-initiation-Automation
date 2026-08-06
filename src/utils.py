import cv2
from typing import List  , Tuple
import json 
import numpy as np 

def get_bottom_center_of_player(points : List ) -> Tuple : 
    x1 , y1 , x2, y2 = points 
    x_center = int(x1 + (x2-x1  )/2 ) 
    y_center = int(y1-(y1-y2)) 
    return (x_center, y_center) 
 
def read_frames(path) : 
    frames = [] 
    cap = cv2.VideoCapture(path) 
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True : 
        ret , frame = cap.read()  
        if not ret : 
            break 
        frames.append(frame) 
    cap.release() 
    return frames  ,fps 
def crop_frame(frame, ball_center, target_size):
    frame_h, frame_w = frame.shape[:2]
    crop_h, crop_w = target_size

    x, y = map(int, ball_center)

    x1 = x - crop_w // 2
    y1 = y - crop_h // 2

    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Shift horizontally
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if x2 > frame_w:
        x1 -= (x2 - frame_w)
        x2 = frame_w

    # Shift vertically
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if y2 > frame_h:
        y1 -= (y2 - frame_h)
        y2 = frame_h

    # Final clamp
    x1 = max(0, x1)
    y1 = max(0, y1)

    return frame[y1:y2, x1:x2]


def write_video(  output_path , frames , fps  ) : 
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc,fps , (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames :
            out.write(frame)
        out.release()

def read_annoations(path : str ) -> List : 
    with open(path , 'rb') as f :
        annotations = json.load(f) 
    return annotations


def write_labels(results:List , output_path:str)->None : 
    with open(output_path , 'w',encoding="utf-8") as f : 
        json.dump(results , f , indent=4)  

def find_gaps(annotations : List ) -> List  :
    gap_start = 0 
    index = 0 
    gaps = []
    while index < len(annotations): 
        if 'ball_position' in annotations[index] : 
            gap_end = annotations[index]['frame_id'] 
            if gap_end - gap_start > 1 : 
                gaps.append((gap_start , gap_end)) 
            gap_start = gap_end 
        index += 1 
    return gaps 

import pandas as pd 




def distance_between_two_points(point_1 : Tuple , point_2 : Tuple) -> float  : 
    x1 , y1 = point_1 
    x2, y2 = point_2 
    distance = np.sqrt((x2-x1) **2 + (y2-y1)**2)
    return distance 


def detect_player_hitting(near_player : List  , far_player : List , ball_pos:Tuple) -> Tuple : 
    near_player_bottom_center = get_bottom_center_of_player(near_player) 
    far_player_bottom_center = get_bottom_center_of_player(far_player) 
    players = [near_player_bottom_center , far_player_bottom_center]
    classes = ['near' , 'far']
    distances = [distance_between_two_points(ball_pos , player) for player in players ] 
    min_index = np.argmin(distances) 
    return (players[min_index] , classes[min_index] ) 

def load_n_frames(path , n_frames ) :

    cap = cv2.VideoCapture(path)
    max_frames = n_frames
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        # Do your processing on 'frame' here
        count += 1

    cap.release()
    return frames , fps 