import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_opening
class Segmentation: 
    def __init__(self  ,score_threshold :float = .5 , min_segment_frames :int = 30 , max_gap_frames : int = 15 ,
                  point_ratio_normalization :float = .5, conf_score_normalization :float = .3,
                    has_homography_normalization :float = .2):
        self.score_threshold = score_threshold 
        self.min_segment_frames = min_segment_frames
        self.max_gap_frames = max_gap_frames 
        self.point_ratio_normalization = point_ratio_normalization
        self.conf_score_normalization = conf_score_normalization
        self.has_homography_normalization = has_homography_normalization
    def turn_json_to_dataframe(self, data):
        df_rows = {'court_points' : []  , 'frame_id'  :[] ,'court_det_confidence' : [] , 'homography':[] }
        for d in data : 
            court_points = d['court_points'] if 'court_points' in d else None
            frame_id = d['frame_id'] 
            conf = d['court_det_confidence'] if 'court_det_confidence' in d else None 
            homography = d['homography'] if 'homography' in d else None 
            df_rows['court_points'].append(court_points)
            df_rows['frame_id'].append(frame_id)
            df_rows['court_det_confidence'].append(conf)
            df_rows['homography'].append(homography) 
        df = pd.DataFrame(df_rows)
        return df 
    def _count_negatives(self , item) : 
        flattened = [i for sublist in item for i in sublist]
        return flattened.count(-1) 
    def _has_homography(self, item ) : 
        if item is not None : 
            return 1 
        else : 
            return 0
    def _score(self , points_ratio , conf , has_homograply ) : 
        score = (
        self.point_ratio_normalization * points_ratio +
        self.conf_score_normalization * (conf / 100) +
        self.has_homography_normalization * has_homograply
        )
        return score 
    def data_engineering(self, df  ) : 
        df['negative_count'] = df['court_points'].map(self._count_negatives)
        df['num_existence_points'] = (28 - df['negative_count'] ) // 2 
        df['existence_points_ratio'] = df['num_existence_points'] / 14 
        df['has_homography'] = df['homography'].map(self._has_homography)
        df['score'] = df.apply(lambda row: self._score(row['existence_points_ratio'], row['court_det_confidence'], row['has_homography']), axis=1)
        df['smoothed_score'] = df['score'].rolling(window = 15 , center = True).mean()
        return df 

    def detect(self, court_json  )  :
        df = self.turn_json_to_dataframe(court_json)
        df = self.data_engineering(df)
        smoothed_scores = df.smoothed_score.values 
        binary_scores = smoothed_scores >= self.score_threshold
        binary_scores = binary_closing(
                    binary_scores,
                    structure=np.ones(self.max_gap_frames)
                )
        binary_scores = binary_opening(
                    binary_scores,
                    structure=np.ones(self.min_segment_frames)
                )
        
        segments = []
        inside = False
        start = 0
        for i, value in zip(df.frame_id.values.tolist(),binary_scores):

            if value and not inside:

                inside = True
                start = i

            elif not value and inside:

                inside = False
                segments.append((start, i-1))

        if inside:
            segments.append((start, len(binary_scores)-1))

        return {
            'df' : df , 
            "mask": binary_scores,
            "segments": segments,
        }
    



