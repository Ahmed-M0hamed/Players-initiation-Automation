import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_opening
class Segmentation: 
    def __init__(self  ,score_threshold :float = .5 , min_segment_frames :int = 30 , max_gap_frames : int = 15 ,
                  point_ratio_normalization :float = .4, conf_score_normalization :float = .4,
                    has_homography_normalization :float = .2 , homography_diff_threshold :float = 3.0 , min_frames_with_homography_change:int = 5):
        self.score_threshold = score_threshold 
        self.min_segment_frames = min_segment_frames
        self.max_gap_frames = max_gap_frames 
        self.point_ratio_normalization = point_ratio_normalization
        self.conf_score_normalization = conf_score_normalization
        self.has_homography_normalization = has_homography_normalization
        self.min_frames_with_homography_change = min_frames_with_homography_change
        self.homography_diff_threshold = homography_diff_threshold
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
        points = np.asarray(item)
        return np.sum(points == -1)
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
        df['smoothed_score'] = (df['score'].rolling(window=15, center=True, min_periods=1).mean())
        return df 
    def _sample_segments(self, smoothed_scores  , frame_ids ) : 
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
        for i, value in zip(frame_ids ,binary_scores):

            if value and not inside:

                inside = True
                start = i

            elif not value and inside:

                inside = False
                segments.append((start, i-1))

        if inside:
            segments.append((start, frame_ids[-1]))

        return segments , binary_scores
    def _homography_change(self, H1, H2):

        H1 = np.asarray(H1).reshape(3,3)
        H2 = np.asarray(H2).reshape(3,3)

        H1 = H1 / H1[2,2]
        H2 = H2 / H2[2,2]

        return np.linalg.norm(H1 - H2)
    def _check_segments(self, segments , df ) : 
        homography_dict = (
            df.set_index("frame_id")["homography"]
            .to_dict()
        )

        valid_segments = []
        for segment in segments : 
            reference_homography = None 
            reference_homography_frame_id = None 
            segment_start , segment_end = segment
            i = segment_start
            valid = True 
            while i < segment_end : 
                frame_homography = homography_dict.get(i)
                if reference_homography is None:

                    if frame_homography is None:
                        i += 1
                        continue

                    reference_homography = frame_homography
                    reference_homography_frame_id = i
                    i += 1
                    continue

                if frame_homography is not None : 
                    homography_diff = self._homography_change(reference_homography ,frame_homography ) 

                    if homography_diff < self.homography_diff_threshold : 
                        reference_homography = frame_homography 
                        reference_homography_frame_id = i 
                    else : 
                        j = 1 
                        state = False
                        while j < self.min_frames_with_homography_change and not state : 
                            nested_frame_homography = homography_dict.get(i+j)
                            if nested_frame_homography is not None : 
                                nested_homography_diff = self._homography_change(reference_homography ,nested_frame_homography ) 
                                if nested_homography_diff < self.homography_diff_threshold : 
                                    reference_homography = nested_frame_homography 
                                    reference_homography_frame_id = (i+j) 
                                    state = True 
                                    i += j
                            j +=1 
                        if not state : 
                            print('invalid_segment')
                            valid = False 
                i+=1 
            if valid : 
                valid_segments.append(segment)
        return valid_segments

            

    def detect(self, court_json  )  :
        df = self.turn_json_to_dataframe(court_json)
        df = self.data_engineering(df)
        smoothed_scores = df.smoothed_score.values 
        segments , binary_scores = self._sample_segments(smoothed_scores , df.frame_id.values.tolist())
        df['binary_scores' ] = binary_scores 
        valid_segments = self._check_segments(segments , df)
        return valid_segments , df 



