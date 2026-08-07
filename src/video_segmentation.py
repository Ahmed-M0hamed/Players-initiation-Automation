import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_opening
class Segmentation: 
    def __init__(self,
        score_threshold: float = 0.5,
        min_segment_frames: int = 30,
        max_gap_frames: int = 15,
        smoothing_window_frames: int = 15,
        point_ratio_normalization: float = 0.4,
        conf_score_normalization: float = 0.4,
        has_homography_normalization: float = 0.2,
        confidence_scale: float = 100.0,
        homography_diff_threshold: float = 3.0,
        min_frames_with_homography_change: int = 5):


        self.score_threshold = score_threshold
        self.min_segment_frames = min_segment_frames
        self.max_gap_frames = max_gap_frames
        self.smoothing_window_frames = smoothing_window_frames
        self.point_ratio_normalization = point_ratio_normalization
        self.conf_score_normalization = conf_score_normalization
        self.has_homography_normalization = has_homography_normalization
        self.confidence_scale = confidence_scale
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
    def _existence_ratio(self, item):
        """
        Returns fraction of keypoints that are present (not [-1,-1]).
        Handles missing detections (None) as ratio 0 instead of silently
        being treated as fully present.
        """
        if item is None:
            return 0.0
        points = np.asarray(item, dtype=float).reshape(-1, 2)
        if points.shape[0] == 0:
            return 0.0
        missing = np.all(np.isclose(points, -1.0), axis=1)
        return float((~missing).sum()) / points.shape[0]

    def _has_homography(self, item):
        if item is None:
            return 0
        arr = np.asarray(item, dtype=float)
        if arr.size == 0 or np.allclose(arr, 0.0):
            return 0
        return 1
    def _score(self, points_ratio, conf, has_homography):
        conf = 0.0 if conf is None or (isinstance(conf, float) and np.isnan(conf)) else conf
        score = (
            self.point_ratio_normalization * points_ratio
            + self.conf_score_normalization * (conf / self.confidence_scale)
            + self.has_homography_normalization * has_homography
        )
        return score
    def data_engineering(self, df):
        df = df.copy()
        df["existence_points_ratio"] = df["court_points"].map(self._existence_ratio)
        df["has_homography"] = df["homography"].map(self._has_homography)
        df["score"] = df.apply(
            lambda row: self._score(row["existence_points_ratio"], row["court_det_confidence"], row["has_homography"]),
            axis=1,
        )
        df["smoothed_score"] = (
            df["score"].rolling(window=self.smoothing_window_frames, center=True, min_periods=1).mean()
        )
        return df
    def _sample_segments(self, smoothed_scores, frame_ids):
        binary_scores = smoothed_scores >= self.score_threshold
        binary_scores = binary_closing(binary_scores, structure=np.ones(self.max_gap_frames))
        binary_scores = binary_opening(binary_scores, structure=np.ones(self.min_segment_frames))

        frame_ids = np.asarray(frame_ids)
        segments = []
        inside = False
        start = None
        prev_id = None
        for fid, value in zip(frame_ids, binary_scores):
            if value and not inside:
                inside = True
                start = fid
            elif not value and inside:
                inside = False
                segments.append((start, prev_id))
            prev_id = fid
        if inside:
            segments.append((start, frame_ids[-1]))
        return segments, binary_scores
    def _homography_change(self, H1, H2):
        H1 = np.asarray(H1, dtype=float).reshape(3, 3)
        H2 = np.asarray(H2, dtype=float).reshape(3, 3)
        if np.isclose(H1[2, 2], 0) or np.isclose(H2[2, 2], 0):
            return np.inf
        H1 = H1 / H1[2, 2]
        H2 = H2 / H2[2, 2]
        return np.linalg.norm(H1 - H2)
    def _split_segment_on_homography_breaks(self, segment, homography_dict):
        """
        Scans a segment for confirmed homography discontinuities (a jump
        that isn't recovered within `min_frames_with_homography_change`
        frames) and splits the segment at each break, instead of
        discarding the whole thing. Sub-segments shorter than
        min_segment_frames are dropped.
        """
        segment_start, segment_end = segment
        sub_segments_raw = []
        cur_start = segment_start
        reference_homography = None
        i = segment_start

        while i <= segment_end:
            frame_h = homography_dict.get(i)

            if reference_homography is None:
                if frame_h is not None:
                    reference_homography = frame_h
                i += 1
                continue

            if frame_h is not None:
                diff = self._homography_change(reference_homography, frame_h)
                if diff < self.homography_diff_threshold:
                    reference_homography = frame_h
                    i += 1
                    continue

                # Jump detected — look ahead for recovery
                recovered = False
                for j in range(1, self.min_frames_with_homography_change):
                    nested_h = homography_dict.get(i + j)
                    if nested_h is not None:
                        nested_diff = self._homography_change(reference_homography, nested_h)
                        if nested_diff < self.homography_diff_threshold:
                            recovered = True
                            reference_homography = nested_h
                            i = i + j
                            break

                if not recovered:
                    print('invalid_segment')
                    sub_segments_raw.append((cur_start, i - 1))
                    cur_start = i
                    reference_homography = frame_h  # start fresh reference at the break

            i += 1

        sub_segments_raw.append((cur_start, segment_end))

        return [
            (s, e) for s, e in sub_segments_raw
            if (e - s + 1) >= self.min_segment_frames
        ]

    def _check_segments(self, segments, df):
        homography_dict = df.set_index("frame_id")["homography"].to_dict()
        valid_segments = []
        for segment in segments:
            valid_segments.extend(self._split_segment_on_homography_breaks(segment, homography_dict))
        return valid_segments

            

    def detect(self, court_json  )  :
        df = self.turn_json_to_dataframe(court_json)
        df = self.data_engineering(df)
        smoothed_scores = df.smoothed_score.values 
        segments , binary_scores = self._sample_segments(smoothed_scores , df.frame_id.values.tolist())
        df['binary_scores' ] = binary_scores 
        valid_segments = self._check_segments(segments , df)
        return valid_segments , df 



