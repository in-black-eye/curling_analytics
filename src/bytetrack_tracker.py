"""
ByteTrack implementation for stone tracking in curling videos.
Based on ByteTrack: Multi-Object Tracking by Associating Every Detection Box
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from collections import deque


class STrack:
    """Single object tracking representation for ByteTrack"""
    
    def __init__(self, tlwh, score, track_id=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)  # [top, left, width, height]
        self.score = score
        self.track_id = track_id
        self.is_activated = False
        self.state = 'New'  # New, Tracked, Lost, Removed
        self.frame_id = 0
        self.start_frame = 0
        
        # Kalman filter states
        self.mean = np.zeros((4, 1), dtype=np.float32)
        self.covariance = np.zeros((4, 4), dtype=np.float32)
        self._init_kalman()
        
    def _init_kalman(self):
        """Initialize Kalman filter for motion prediction"""
        # State: [cx, cy, s, r] where (cx,cy) is center, s is scale, r is aspect ratio
        self.mean[0] = self.tlwh[0] + self.tlwh[2] / 2  # cx
        self.mean[1] = self.tlwh[1] + self.tlwh[3] / 2  # cy
        self.mean[2] = np.sqrt(self.tlwh[2] * self.tlwh[3])  # s
        self.mean[3] = self.tlwh[2] / self.tlwh[3]  # r
        
        # Covariance matrix
        self.covariance = np.eye(4, dtype=np.float32) * 2
        self.covariance[2:, 2:] *= 10.0
        
    def predict(self):
        """Predict next state using Kalman filter"""
        # Extend state to include velocity if needed
        if self.mean.shape[0] == 4:
            self.mean = np.vstack([self.mean, np.zeros((4, 1))])
            self.covariance = np.vstack([self.covariance, np.zeros((4, 4))])
            self.covariance = np.hstack([self.covariance, np.zeros((8, 4))])
        
        if self.state != 'Tracked':
            self.mean[7] = 0
        self.mean, self.covariance = self._kalman_predict(self.mean, self.covariance)
        
    def update(self, new_track, frame_id):
        """Update track with new detection"""
        self.frame_id = frame_id
        if new_track is not None:
            self.track_id = new_track.track_id
            self.is_activated = True
            self.state = 'Tracked'
            self.score = new_track.score
            
            # Update Kalman filter
            new_mean = np.zeros((4, 1), dtype=np.float32)
            new_mean[0] = new_track.tlwh[0] + new_track.tlwh[2] / 2
            new_mean[1] = new_track.tlwh[1] + new_track.tlwh[3] / 2
            new_mean[2] = np.sqrt(new_track.tlwh[2] * new_track.tlwh[3])
            new_mean[3] = new_track.tlwh[2] / new_track.tlwh[3]
            
            self.mean, self.covariance = self._kalman_update(
                self.mean, self.covariance, new_mean
            )
            
            # Update bbox
            self.tlwh = self._convert_to_tlwh(self.mean)
        else:
            self.state = 'Lost'
            
    def _kalman_predict(self, mean, covariance):
        """Kalman filter prediction step"""
        # Motion model: constant velocity
        dt = 1.0
        F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Extend state to include velocity
        if mean.shape[0] == 4:
            mean = np.vstack([mean, np.zeros((4, 1))])
            covariance = np.vstack([covariance, np.zeros((4, 4))])
            covariance = np.hstack([covariance, np.zeros((8, 4))])
        
        # Prediction
        mean = F @ mean
        Q = np.eye(8, dtype=np.float32) * 0.1
        covariance = F @ covariance @ F.T + Q
        
        return mean[:4], covariance[:4, :4]
    
    def _kalman_update(self, mean, covariance, measurement):
        """Kalman filter update step"""
        # Measurement model: direct observation of state
        H = np.eye(4, dtype=np.float32)
        R = np.eye(4, dtype=np.float32) * 0.1
        
        # Innovation
        y = measurement - H @ mean
        S = H @ covariance @ H.T + R
        
        # Kalman gain
        K = covariance @ H.T @ np.linalg.inv(S)
        
        # Update
        mean = mean + K @ y
        covariance = (np.eye(4) - K @ H) @ covariance
        
        return mean, covariance
    
    def _convert_to_tlwh(self, mean):
        """Convert Kalman state to tlwh format"""
        cx, cy, s, r = mean.flatten()
        w = s * np.sqrt(r)
        h = s / np.sqrt(r)
        return np.array([cx - w/2, cy - h/2, w, h], dtype=np.float32)
    
    @property
    def tlbr(self):
        """Convert tlwh to tlbr format"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @property
    def xyxy(self):
        """Convert tlwh to xyxy format"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret


class ByteTracker:
    """ByteTrack multi-object tracker"""
    
    def __init__(self, 
                 frame_rate=30,
                 track_thresh=0.5,
                 high_thresh=0.6,
                 match_thresh=0.8,
                 track_buffer=30,
                 mot20=False):
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.mot20 = mot20
        
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []     # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.frame_id = 0
        self.det_thresh = track_thresh + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.kalman_filter = None
        
    def update(self, detections: List[Dict]) -> List[STrack]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts with keys ['bbox', 'score', 'team']
            
        Returns:
            List of active tracks
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Convert detections to STrack format
        detections_strack = []
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            score = det['score']
            team = det.get('team', 'unknown')
            
            # Convert xyxy to tlwh
            tlwh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            track = STrack(tlwh, score)
            track.team = team
            detections_strack.append(track)
        
        # Separate detections by confidence
        detections_high = [d for d in detections_strack if d.score >= self.high_thresh]
        detections_low = [d for d in detections_strack if self.track_thresh <= d.score < self.high_thresh]
        
        # If no high confidence detections, use all detections above track_thresh
        if len(detections_high) == 0:
            detections_high = [d for d in detections_strack if d.score >= self.track_thresh]
            detections_low = []
        
        # Predict current tracks
        for track in self.tracked_stracks + self.lost_stracks:
            track.predict()
        
        # Associate high confidence detections
        tracked_stracks = self.tracked_stracks
        detections_high, tracked_stracks, unmatched_dets_high, unmatched_tracks_high = \
            self.associate_detections_to_trackers(detections_high, tracked_stracks, iou_threshold=self.match_thresh)
        
        # Update matched tracks
        for track in tracked_stracks:
            if track.state == 'Tracked':
                track.update(track, self.frame_id)
                activated_starcks.append(track)
            else:
                track.update(None, self.frame_id)
                refind_stracks.append(track)
        
        # Associate low confidence detections with lost tracks
        detections_low, lost_stracks, unmatched_dets_low, unmatched_lost_stracks = \
            self.associate_detections_to_trackers(detections_low, self.lost_stracks, iou_threshold=0.5)
        
        # Update matched lost tracks
        for track in lost_stracks:
            if track.state == 'Lost':
                track.update(track, self.frame_id)
                refind_stracks.append(track)
            else:
                track.update(None, self.frame_id)
                lost_stracks.append(track)
        
        # Handle unmatched detections (new tracks)
        for det in unmatched_dets_high + unmatched_dets_low:
            det.track_id = self._get_next_id()
            activated_starcks.append(det)
        
        # Handle unmatched tracks
        for track in unmatched_tracks_high + unmatched_lost_stracks:
            if track.state == 'Tracked':
                track.state = 'Lost'
                lost_stracks.append(track)
            else:
                track.state = 'Removed'
                removed_stracks.append(track)
        
        # Remove old tracks
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 'Tracked']
        self.lost_stracks = [t for t in self.lost_stracks if t.state == 'Lost']
        self.removed_stracks.extend(removed_stracks)
        
        # Update track states
        for track in self.tracked_stracks + self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = 'Removed'
                removed_stracks.append(track)
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 'Tracked']
        self.lost_stracks = [t for t in self.lost_stracks if t.state == 'Lost']
        
        return self.tracked_stracks
    
    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.5):
        """Associate detections to trackers using IoU"""
        if len(trackers) == 0:
            return detections, [], detections, []
        
        if len(detections) == 0:
            return [], trackers, [], trackers
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(detections, trackers)
        
        # Simple greedy assignment
        matched_detections = []
        matched_trackers = []
        unmatched_detections = []
        unmatched_trackers = []
        
        used_detections = set()
        used_trackers = set()
        
        # Find best matches
        for i in range(len(detections)):
            best_iou = 0
            best_tracker_idx = -1
            
            for j in range(len(trackers)):
                if j in used_trackers:
                    continue
                    
                iou = iou_matrix[i, j]
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_tracker_idx = j
            
            if best_tracker_idx != -1:
                matched_detections.append(detections[i])
                matched_trackers.append(trackers[best_tracker_idx])
                used_detections.add(i)
                used_trackers.add(best_tracker_idx)
        
        # Add unmatched detections and trackers
        for i, det in enumerate(detections):
            if i not in used_detections:
                unmatched_detections.append(det)
        
        for j, track in enumerate(trackers):
            if j not in used_trackers:
                unmatched_trackers.append(track)
        
        return matched_detections, matched_trackers, unmatched_detections, unmatched_trackers
    
    def _compute_iou_matrix(self, detections, trackers):
        """Compute IoU matrix between detections and trackers"""
        if len(detections) == 0 or len(trackers) == 0:
            return np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for i, det in enumerate(detections):
            for j, track in enumerate(trackers):
                iou_matrix[i, j] = self._compute_iou(det.tlbr, track.tlbr)
        
        return iou_matrix
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes in tlbr format"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _linear_assignment(self, cost_matrix):
        """Linear assignment using Hungarian algorithm (simplified version)"""
        # Simplified greedy assignment for now
        # In production, use scipy.optimize.linear_sum_assignment
        assignments = []
        used_rows = set()
        used_cols = set()
        
        # Sort by cost (descending)
        flat_indices = np.argsort(cost_matrix.flatten())[::-1]
        
        for idx in flat_indices:
            row, col = divmod(idx, cost_matrix.shape[1])
            if row not in used_rows and col not in used_cols:
                assignments.append([row, col])
                used_rows.add(row)
                used_cols.add(col)
        
        return np.array(assignments)
    
    def _get_next_id(self):
        """Get next available track ID"""
        if not hasattr(self, '_next_id'):
            self._next_id = 1
        self._next_id += 1
        return self._next_id - 1


def track_stones_with_bytetrack(video_path: str, homography_matrix: np.ndarray) -> List[Dict]:
    """
    Track stones in video using ByteTrack algorithm
    
    Args:
        video_path: Path to input video
        homography_matrix: Homography matrix for coordinate transformation
        
    Returns:
        List of stone tracks with positions and metadata
    """
    from src.stones_processing import detect_stones_in_frame, classify_stone_team
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")
    
    # Initialize ByteTracker
    tracker = ByteTracker(
        frame_rate=30,
        track_thresh=0.5,
        high_thresh=0.6,
        match_thresh=0.8,
        track_buffer=30
    )
    
    frame_num = 0
    all_tracks = []
    
    print("Начинаю трекинг камней с ByteTrack...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect stones in current frame
        detections = detect_stones_in_frame(frame)
        
        # Classify stone teams
        for det in detections:
            det["team"] = classify_stone_team(frame, det["bbox"])
        
        # Update tracker
        tracked_objects = tracker.update(detections)
        
        # Store track information
        for track in tracked_objects:
            bbox = track.xyxy  # Convert to xyxy format
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            track_data = {
                "track_id": track.track_id,
                "frame": frame_num,
                "bbox": bbox.tolist(),
                "center": [center_x, center_y],
                "team": getattr(track, 'team', 'unknown'),
                "score": track.score,
                "state": track.state
            }
            
            # Convert to real coordinates if homography is available
            if homography_matrix is not None:
                pixel_point = np.array([[[center_x, center_y]]], dtype=np.float32)
                real_point = cv2.perspectiveTransform(pixel_point, homography_matrix)
                rx, ry = real_point[0][0]
                track_data["center_real"] = [float(rx), float(ry)]
            
            all_tracks.append(track_data)
        
        frame_num += 1
        
        if frame_num % 30 == 0:
            print(f"Обработано {frame_num} кадров...")
    
    cap.release()
    
    # Group tracks by track_id
    tracks_dict = {}
    for track_data in all_tracks:
        track_id = track_data["track_id"]
        if track_id not in tracks_dict:
            tracks_dict[track_id] = {
                "stone_id": track_id,
                "team": track_data["team"],
                "positions_px": [],
                "positions_real": [],
                "bboxes": [],
                "scores": []
            }
        
        # Store pixel positions
        tracks_dict[track_id]["positions_px"].append({
            "frame": track_data["frame"],
            "x": track_data["center"][0],
            "y": track_data["center"][1]
        })
        
        # Store real positions if available
        if "center_real" in track_data:
            tracks_dict[track_id]["positions_real"].append({
                "frame": track_data["frame"],
                "x": track_data["center_real"][0],
                "y": track_data["center_real"][1]
            })
        
        # Store additional metadata
        tracks_dict[track_id]["bboxes"].append({
            "frame": track_data["frame"],
            "bbox": track_data["bbox"]
        })
        tracks_dict[track_id]["scores"].append({
            "frame": track_data["frame"],
            "score": track_data["score"]
        })
    
    # Convert to list and add metadata
    stone_tracks = list(tracks_dict.values())
    
    for track in stone_tracks:
        if len(track["positions_px"]) > 0:
            track["throw_start_frame"] = track["positions_px"][0]["frame"]
            track["throw_end_frame"] = track["positions_px"][-1]["frame"]
            if len(track["positions_real"]) > 0:
                track["final_position_real"] = track["positions_real"][-1]
    
    print(f"✅ ByteTrack завершен. Обнаружено {len(stone_tracks)} камней.")
    
    return stone_tracks
