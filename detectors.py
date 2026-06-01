# utils/detectors.py
import numpy as np
import cv2
from collections import defaultdict

class SuspicionEngine:
    def __init__(self, frame_shape, fps):
        self.h, self.w = frame_shape[:2]
        self.fps = fps
        self.heatmap = np.zeros((self.h, self.w), dtype=np.float32)
        self.events = []  # list of dicts: {type, start, end, ids, bbox, score}
        self.active_events = {}
        # configuration thresholds (tweakable)
        self.LOITER_FRAMES = int(fps * 12)  # 12 seconds
        self.LOITER_RADIUS = 40  # px
        self.CROWD_THRESHOLD = 6
        self.FIGHTING_PROX = 60
        self.FIGHT_VARIANCE = 50.0
        self.COLLAPSE_AREA_RATIO = 1.6  # width/height changes suggest lying
        self.ACCIDENT_SPEED_DROP = 8.0

        # per-id trackers for suspicious flags
        self.loiter_counters = defaultdict(int)  # id -> frames
        self.prev_speeds = {}

    def accumulate_heat(self, bbox):
        x1,y1,x2,y2 = [int(v) for v in bbox]
        x1 = max(0, min(self.w-1, x1))
        x2 = max(0, min(self.w-1, x2))
        y1 = max(0, min(self.h-1, y1))
        y2 = max(0, min(self.h-1, y2))
        self.heatmap[y1:y2, x1:x2] += 1.0

    def detect(self, frame_idx, timestamp, detections, tracker, fps):
        """
        detections: list of dicts: {'bbox':[x1,y1,x2,y2], 'class': 'person'/'car', 'conf': float, 'id': int}
        tracker: instance of CentroidTracker to get history and speeds
        timestamp: seconds
        """
        # Accumulate
        for det in detections:
            self.accumulate_heat(det['bbox'])

        # Crowd detection
        person_count = sum(1 for d in detections if d['class'] == 'person')
        if person_count >= self.CROWD_THRESHOLD:
            self._record_event('crowd', frame_idx, timestamp, [d for d in detections if d['class']=='person'])

        # Per-person heuristics
        persons = [d for d in detections if d['class']=='person']
        ids = [p['id'] for p in persons if p['id'] is not None]
        # Fighting detection: two persons very close with high path variance
        for i,p in enumerate(persons):
            pid = p['id']
            if pid is None: continue
            speed = tracker.get_speed(pid)
            var = tracker.get_path_variance(pid)
            # loiter
            if len(tracker.history[pid]) >= self.LOITER_FRAMES:
                # check if centroid remained inside small radius over LOITER_FRAMES
                pts = list(tracker.history[pid])[-self.LOITER_FRAMES:]
                xs = [c[0] for c in pts]; ys = [c[1] for c in pts]
                if (max(xs)-min(xs) < self.LOITER_RADIUS) and (max(ys)-min(ys) < self.LOITER_RADIUS):
                    self._record_event('loitering', frame_idx, timestamp, [p])
            # collapse heuristic: very low speed + bbox aspect change (wide not tall)
            x1,y1,x2,y2 = p['bbox']
            w = x2-x1; h = y2-y1+1
            if speed < 1.0 and (w / float(h) > self.COLLAPSE_AREA_RATIO):
                self._record_event('collapse', frame_idx, timestamp, [p])
            # fighting check: nearby persons with high path variance
            for j,q in enumerate(persons):
                if i == j: continue
                pid2 = q['id']; 
                if pid2 is None: continue
                # distance
                cx = (x1+x2)/2; cy = (y1+y2)/2
                x12,y12 = ( (q['bbox'][0]+q['bbox'][2])/2, (q['bbox'][1]+q['bbox'][3])/2 )
                dist = np.hypot(cx-x12, cy-y12)
                var2 = tracker.get_path_variance(pid2)
                if dist < self.FIGHTING_PROX and (var + var2) > self.FIGHT_VARIANCE:
                    self._record_event('fighting', frame_idx, timestamp, [p,q])
            # accident-like: sudden speed drop (e.g., person hit by vehicle)
            prev_speed = self.prev_speeds.get(pid, speed)
            if prev_speed - speed > self.ACCIDENT_SPEED_DROP and any(d['class'] in ('car','truck','motorbike','bicycle') for d in detections):
                self._record_event('accident', frame_idx, timestamp, [p])
            self.prev_speeds[pid] = speed

    def _record_event(self, evt_type, frame_idx, timestamp, dets):
        # If similar active event exists recently, extend it; else create new
        key = (evt_type)
        if key in self.active_events:
            ev = self.active_events[key]
            ev['end_frame'] = frame_idx
            ev['end_time'] = timestamp
            ev['dets'].extend(dets)
        else:
            ev = {
                'type': evt_type,
                'start_frame': frame_idx,
                'start_time': timestamp,
                'end_frame': frame_idx,
                'end_time': timestamp,
                'dets': list(dets)
            }
            self.active_events[key] = ev
            self.events.append(ev)

    def finalize_events(self):
        # convert active_events to finalized list (already in events)
        self.active_events.clear()

    def generate_heatmap_overlay(self):
        # normalize heatmap and convert to colored heatmap
        hm = self.heatmap.copy()
        if hm.max() > 0:
            hm = (hm / hm.max() * 255).astype('uint8')
        else:
            hm = hm.astype('uint8')
        hm_small = cv2.resize(hm, (self.w, self.h))
        colored = cv2.applyColorMap(hm_small, cv2.COLORMAP_JET)
        return colored
