# utils/tracker.py
import numpy as np
from collections import OrderedDict, deque

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()  # id -> centroid
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.history = {}  # id -> deque of centroids (for loiter / motion)
        self.bboxes = {}   # id -> last bbox

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.history[self.nextObjectID] = deque(maxlen=150)
        self.history[self.nextObjectID].append(centroid)
        self.bboxes[self.nextObjectID] = bbox
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.history[objectID]
        del self.bboxes[objectID]

    def update(self, rects):
        """
        rects: list of bboxes [x1,y1,x2,y2]
        returns: dict id -> bbox
        """
        if len(rects) == 0:
            # mark all disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.bboxes.copy()

        inputCentroids = []
        for (x1,y1,x2,y2) in rects:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids.append((cX, cY))

        if len(self.objects) == 0:
            for i,cent in enumerate(inputCentroids):
                self.register(cent, rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute distance matrix
            D = np.linalg.norm(np.array(objectCentroids)[:,None,:] - np.array(inputCentroids)[None,:,:], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row,col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bboxes[objectID] = rects[col]
                self.disappeared[objectID] = 0
                self.history[objectID].append(inputCentroids[col])

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # disappeared
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)

            # register new
            for col in unusedCols:
                self.register(inputCentroids[col], rects[col])

        return self.bboxes.copy()

    # Helpers for heuristics
    def get_speed(self, objectID):
        h = self.history.get(objectID)
        if not h or len(h) < 2:
            return 0.0
        (x1,y1) = h[-2]; (x2,y2) = h[-1]
        return np.hypot(x2-x1, y2-y1)

    def get_path_variance(self, objectID):
        h = self.history.get(objectID)
        if not h or len(h) < 4:
            return 0.0
        arr = np.array(h)
        return float(arr.var(axis=0).sum())
