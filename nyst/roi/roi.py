import numpy as np

class FirstRoi:
    def __init__(self, face_obj):
        self.left_eye = np.array(face_obj['face']['facial_area']['left_eye'], dtype=int)
        self.right_eye = np.array(face_obj['face']['facial_area']['right_eye'], dtype=int)

        distance = int(np.linalg.norm(self.right_eye - self.left_eye))

        half_distance = distance // 2

        self.left_eye_roi = np.concatenate([self.left_eye - half_distance, self.left_eye + half_distance])
        self.right_eye_roi = np.concatenate([self.right_eye - half_distance, self.right_eye + half_distance])

    def get_left_eye(self):
        return self.left_eye
    
    def get_right_eye(self):
        return self.right_eye

    def get_left_eye_roi(self):
        return self.left_eye_roi
    
    def get_right_eye_roi(self):
        return self.right_eye_roi