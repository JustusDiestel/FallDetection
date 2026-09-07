


class FallDetector:
    def __init__(self):
        self.frames_on_ground = 0

    def update(self, keypoints, frame_height):
        if keypoints is None or len(keypoints) == 0:
            self.frames_on_ground = 0
            return "NO PERSON DETECTED"


        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]


        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2

        body_height = ankle_y - shoulder_y
        near_floor = hip_y > frame_height * 0.8 and ankle_y > frame_height * 0.9
        compressed_body = body_height < frame_height * 0.35

        if near_floor and compressed_body:
            self.frames_on_ground += 1
        else:
            self.frames_on_ground = 0
        if self.frames_on_ground > 30:
            return "FALL"
        if self.frames_on_ground > 0:
            return "POSSIBLE_FALL"
        return "NORMAL"

