import time
from collections import deque


class FallDetector:
    def __init__(self):
        self.hip_history = deque()

        self.history_seconds = 1.0
        self.ground_seconds = 2.0
        self.fall_candidate_seconds = 1.5

        self.state = "NORMAL"
        self.on_ground_since = None
        self.fall_candidate_since = None

    def update(self, keypoints, frame_height):
        now = time.time()

        if keypoints is None or len(keypoints) == 0:
            return self.state

        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]

        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
        hip_x = (left_hip[0] + right_hip[0]) / 2


        self.hip_history.append((now, hip_y))

        while (
            self.hip_history
            and now - self.hip_history[0][0] > self.history_seconds
        ):
            self.hip_history.popleft()

        body_height = abs(ankle_y - shoulder_y)
        torso_width = abs(hip_x - shoulder_x)
        torso_height = abs(hip_y - shoulder_y)

        near_floor = hip_y > frame_height * 0.5
        body_is_horizontal = torso_width > torso_height

        fast_downward_movement = False

        if len(self.hip_history) >= 2:
            old_hip_y = self.hip_history[0][1]
            movement = hip_y - old_hip_y

            if movement > frame_height * 0.12:
                fast_downward_movement = True
                self.fall_candidate_since = now

        recent_downward_movement = (
            self.fall_candidate_since is not None
            and now - self.fall_candidate_since < self.fall_candidate_seconds
        )

        if self.state == "NORMAL":
            if recent_downward_movement and near_floor:
                self.state = "POSSIBLE_FALL"
                self.on_ground_since = now

        elif self.state == "POSSIBLE_FALL":
            if near_floor and body_is_horizontal:
                if now - self.on_ground_since >= self.ground_seconds:
                    self.state = "FALL"
            elif not recent_downward_movement:
                self.state = "NORMAL"
                self.on_ground_since = None


        elif self.state == "FALL":
            if not body_is_horizontal:
                self.state = "NORMAL"
                self.on_ground_since = None
                self.fall_candidate_since = None

        print(
            f"state={self.state} | "
            f"hip_y={hip_y:.0f} | "
            f"near_floor={near_floor} | "
            f"horizontal={body_is_horizontal} | "
            f"fast_down={fast_downward_movement} | "
            f"recent_down={recent_downward_movement}"
        )

        return self.state