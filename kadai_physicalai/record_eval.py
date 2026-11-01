"""Run the official BEHAVIOR evaluator with progress reporting and optional spectator camera.

The head/wrist MP4 uses the official camera layout with camera-name captions.
Scoring JSON remains authored by upstream. Each completed simulation step
is reported to the GUI using a tagged stdout line. A second, clearly
labeled presentation MP4 includes a third-person view.
No extra observation is supplied to the policy.

This module intentionally imports OmniGibson only within main() so the package's
GUI, CLI, doctor, and unit tests remain usable without Isaac Sim installed.
"""
from __future__ import annotations

import os


def compose_overview(external, head, left_wrist, right_wrist, *, label="FIXED OVERHEAD CAMERA"):
    """Make a 1280x720 RGB presentation frame, preserving the complete camera views.

    The spectator pane is letterboxed (never center-cropped) so robot limbs are
    not cut off just to fill the presentation video. Camera arrays can be RGB or
    RGBA, uint8 numpy arrays or CPU/GPU torch tensors.
    """
    import cv2
    import numpy as np

    def to_rgb(data):
        if hasattr(data, 'detach'):
            data = data.detach().cpu().numpy()
        data = np.asarray(data)
        if data.ndim != 3 or data.shape[2] < 3:
            raise ValueError(f'Invalid camera frame shape: {data.shape}')
        return np.ascontiguousarray(data[:, :, :3].astype(np.uint8, copy=False))

    def contain(image, h, w):
        ih, iw = image.shape[:2]
        if ih <= 0 or iw <= 0:
            raise ValueError('Camera returned an empty frame')
        scale = min(w / iw, h / ih)
        nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        surface = np.zeros((h, w, 3), dtype=np.uint8)
        surface[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw] = resized
        return surface

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :854] = contain(to_rgb(external), 720, 854)
    frame[:426, 854:] = contain(to_rgb(head), 426, 426)
    frame[426:639, 854:1067] = contain(to_rgb(left_wrist), 213, 213)
    frame[426:639, 1067:] = contain(to_rgb(right_wrist), 213, 213)
    for text, x, y in (
        (label, 14, 32),
        ('HEAD CAMERA', 869, 32),
        ('LEFT WRIST', 865, 659),
        ('RIGHT WRIST', 1077, 659),
    ):
        cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.59, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.59, (255, 255, 255), 1, cv2.LINE_AA)
    return frame



def compose_standard(head, left_wrist, right_wrist):
    """Official 672x448 head/wrist layout, with captions matching overview.

    Original placement (BEHAVIOR-1K v3.9.3-post2): left wrist top-left,
    right wrist bottom-left (224px each), head right (448px). Only the
    encoded video is annotated; policy observations are never modified.
    """
    import cv2
    import numpy as np

    def resize_camera(data, size):
        if hasattr(data, 'detach'):
            data = data.detach().cpu().numpy()
        rgb = np.asarray(data)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or not rgb.shape[0] or not rgb.shape[1]:
            raise ValueError(f'Invalid camera frame shape: {rgb.shape}')
        return cv2.resize(np.ascontiguousarray(rgb[:, :, :3].astype(np.uint8, copy=False)), size)

    head_rgb = resize_camera(head, (448, 448))
    left_rgb = resize_camera(left_wrist, (224, 224))
    right_rgb = resize_camera(right_wrist, (224, 224))
    frame = np.hstack((np.vstack((left_rgb, right_rgb)), head_rgb))
    for caption, x, y in (
        ('LEFT WRIST', 12, 26),
        ('RIGHT WRIST', 12, 250),
        ('HEAD CAMERA', 237, 26),
    ):
        cv2.putText(frame, caption, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.53, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, caption, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.53, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _look_at_pose(eye, target):
    """Build a world-space Isaac/Omni camera pose (local -Z looks at target).

    Using world Y for the camera-up hint near vertical avoids the singularity
    in the original orbit camera's cross(world_z, view_back) calculation.
    """
    import numpy as np
    import torch as th
    import omnigibson.utils.transform_utils as T

    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    backward = eye - target
    distance = np.linalg.norm(backward)
    if distance < 1e-6:
        raise ValueError('Camera eye and target are identical')
    backward /= distance
    up_hint = np.array([0., 0., 1.])
    if abs(np.dot(backward, up_hint)) > 0.95:
        # A downward-pointing ceiling camera needs a different "up" axis.
        up_hint = np.array([0., 1., 0.])
    right = np.cross(up_hint, backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    rotation = np.column_stack((right, up, backward))
    quaternion = T.mat2quat(th.as_tensor(rotation, dtype=th.float32))
    return th.as_tensor(eye, dtype=th.float32), quaternion


def fixed_overhead_pose(robot, *, height=2.2, offset_x=0.40, offset_y=-0.35,
                        target_height=0.65):
    """Ceiling-style stationary camera near the *initial* robot position.

    Unlike the previous robot-following camera 4 m away, this eye stays inside
    the robot's immediate room in many indoor scenes. Its position is fixed
    after the first frame, so the viewer sees robot motion objectively.
    Note: actual roofs or tall furniture can still occlude any external eye.
    """
    import numpy as np
    base_pos, _ = robot.get_position_orientation()
    base = base_pos.detach().cpu().numpy() if hasattr(base_pos, 'detach') else np.asarray(base_pos)
    eye = base + np.array([offset_x, offset_y, height], dtype=float)
    target = base + np.array([0., 0., target_height], dtype=float)
    return _look_at_pose(eye, target)


def _rear_follow_eye_target(base, rotation, *, distance=0.9, side=0.425,
                            height=2.0, target_height=1.0, target_forward=0.3):
    """Compute a camera position behind and to the robot's right, in world XYZ.

    Galaxea's robot-local +X is forward and +Y is left. Keep the camera's
    elevation on the *world* vertical so turning / base tilt never moves the
    camera below the floor. Recalculate from each robot pose (not the initial
    pose) so the camera continues tracking after a turn or translation.
    """
    import numpy as np
    base = np.asarray(base, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    forward = rotation[:2, 0].copy()
    length = np.linalg.norm(forward)
    if length < 1e-7:
        raise ValueError('Robot heading is undefined')
    forward /= length
    left = np.array([-forward[1], forward[0]], dtype=float)
    eye = base.copy()
    eye[:2] -= distance * forward + side * left
    eye[2] += height
    target = base.copy()
    target[:2] += target_forward * forward
    target[2] += target_height
    return eye, target


def spectator_pose(robot, *, distance=0.9, side=0.425, height=2.0,
                   target_height=1.0):
    """Robot-relative rear-quarter chase camera, facing the robot from behind."""
    import numpy as np
    import omnigibson.utils.transform_utils as T
    base_pos, base_quat = robot.get_position_orientation()
    base = base_pos.detach().cpu().numpy() if hasattr(base_pos, 'detach') else np.asarray(base_pos)
    rotation = T.quat2mat(base_quat)
    rotation = rotation.detach().cpu().numpy() if hasattr(rotation, 'detach') else np.asarray(rotation)
    eye, target = _rear_follow_eye_target(base, rotation, distance=distance,
                                          side=side, height=height,
                                          target_height=target_height)
    return _look_at_pose(eye, target)


def main():
    import logging
    from pathlib import Path

    import numpy as np
    import omnigibson as og
    import omnigibson.eval.eval as official_eval
    from omnigibson.eval.utils.obs_utils import create_video_writer, write_video
    from omnigibson.eval.utils.eval_utils import EVAL_TIMEOUT_MULTIPLIER

    logger = logging.getLogger('kadai_physicalai.record_eval')
    logger.setLevel(logging.INFO)

    class PresentationEvaluator(official_eval.BatchedEvaluator):
        """Keep all upstream scoring, controls, and official recording intact."""

        def __init__(self, cfg):
            self._kadai_overview = {}
            self._kadai_frame_count = {}
            self._kadai_steps = {}
            self._kadai_fixed_pose = {}
            self._kadai_camera_warmed = set()
            self._kadai_camera_mode = os.environ.get('KADAI_CAMERA_MODE', 'follow')
            if self._kadai_camera_mode not in ('fixed_overhead', 'follow'):
                raise ValueError('KADAI_CAMERA_MODE must be fixed_overhead or follow')
            self._kadai_disabled = os.environ.get('KADAI_OVERVIEW_VIDEO', '1') == '0' 
            super().__init__(cfg)
            # The pinned BEHAVIOR-1K v3.9.3-post2 evaluator sets the task
            # timeout in load_env(), but does NOT expose self.max_steps.
            # Resolve the displayed limit exactly as upstream does instead of
            # accessing an attribute that exists only in newer revisions.
            configured_limit = self.cfg.max_steps
            self._kadai_max_steps = (
                int(configured_limit)
                if configured_limit is not None
                else int(self.human_stats["length"] * EVAL_TIMEOUT_MULTIPLIER)
            )

        @staticmethod
        def _close_writer(writer):
            if writer is None:
                return
            container, stream = writer
            try:
                for packet in stream.encode():
                    container.mux(packet)
            finally:
                container.close()

        def _discard_overview(self, env_idx, *, delete_file=False):
            record = self._kadai_overview.pop(env_idx, None)
            if record is None:
                return
            path, writer = record
            frames = self._kadai_frame_count.pop(env_idx, 0)
            try:
                self._close_writer(writer)
            except Exception as exc:
                logger.warning('Could not finalize presentation video %s: %s', path, exc)
                delete_file = True
            finally:
                if delete_file or frames == 0:
                    Path(path).unlink(missing_ok=True)

        def _set_video_writer(self, instance_eval_state, video_writer):
            # Called by the *original* evaluator on episode finish and shutdown.
            if video_writer is None:
                self._discard_overview(instance_eval_state.env_idx)
            super()._set_video_writer(instance_eval_state, video_writer)

        def load_batch(self, env_idx_to_instance, write_video=False, video_path=None, rollout_id=0, video_fps=30):
            # Keep the original evaluator in charge of resets, policy, and head/wrist video.
            for env_idx in env_idx_to_instance:
                self._discard_overview(env_idx)
                self._kadai_fixed_pose.pop(env_idx, None)
                self._kadai_camera_warmed.discard(env_idx)
            super().load_batch(env_idx_to_instance, write_video, video_path, rollout_id, video_fps)
            # Called after scene reset. Count one successful action / simulator
            # advancement per _write_video callback in official eval.run().
            for env_idx in env_idx_to_instance:
                self._kadai_steps[env_idx] = 0
                print(f'[KADAI_PROGRESS] step=0 total={self._kadai_max_steps}', flush=True)
            if not write_video or self._kadai_disabled:
                return
            if og.sim.viewer_camera is None:
                logger.warning('Third-person camera unavailable: official video will still be saved.')
                self._kadai_disabled = True
                return
            try:
                for env_idx, instance in env_idx_to_instance.items():
                    path = os.path.join(video_path, f'{self.cfg.task.name}_{instance}_{rollout_id}_overview.mp4')
                    writer = create_video_writer(fpath=path, resolution=(720, 1280), rate=video_fps)
                    self._kadai_overview[env_idx] = (path, writer)
                    self._kadai_frame_count[env_idx] = 0
                logger.info('Presentation recording enabled: %s (camera: %s)', video_path, self._kadai_camera_mode)
            except Exception as exc:
                logger.warning('Cannot create presentation video, official video unaffected: %s', exc)
                self._kadai_disabled = True
                for env_idx in list(self._kadai_overview):
                    self._discard_overview(env_idx, delete_file=True)

        def _write_video(self, instance_eval_state):
            # Preserve upstream's exact 672x448 camera layout but caption each
            # view. Read-only: no modification to obs or the policy input.
            obs = getattr(instance_eval_state, 'obs', None)
            writer = getattr(instance_eval_state, 'video_writer', None)
            names = getattr(self, 'robot_camera_names', {})
            if (obs is not None and writer is not None
                    and all(role in names and names[role] + '::rgb' in obs
                            for role in ('head', 'left_wrist', 'right_wrist'))):
                standard_frame = compose_standard(
                    obs[names['head'] + '::rgb'],
                    obs[names['left_wrist'] + '::rgb'],
                    obs[names['right_wrist'] + '::rgb'],
                )
                write_video(np.expand_dims(standard_frame, 0),
                            video_writer=writer, batch_size=1, mode='rgb')
            else:
                # Retain upstream handling for evaluators that provide no
                # standard recording or use a different camera set.
                super()._write_video(instance_eval_state)
            env_idx = instance_eval_state.env_idx
            self._kadai_steps[env_idx] = self._kadai_steps.get(env_idx, 0) + 1
            print(f'[KADAI_PROGRESS] step={self._kadai_steps[env_idx]} total={self._kadai_max_steps}', flush=True)
            if env_idx not in self._kadai_overview:
                return
            try:
                robot = instance_eval_state.env_accessor.robot
                height = float(os.getenv('KADAI_CAMERA_HEIGHT', '2.2'))
                if self._kadai_camera_mode == 'fixed_overhead':
                    # Capture *once* per reset: fixed in world coordinates, not
                    # recalculated as the robot translates or turns.
                    if env_idx not in self._kadai_fixed_pose:
                        self._kadai_fixed_pose[env_idx] = fixed_overhead_pose(
                            robot, height=height,
                            offset_x=float(os.getenv('KADAI_CAMERA_OVERHEAD_X', '0.40')),
                            offset_y=float(os.getenv('KADAI_CAMERA_OVERHEAD_Y', '-0.35')),
                        )
                    eye, quat = self._kadai_fixed_pose[env_idx]
                else:
                    eye, quat = spectator_pose(
                        robot,
                        distance=float(os.getenv('KADAI_CAMERA_DISTANCE', '0.9')),
                        side=float(os.getenv('KADAI_CAMERA_SIDE', '0.425')),
                        height=height,
                    )
                camera = og.sim.viewer_camera
                camera.set_position_orientation(position=eye, orientation=quat)
                # OmniGibson camera frames may lag a camera-pose change. Warm
                # up only the first frame (no physics steps) then render once.
                if env_idx not in self._kadai_camera_warmed:
                    for _ in range(3):
                        og.sim.render()
                    self._kadai_camera_warmed.add(env_idx)
                else:
                    og.sim.render()
                camera_image = camera.get_obs()[0]['rgb']
                obs = instance_eval_state.obs
                names = self.robot_camera_names
                key = lambda role: names[role] + '::rgb'
                presentation_frame = compose_overview(
                    camera_image, obs[key('head')], obs[key('left_wrist')], obs[key('right_wrist')],
                    label='FIXED OVERHEAD CAMERA' if self._kadai_camera_mode == 'fixed_overhead' else 'REAR FOLLOW CAMERA'
                )
                write_video(np.expand_dims(presentation_frame, 0),
                            video_writer=self._kadai_overview[env_idx][1], batch_size=1, mode='rgb')
                self._kadai_frame_count[env_idx] += 1
            except Exception as exc:
                # The original challenge output must survive a viewer-camera failure.
                logger.warning('Third-person capture disabled (official video preserved): %s', exc)
                self._kadai_disabled = True
                self._discard_overview(env_idx, delete_file=True)

    official_eval.BatchedEvaluator = PresentationEvaluator
    official_eval.main()


if __name__ == '__main__':
    main()
