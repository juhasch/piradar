"""
Gym-compatible environment wrapper for PiRadar to support RL loops.

This keeps the control-plane (actions/parameter updates) separate from the
data-plane (frames) and offers a minimal, stable API to integrate with
Gymnasium/Stable-Baselines-style training code without imposing a hard dep.

Usage example (random agent):

    from piradar.hw import BGT60TR13C
    from piradar.rl_env import RadarRLEnv, ActionSpec

    radar = BGT60TR13C()
    env = RadarRLEnv(radar, action_spec=ActionSpec(
        params=[
            ("tx_power", 0, 31),
            ("ramp_time_up1", 0, 16383),
        ],
        normalize=True,
    ))
    obs, info = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

Notes:
 - Gymnasium is optional. If not installed, a small shim provides .action_space
   and .observation_space plus step/reset signatures compatible with Gymnasium.
 - Reward is user-defined via a callback; defaults to 0.0.
 - Parameter writes are applied via AdaptiveRadarController to ensure validation.
"""

import time
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .bgt60tr13c import BGT60TR13C
from .radar_controller import AdaptiveRadarController


try:
    import gymnasium as gym
    from gymnasium.spaces import Box
except ImportError:
    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

        def sample(self):
            low = np.array(self.low, dtype=self.dtype)
            high = np.array(self.high, dtype=self.dtype)
            return low + (high - low) * np.random.rand(*self.shape)

    class _DummyEnv:
        observation_space: _Box
        action_space: _Box

    gym = _DummyEnv()  # type: ignore
    Box = _Box  # type: ignore


RewardFn = Callable[[np.ndarray, Dict], float]


@dataclass
class ActionSpec:
    """Map a continuous action vector to named radar parameters.

    Each tuple is (param_name, min_value, max_value). When normalize=True, the
    agent actions are expected in [-1, 1] and are scaled into [min, max].
    """

    params: Sequence[Tuple[str, int, int]]
    normalize: bool = True


class RadarRLEnv:
    """A minimal Gym-like Env for running RL over PiRadar.

    - Observation: last `stack_size` frames stacked along axis 0, optionally
      reduced by a simple transform function.
    - Action: continuous Box mapped to integer register parameters via
      AdaptiveRadarController.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        radar: BGT60TR13C,
        action_spec: ActionSpec,
        stack_size: int = 1,
        frame_timeout_s: float = 1.0,
        reward_fn: Optional[RewardFn] = None,
        obs_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self._radar = radar
        self._ctrl = AdaptiveRadarController(radar)
        self._spec = action_spec
        self._stack = stack_size
        self._timeout = frame_timeout_s
        self._reward_fn = reward_fn or (lambda obs, info: 0.0)
        self._obs_transform = obs_transform

        # Action space
        if self._spec.normalize:
            low = np.full((len(self._spec.params),), -1.0, dtype=np.float32)
            high = np.full((len(self._spec.params),), 1.0, dtype=np.float32)
        else:
            lows = [p[1] for p in self._spec.params]
            highs = [p[2] for p in self._spec.params]
            low = np.array(lows, dtype=np.float32)
            high = np.array(highs, dtype=np.float32)
        self.action_space = Box(low=low, high=high, shape=low.shape, dtype=np.float32)

        # We will discover observation shape after first reset/frame
        self.observation_space: Box
        self._obs_shape: Optional[Tuple[int, ...]] = None
        self._rng = np.random.default_rng()
        self._step_idx = 0

        # Start radar if not running
        if not self._radar.is_running:
            self._radar.start()

    def _fetch_frame(self) -> np.ndarray:
        """Get a single frame from the radar buffer with timeout."""
        deadline = time.time() + self._timeout
        while time.time() < deadline:
            try:
                frame = self._radar.frame_buffer.get(timeout=0.05)
                return frame
            except queue.Empty:
                pass
        raise TimeoutError("No frame received within timeout")

    def _stack_frames(self) -> np.ndarray:
        frames: List[np.ndarray] = []
        for _ in range(self._stack):
            f = self._fetch_frame()
            frames.append(f)
        stacked = np.stack(frames, axis=0)
        return stacked

    def _ensure_obs_space(self, obs: np.ndarray) -> None:
        if self._obs_shape is None:
            o = self._obs_transform(obs) if self._obs_transform else obs
            self._obs_shape = o.shape
            low = np.full(self._obs_shape, -np.inf, dtype=np.float32)
            high = np.full(self._obs_shape, np.inf, dtype=np.float32)
            self.observation_space = Box(low=low, high=high, shape=self._obs_shape, dtype=np.float32)  # type: ignore

    def _apply_action(self, action: np.ndarray) -> Dict[str, int]:
        # Map action vector to parameter dict
        changes: Dict[str, int] = {}
        for i, (name, pmin, pmax) in enumerate(self._spec.params):
            v = float(action[i])
            if self._spec.normalize:
                # scale [-1, 1] -> [pmin, pmax]
                v = (v + 1.0) * 0.5  # [0,1]
                v = pmin + v * (pmax - pmin)
            v_int = int(round(max(pmin, min(pmax, v))))
            changes[name] = v_int
        # Apply via controller (validates ranges and writes registers)
        self._ctrl.batch_write_parameters(changes, source="rl_step")
        return changes

    # Gymnasium-style API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_idx = 0
        obs = self._stack_frames()
        self._ensure_obs_space(obs)
        obs = self._obs_transform(obs) if self._obs_transform else obs
        info = {"step": self._step_idx}
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray):
        self._step_idx += 1
        applied = self._apply_action(np.asarray(action, dtype=np.float32))
        obs = self._stack_frames()
        obs = self._obs_transform(obs) if self._obs_transform else obs
        info = {"step": self._step_idx, "applied": applied}
        reward = float(self._reward_fn(obs, info))
        terminated = False
        truncated = False
        return obs.astype(np.float32), reward, terminated, truncated, info

    def close(self) -> None:
        try:
            if self._radar.is_running:
                self._radar.stop()
        except (OSError, RuntimeError):
            pass

