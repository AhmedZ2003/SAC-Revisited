from collections import deque
import warnings
import os
import cv2
import numpy as np

# Prefer gymnasium if installed (we'll adapt automatically)
try:
    import gymnasium as gym
    _HAS_GYMNASIUM = True
except Exception:
    import gym
    _HAS_GYMNASIUM = False

# Optional fast env provider
try:
    import envpool
except Exception:
    envpool = None

# vector env from tianshou (ShmemVectorEnv is used for speed; fallback to DummyVectorEnv)
from tianshou.env import ShmemVectorEnv, DummyVectorEnv

# Minimal runtime-safe helper (DO NOT call env.reset() at import time)
def _is_running_on_windows():
    return os.name == "nt"

# Wrappers (OpenAI baselines style)
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        try:
            assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        except Exception:
            pass

    def reset(self, *, seed=None, options=None):
        # handle gymnasium/gym reset signatures safely
        try:
            result = self.env.reset(seed=seed, options=options)
        except TypeError:
            result = self.env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        # RNG for env
        rng = getattr(self.unwrapped, "np_random", None)
        if rng is None:
            rng = np.random
        noops = rng.integers(1, self.noop_max + 1) if hasattr(rng, "integers") else rng.randint(1, self.noop_max + 1)
        for _ in range(noops):
            step_result = self.env.step(0)
            if len(step_result) == 5:
                obs, _, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, _, done, info = step_result
            if done:
                try:
                    result = self.env.reset(seed=seed, options=options)
                except TypeError:
                    result = self.env.reset()
                if isinstance(result, tuple):
                    obs, info = result
                else:
                    obs, info = result, {}
        return (obs, info) if _HAS_GYMNASIUM else obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        obs_list = []
        terminated = truncated = False
        info = {}
        for _ in range(self._skip):
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, term, trunc, info = result
                terminated = terminated or term
                truncated = truncated or trunc
            else:
                obs, reward, done, info = result
                terminated, truncated = done, False
            obs_list.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = np.max(obs_list[-2:], axis=0)
        return (max_frame, total_reward, terminated, truncated, info) if _HAS_GYMNASIUM else (max_frame, total_reward, (terminated or truncated), info)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 5:
            obs, reward, terminated, truncated, info = res
            done = terminated or truncated
        else:
            obs, reward, done, info = res
            terminated, truncated = done, False
        self.was_real_done = done
        try:
            lives = self.env.unwrapped.ale.lives()
        except Exception:
            lives = 0
        if 0 < lives < self.lives:
            # life lost -> early done for training
            if _HAS_GYMNASIUM:
                return obs, reward, True, False, info
            else:
                return obs, reward, True, info
        self.lives = lives
        return (obs, reward, terminated, truncated, info) if _HAS_GYMNASIUM else (obs, reward, done, info)

    def reset(self, *, seed=None, options=None):
        if self.was_real_done:
            try:
                result = self.env.reset(seed=seed, options=options)
            except TypeError:
                result = self.env.reset()
            if isinstance(result, tuple):
                obs, info = result
            else:
                obs, info = result, {}
        else:
            step_result = self.env.step(0)
            if len(step_result) == 5:
                obs, info = step_result[0], step_result[-1]
            else:
                obs, info = step_result[0], step_result[-1]
        try:
            self.lives = self.env.unwrapped.ale.lives()
        except Exception:
            self.lives = 0
        return (obs, info) if _HAS_GYMNASIUM else obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        try:
            assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        except Exception:
            pass

    def reset(self, *, seed=None, options=None):
        try:
            result = self.env.reset(seed=seed, options=options)
        except TypeError:
            result = self.env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        step_result = self.env.step(1)
        if len(step_result) == 5:
            obs, info = step_result[0], step_result[-1]
        else:
            obs, info = step_result[0], step_result[-1]
        return (obs, info) if _HAS_GYMNASIUM else obs


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width), dtype=np.uint8)

    def observation(self, frame):
        if frame is None:
            return frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=shp, dtype=np.float32)

    def observation(self, obs):
        return (obs.astype(np.float32)) / 255.0


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(k,) + shp, dtype=env.observation_space.dtype)

    def reset(self, *, seed=None, options=None):
        try:
            result = self.env.reset(seed=seed, options=options)
        except TypeError:
            result = self.env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        for _ in range(self.k):
            self.frames.append(obs)
        stacked = np.stack(self.frames, axis=0)
        return (stacked, info) if _HAS_GYMNASIUM else stacked

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 5:
            obs, reward, terminated, truncated, info = res
            done = terminated or truncated
            terminated_flag, truncated_flag = terminated, truncated
        else:
            obs, reward, done, info = res
            terminated_flag, truncated_flag = done, False
        self.frames.append(obs)
        stacked = np.stack(self.frames, axis=0)
        return (stacked, reward, terminated_flag, truncated_flag, info) if _HAS_GYMNASIUM else (stacked, reward, done, info)

# wrap_deepmind and adapter
def wrap_deepmind(env_or_id, episode_life=True, clip_rewards=True, frame_stack=4, scale=False, warp_frame=True):
    if isinstance(env_or_id, str):
        env = gym.make(env_or_id)
    else:
        env = env_or_id

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    try:
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
    except Exception:
        pass
    if warp_frame:
        env = WarpFrame(env, width=84, height=84)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, frame_stack)
    return env


class GymnasiumToGymAdapter(gym.Wrapper):
    """
    Adapter so code written against Gym's simple (obs, reward, done, info) API can
    work with Gymnasium (which uses (obs, info) on reset and returns (obs,reward,terminated,truncated,info)).
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *, seed=None, options=None):
        try:
            result = self.env.reset(seed=seed, options=options)
        except TypeError:
            result = self.env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        return obs

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 5:
            obs, reward, terminated, truncated, info = res
            done = terminated or truncated
            return obs, reward, done, info
        else:
            return res

    def seed(self, seed=None):
        try:
            self.env.reset(seed=seed)
        except Exception:
            pass

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


# make_atari_env - safe version
def _make_env_fn(task, episode_life, clip_rewards, frame_stack, scale, warp_frame):
    """
    Return a callable that, when invoked inside a child process, creates the env.
    This avoids creating environments at import time.
    """
    def _fn():
        base = gym.make(task)
        wrapped = wrap_deepmind(base, episode_life=episode_life, clip_rewards=clip_rewards,
                                frame_stack=frame_stack, scale=scale, warp_frame=warp_frame)
        if _HAS_GYMNASIUM:
            # adapt to gym API expected by old code
            return GymnasiumToGymAdapter(wrapped)
        return wrapped
    return _fn


def make_atari_env(task, seed=0, training_num=1, test_num=1, frame_stack=4, scale=False, clip_rewards=True, episode_life=True):
    """
    Create (single eval env, train_envs, test_envs).
    Windows-safety note: if running on Windows and training_num > 1, we force training_num=1
    to avoid multiprocessing deadlocks with ALE + shared memory.
    """
    # If envpool available, prefer it (fast & reliable)
    if envpool is not None and isinstance(task, str):
        if "NoFrameskip" in task:
            epool_id = task.replace("NoFrameskip-v4", "-v5")
        else:
            epool_id = task
        train_envs = envpool.make_gym(epool_id, num_envs=training_num, stack_num=frame_stack,
                                      episodic_life=episode_life, reward_clip=clip_rewards, seed=seed)
        test_envs = envpool.make_gym(epool_id, num_envs=test_num, stack_num=frame_stack,
                                     episodic_life=False, reward_clip=False, seed=seed + 100)
        env = train_envs  # envpool env acts like a vector env
        return env, train_envs, test_envs

    # Windows safety: avoid launching many child processes with ShmemVectorEnv
    if _is_running_on_windows() and training_num > 1:
        warnings.warn("Running on Windows: forcing training_num=1 to avoid multiprocessing issues with ALE. "
                      "If you need multi-process training, run inside WSL or Linux or use envpool.")
        training_num = 1

    # single eval
    single_base = gym.make(task)
    single_wrapped = wrap_deepmind(single_base, episode_life=episode_life, clip_rewards=clip_rewards,
                                   frame_stack=frame_stack, scale=scale, warp_frame=True)
    env = GymnasiumToGymAdapter(single_wrapped) if _HAS_GYMNASIUM else single_wrapped

    # vector envs: create factories so env.make happens inside worker process
    train_fns = [_make_env_fn(task, True, clip_rewards, frame_stack, scale, True) for _ in range(training_num)]
    test_fns = [_make_env_fn(task, False, False, frame_stack, scale, True) for _ in range(test_num)]

    # Choose ShmemVectorEnv when training_num==1 or on Linux. On Windows we fallback to DummyVectorEnv if more safety needed.
    try:
        if training_num == 1 and test_num == 1:
            train_envs = DummyVectorEnv(train_fns)
            test_envs = DummyVectorEnv(test_fns)
        else:
            train_envs = ShmemVectorEnv(train_fns)
            test_envs = ShmemVectorEnv(test_fns)
    except Exception:
        # Fallback to DummyVectorEnv if ShmemVectorEnv fails
        warnings.warn("ShmemVectorEnv failed to initialize, falling back to DummyVectorEnv (no multiprocessing).")
        train_envs = DummyVectorEnv(train_fns)
        test_envs = DummyVectorEnv(test_fns)

    # Attempt to seed
    try:
        env.reset(seed=seed)
    except Exception:
        try:
            env.seed(seed)
        except Exception:
            pass

    try:
        train_envs.seed(seed)
        test_envs.seed(seed + 100)
    except Exception:
        pass

    return env, train_envs, test_envs
