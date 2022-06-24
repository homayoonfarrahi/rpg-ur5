# import gym
# import gym_pygame
# import gym_minatar
# import gym_exploration
try:
  import pybullet
  import pybullet_envs
except ImportError:
  pass
from gym.wrappers.time_limit import TimeLimit

from envs.wrapper import *
from senseact.envs.ur.reacher_env_v2 import ReacherEnvV2
from senseact.utils import tf_set_seeds, NormalizedEnv

def make_env(env_name, max_episode_steps, episode_life=True):
  if env_name == 'UR5_2D_V2':
    # Problem
    random_state = np.random.get_state()

    # Create UR5 Reacher2D environment
    env = ReacherEnvV2(
        # setup="UR5_default",
        setup="UR5_2D_V2",
        host='129.128.159.210',
        dof=2,
        control_type="velocity",
        target_type="position",
        reset_type="zero",
        reward_type="precision",
        derivative_type="none",
        deriv_action_max=5,
        first_deriv_max=2,
        accel_max=1,
        speed_max=1.0,
        speedj_a=2.0,
        episode_length_time=4.0,
        episode_length_step=None,
        actuation_sync_period=1,
        dt=0.04,
        run_mode="multiprocess",
        rllab_box=False,
        movej_t=2.0,
        delay=0.0,
        random_state=random_state
    )
    env = NormalizedEnv(env)
    # # Start environment processes
    # env.seed(seed)
    # env.start()
  else:
    env = gym.make(env_name)
    env_group_title = get_env_group_title(env)
    # print(env_group_title, env_name)
    if env_group_title == 'gym_minatar':
      env = make_minatar(env, max_episode_steps, scale=False)
      if len(env.observation_space.shape) == 3:
        env = TransposeImage(env)
    elif env_group_title == 'atari' and '-ram' in env_name:
      make_atari_ram(env, max_episode_steps, scale=True)
    elif env_group_title == 'atari':
      env = make_atari(env, max_episode_steps)
      env = ReturnWrapper(env)
      env = wrap_deepmind(env,
                          episode_life=episode_life,
                          clip_rewards=False,
                          frame_stack=False,
                          scale=False)
      if len(env.observation_space.shape) == 3:
        env = TransposeImage(env)
      env = FrameStack(env, 4)
    elif env_group_title in ['classic_control', 'box2d', 'gym_pygame', 'gym_exploration', 'pybullet', 'mujoco', 'robotics']:
      if max_episode_steps > 0: # Set max episode steps
        env = TimeLimit(env.unwrapped, max_episode_steps)
  return env


def get_env_group_title(env):
  '''
  Return the group name the environment belongs to.
  Possible group name includes: 
    - gym: atari, algorithmic, classic_control, box2d, toy_text, mujoco, robotics, unittest
    - gym_ple 
    - gym_pygame
    - gym_minatar
    - gym_exploration
    - pybullet
  '''
  # env_name = env.unwrapped.spec.id
  s = env.unwrapped.spec.entry_point
  if 'gym_ple' in s:           # e.g. 'gym_ple:PLEEnv'
    group_title = 'gym_ple'
  elif 'gym_pygame' in s:      # e.g. 'gym_pygame.envs:CatcherEnv'
    group_title = 'gym_pygame'
  elif 'gym_minatar' in s:     # e.g. 'gym_minatar.envs:BreakoutEnv'
    group_title = 'gym_minatar'
  elif 'gym_exploration' in s: # e.g. 'gym_exploration.envs:NChainEnv'
    group_title = 'gym_exploration'
  elif 'pybullet' in s:        # e.g. 'pybullet_envs.gym_locomotion_envs:AntBulletEnv'
    group_title = 'pybullet'  
  elif 'gym' in s:             # e.g. 'gym.envs.classic_control:CartPoleEnv'
    group_title = s.split('.')[2].split(':')[0]
  else:
    group_title = None
  
  return group_title
