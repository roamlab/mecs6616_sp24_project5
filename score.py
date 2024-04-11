import numpy as np
import time
from geometry import polar2cartesian


def test_policy(policy, env, goal):
    env.reset()
    env.set_goal(goal)
    env.arm.reset()  # force arm to be in vertical configuration
    # import ipdb; ipdb.set_trace()
    obs, rewards, done, info = env.step(env.action_space.sample() * 0)
    while True:
        action, _states = policy.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        if done:
            break
    state = env.arm.get_state()
    pos_ee = env.arm.dynamics.compute_fk(state)
    dist = np.linalg.norm((pos_ee - goal))
    if dist < 0.05:
        return 1.5
    elif dist < 0.1:
        return 1
    else:
        return 0


def score_policy(policy, env):
    print("\n--- Computing score ---")
    score = 0

    goal = polar2cartesian(1.8, 0.2 - np.pi / 2.0)
    _score = test_policy(policy, env, goal)
    score += _score
    print(f"\nGoal 1: {_score}")

    goal = polar2cartesian(1.9, -0.15 - np.pi / 2.0)
    _score = test_policy(policy, env, goal)
    score += _score
    print(f"\nGoal 2: {_score}")

    goal = polar2cartesian(1.6, 0.25 - np.pi / 2.0)
    _score = test_policy(policy, env, goal)
    score += _score
    print(f"\nGoal 3: {_score}")

    goal = polar2cartesian(1.8, -0.25 - np.pi / 2.0)
    _score = test_policy(policy, env, goal)
    score += _score
    print(f"\nGoal 4: {_score}")

    goal = polar2cartesian(1.6, 0.45 - np.pi / 2.0)
    _score = test_policy(policy, env, goal)
    score += _score
    print(f"\nGoal 5: {_score}")

    print('\n\n---')
    print(f'Final score: {score}/7.5')
    print('---')
    return score

def run_episode(q_network, env, device, goal=None):

    obs = env.reset(goal)
    done = False
    episode_reward = 0
    while not done:
        action = q_network.select_discrete_action(obs, device)
        obs, reward, done, _ = env.step(q_network.action_discrete_to_continuous(action))
        episode_reward += reward
    return episode_reward

def random_episodes(q_network, env, device, args):
    print("Testing for 100 episodes with random goals")
    for episode in range(100):
        episode_reward = run_episode(q_network, env, device, args)
        print(f'\nepisode: {episode}, reward: {episode_reward}')


def test_episode(q_network, env, device, goal, easy_target, hard_target):
    episode_reward = run_episode(q_network, env, device, goal)
    print(f'Total reward: {episode_reward}')
    print(f'easy target: {easy_target}')
    print(f'hard target: {hard_target}')
    points = 0
    if (episode_reward > easy_target): points += 1
    if (episode_reward > hard_target): points += 0.5
    print(f'points: {points}')
    return points
        
def compute_score(q_network, env, device):
    print("---Computing score---")
    score = 0
    
    print("\nGoal 1:")
    goal = polar2cartesian(1.9, -0.25 - np.pi/2.0)
    score += test_episode(q_network, env, device, goal, -7, -5)
    
    print("\nGoal 2:")
    goal = polar2cartesian(1.6, 0.25 - np.pi/2.0)
    score += test_episode(q_network, env, device, goal, -7, -5)

    print("\nGoal 3:")
    goal = polar2cartesian(1.8, 0.3 - np.pi/2.0)
    score += test_episode(q_network, env, device, goal, -7, -5)

    print("\nGoal 4:")
    goal = polar2cartesian(1.5, 0.3 - np.pi/2.0)
    score += test_episode(q_network, env, device, goal, -7, -5)

    print("\nGoal 5:")
    goal = polar2cartesian(1.6, 0.40 - np.pi/2.0)
    score += test_episode(q_network, env, device, goal, -10, -7)

    print(f'\n\nFinal score: {score}')
    return score
