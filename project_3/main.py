import numpy as np
import environment


if __name__ == '__main__':
    env = environment.Soccer()
    observation = env.reset()

    env._pos_p0 = [1, 1]
    env._pos_p1 = [1, 2]
    env._ball = 1
    env.render()

    observation, rewards, done, info = env.step(3, 4)
    print(f'Actions: {[3, 4]}, Rewards: {rewards}, Done: {done}, Info: {info}')

    env.render()

    exit(1)

    done = False
    while not done:
        env.render()
        print(observation)
        a0 = np.random.randint(env.num_actions)
        a1 = np.random.randint(env.num_actions)
        observation, rewards, done, info = env.step(a0, a1)
        print(f'Actions: {[a0, a1]}, Rewards: {rewards}, Done: {done}')
        if done:
            env.render()
            print('DONE!')
