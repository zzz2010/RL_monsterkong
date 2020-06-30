import os, sys
from ple.games.pixelcopter import Pixelcopter as GameEnv
from ple import PLE
import parl
from parl.utils import logger
# from parl.algorithms.fluid import DQN as RL_Alg
import numpy as np
from config import *
# from cnn_dqn import  Model,Agent
from parl.algorithms.fluid import DDPG as RL_Alg
from DDPG import  Model,Agent
# from fc_dqn import  Model,Agent

dummy_mode=False ###check if the model can overfit simple linear reward function
if dummy_mode:
    max_frames=10000
    GAMMA=0.0001
    LEARNING_RATE=1e-6

def get_obs(p):
    obs = np.array(list(p.getGameState().values()))
    obs[2:4] -= obs[0]
    obs[5:7] -= obs[0]
    obs = (obs / np.array([48, 10, 24, 24, 48, 24, 24])) - 0.5
    return obs

def get_env_obs(ple_env,last_obs=None):
    obs=get_obs(ple_env)
    obs = ple_env.getScreenGrayscale()
    if last_obs is not None:
        return np.concatenate([last_obs[1:, :],obs[np.newaxis, :]])
    else:
        return np.concatenate([obs[np.newaxis, :],obs[np.newaxis, :]])

def get_dummy_reward(obs,action):
    if isinstance(action, int):
        np.random.seed(1234 + action)
    else:
        np.random.seed(1234)
    obs_flatten = obs.ravel()
    obs_flatten = obs_flatten / np.mean(np.square(obs_flatten))
    hidden_mat = np.random.randn(obs_flatten.shape[0])

    reward = np.clip(np.mean(obs_flatten * hidden_mat), -1, 1)
    return reward

def get_reward(ple_env,obs,action):
    reward = ple_env.act(action)
    if dummy_mode:
        reward=get_dummy_reward(obs,action)

        # print(action, np.max(hidden_mat),reward)


    return reward

def run_episode(ple_env, agent, rpm):
    total_reward = 0
    ple_env.reset_game()
    last_obs = None
    obs = get_env_obs(ple_env,last_obs)

    step = 0
    while step < max_frames:
        step += 1
        act_prob = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        action = ple_env.getActionSet()[np.random.choice(range(act_prob.shape[0]), p=act_prob) ]
        # 行动
        reward=get_reward(ple_env, obs, action)



        next_obs =  get_env_obs(ple_env,obs)
        done = ple_env.game_over()
        rpm.append((obs, act_prob, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward


        obs = next_obs
        if done:
            break


    return total_reward


import os.path
import numpy as np
from PIL import Image
def numpy2pil(np_array: np.ndarray)  :
    """
    Convert an HxWx3 numpy array into an RGB Image
    """

    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, 'RGB')
    return img

best_test_score=-100000000
# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(ple_env, agent, render=False):
    global best_test_score
    eval_reward = []

    for i in range(5):
        ple_env.reset_game()
        last_obs = None
        obs = get_env_obs(ple_env,last_obs)
        last_obs=obs
        episode_reward = 0
        step=0
        image_list=[]
        while step<max_frames:
            step+=1
            action_index = np.argmax(agent.predict(obs))  # 选取最优动作

            ### use this line to get the maximum reward in the dummy mode
            # action_index=np.argmax([get_dummy_reward(obs,ple_env.getActionSet()[action_index]) for action_index in range(agent.act_dim)])

            action = ple_env.getActionSet()[action_index]
            reward = get_reward(ple_env, obs, action)
            obs = get_env_obs(ple_env, last_obs)
            last_obs = obs
            episode_reward += reward
            if render:
                img = ple_env.getScreenRGB()
                image_list.append(numpy2pil(img).rotate(90).resize((300,300)))
            if ple_env.game_over():
                break
        if render and best_test_score<episode_reward:
            best_test_score=episode_reward
            image_list[0].save('pillow_imagedraw.gif',
                       save_all=True, append_images=image_list[1:], optimize=False, duration=40, loop=0)
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():
    global render_bool
    render_bool=True
    # parl.connect('localhost:8037')
    if dummy_mode:
        render_bool=False
    if not render_bool:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    # else:
    #     pygame.display.set_mode((800, 600 + 60))
    # 创建环境
    game = GameEnv()
    p = PLE(game, display_screen=render_bool, fps=10,
            force_fps=True)  # , fps=30, display_screen=render_bool, force_fps=True)


    p.init()



    # 根据parl框架构建agent
    print(p.getActionSet())
    act_dim = len(p.getActionSet())
    width, height = p.getScreenDims()
    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池
    obs_dim = 2, width, height
    model = Model(act_dim=act_dim)
    alg = RL_Alg(model,gamma=GAMMA, tau=0.001, actor_lr=LEARNING_RATE, critic_lr=LEARNING_RATE  )
    agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)  # e_greed有一定概率随机选取动作，探索

    # 加载模型
    best_eval_reward = -1000

    if os.path.exists('./model_pixelcopter.ckpt'):
        print("loaded model:", './model_pixelcopter.ckpt')
        agent.restore('./model_pixelcopter.ckpt')
        best_eval_reward = evaluate(p, agent, render=render_bool)
        # run_episode(env, agent, train_or_test='test', render=True)
        # exit()
    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(p, agent, rpm)

    max_episode = 200000
    # 开始训练
    episode = 0

    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part
        for i in range(0, 5):
            total_reward = run_episode(p, agent, rpm)
            episode += 1
        # test part
        eval_reward = evaluate(p, agent, render=render_bool)  # render=True 查看显示效果
        logger.info('episode:{}    e_greed:{}   test_reward:{}'.format(
            episode, e_greed, eval_reward))

        # 保存模型到文件 ./model.ckpt
        agent.save('./model_pixelcopter_%d.ckpt' % rate_num)
        if best_eval_reward < eval_reward:
            best_eval_reward = eval_reward
            agent.save('./model_pixelcopter.ckpt')


if __name__ == '__main__':
    main()
