from ple.games.snake import Snake as GameEnv
from ple import PLE
import parl
from parl import layers
import paddle.fluid as fluid
import numpy as np
import os, sys
from parl.utils import logger
from parl.algorithms import DQN
import random
import collections
import pygame
from resnet import ResNet

LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再开启训练
BATCH_SIZE = 164  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
rate_num = int(sys.argv[1])
LEARNING_RATE = 0.00001 * rate_num  # 学习率
GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
max_frames = 10000


class Model(parl.Model):
    def __init__(self, act_dim):
        num_layers = 18
        # ResNet
        # self.res = ResNet( depth=num_layers, num_classes=act_dim)

        # ResNet
        self.fc1 = layers.fc(size=100, act='relu')
        self.fc2 = layers.fc(size=64, act='relu')
        self.fc3 = layers.fc(size=16, act='relu')
        self.fc4 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        hid3 = self.fc3(hid2)
        Q = self.fc4(hid3)

        # Q = self.res(obs)
        return Q


class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):

        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=self.obs_dim, dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=self.obs_dim, dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=self.obs_dim, dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1
        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
               np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'), \
               np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)


def get_env_obs(ple_env):
    obs = ple_env.getScreenGrayscale()
    return obs[np.newaxis, :]


def run_episode(ple_env, agent, rpm):
    total_reward = 0
    ple_env.reset_game()
    obs = get_env_obs(ple_env)
    step = 0
    while step < max_frames:
        step += 1
        action_index = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        action = ple_env.getActionSet()[action_index]
        # 行动
        reward = ple_env.act(action)

        box_pos_tup=np.where(obs == 133)
        if len(box_pos_tup[2]) >1:
            box_pos_x,box_pos_y=box_pos_tup[1][1],box_pos_tup[2][1]
            snake_pos_tup=np.where(obs == 54)
            if len(snake_pos_tup[0])==0:
                snake_pos_tup = np.where(obs == 212)
            if len(snake_pos_tup[2]) > 1:
                snake_pos_x,snake_pos_y=snake_pos_tup[1][1],snake_pos_tup[2][1]
                distance=np.mean(np.square([snake_pos_x-box_pos_x,snake_pos_y-box_pos_y]))
                if distance>10:
                    aux_reward=-distance/10000
                    # reward=reward+aux_reward

        next_obs = get_env_obs(ple_env)
        done = ple_env.game_over()
        rpm.append((obs, action_index, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward

        # pygame.init()
        #
        # win = pygame.display.set_mode((500, 500))
        # img = ple_env.getScreenRGB()
        # main = pygame.image.frombuffer(img, img.shape[:2], "RGB")
        # main = pygame.transform.scale(main, (700, 500))
        # win.blit(main, (0, 0))
        obs = next_obs
        if done:
            break
    print(total_reward)

    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(ple_env, agent, render=False):
    eval_reward = []
    for i in range(5):
        ple_env.reset_game()
        obs = get_env_obs(ple_env)
        episode_reward = 0
        step=0
        while step<10000:
            step+=1
            action_index = agent.predict(obs)  # 选取最优动作
            action = ple_env.getActionSet()[action_index]
            reward = ple_env.act(action)
            obs = get_env_obs(ple_env)
            episode_reward += reward
            if render:
                img = ple_env.getScreenRGB()
            if ple_env.game_over():
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():
    render_bool = True
    if not render_bool:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    # else:
    #     pygame.display.set_mode((800, 600 + 60))
    # 创建环境
    game = GameEnv()
    p = PLE(game, display_screen=render_bool, fps=60,
            force_fps=False)  # , fps=30, display_screen=render_bool, force_fps=True)


    p.init()



    # 根据parl框架构建agent
    print(p.getActionSet())
    act_dim = len(p.getActionSet())
    width, height = p.getScreenDims()
    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池
    obs_dim = 1, width, height
    model = Model(act_dim=act_dim)
    alg = DQN(model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim, e_greed=0.5,
                  e_greed_decrement=0.00001)  # e_greed有一定概率随机选取动作，探索

    # 加载模型
    best_eval_reward = -1000

    if os.path.exists('./model_dqn.ckpt'):
        print("loaded model:", './model_dqn.ckpt')
        agent.restore('./model_dqn.ckpt')
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
            episode, agent.e_greed, eval_reward))

        # 保存模型到文件 ./model.ckpt
        agent.save('./model_dqn_%d.ckpt' % rate_num)
        if best_eval_reward < eval_reward:
            best_eval_reward = eval_reward
            agent.save('./model_dqn.ckpt')


if __name__ == '__main__':
    main()
