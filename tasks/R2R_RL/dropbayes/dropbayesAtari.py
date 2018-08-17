import argparse
import os
import random
from collections import deque
from datetime import datetime

import logging
import torch
import gym
import numpy as np
import sys
import cv2

from torch import autograd, optim, nn
from torchvision import transforms as T

from tensorboardX import SummaryWriter

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

parser = argparse.ArgumentParser(description='Bayesian DQN Configuration')
parser.add_argument('--model', default='dqn', type=str, help='Test')
parser.add_argument('--prefix', default=datetime.today().strftime("%Yy%mm%dd_%HH%MM"), type=str)
parser.add_argument('--episode', default=None, type=int)
parser.add_argument('--game', default='Amidar-v0', type=str)
parser.add_argument('--record', default=True, action='store_true')
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--mode', default='train', type=str)
parser = parser.parse_args()

# Random Seed
torch.manual_seed(parser.seed)
torch.cuda.manual_seed(parser.seed)
np.random.seed(parser.seed)


class Environment(object):
    def __init__(self, game, record=False, width=84, height=84, seed=0):
        self.game = gym.make(game)
        self.game.seed(seed)
        # if record:
        #     self.game = Monitor(self.game, './video', force=True)

        self.width = width
        self.height = height
        self._toTensor = T.Compose([T.ToPILImage(), T.ToTensor()])

    # def play_sample(self, mode='human'):
    #     observation = self.game.reset()
    #
    #     while True:
    #         screen = self.game.render(mode=mode)
    #         if mode == 'rgb_array':
    #             screen = self.preprocess(screen)
    #         action = self.game.action_space.sample()
    #         observation, reward, done, info = self.game.step(action)
    #         if done:
    #             break
    #     self.game.close()

    def preprocess(self, screen):
        preprocessed = cv2.resize(screen, (self.height, self.width))  # 84 * 84 로 변경
        preprocessed = np.dot(preprocessed[..., :3], [0.299, 0.587, 0.114])  # Gray scale 로 변경
        # preprocessed: np.array = preprocessed.transpose((2, 0, 1))  # (C, W, H) 로 변경
        preprocessed = preprocessed.astype('float32') / 255.
        return preprocessed

    # def init(self):
    #     """
    #     @return observation
    #     """
    #     return self.game.reset()

    def get_screen(self):
        screen = self.game.render('rgb_array')
        screen = self.preprocess(screen)
        return screen

    def step(self, action: int):
        observation, reward, done, info = self.game.step(action)
        return observation, reward, done, info

    def reset(self):
        """
        :return: observation array
        """
        observation = self.game.reset()
        observation = self.preprocess(observation)
        return observation

    @property
    def action_space(self):
        return self.game.action_space.n

class DQNAgent():
    def __init__(self, parser, action_size):
        # self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_final = 0.001
        # self.epsilon_by_frame = lambda frame_idx: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)
        self.batch_size = 64
        self.train_start = 1000
        self.drop_p = 0.005

        self.memory = deque(maxlen=2000)
        if parser.model == 'dqn':
            Net = CNNQNet
        elif parser.model == 'dropdqn':
            Net = DropCNNQNet
        else:
            print("Model Argument Error")
            sys.exit()

        self.model = self.build_model(Net)
        if USE_CUDA :
            self.model.cuda()
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())

        self.target_model = self.build_model(Net)
        self.update_target_model()

        self.tau = 1.0
        self.lengthscale = 1e-2
        self.reg = self.lengthscale**2 * (1 - self.drop_p) / (2. * self.batch_size * self.tau)
        print("regression lambda : {}".format(self.reg))


    def build_model(self, Net):
        model = Net(self.action_size, self.drop_p)
        def weights_init(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.uniform_(m.bias.data)
        model.apply(weights_init)
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def epsilon_greedy(self, state):
        if np.random.rand() <= self.epsilon :
            return random.randrange(self.action_size)
        else :
            self.model.eval()
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.model(state)
            action = q_value.max(1)[1].data.cpu().numpy()[0]
            return action

    def thomson_sampling(self, state, mode="train"):
        if np.random.rand() <= self.epsilon :
            return random.randrange(self.action_size)
        else :
            if mode == "train":
                self.model.train()
            else:
                self.model.eval()
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.model(state)
            action = q_value.max(1)[1].data.cpu().numpy()[0]
            # action = action.data
            # action = action.cpu().numpy()
            # action = action[0]
        return action

    def greedy(self, state):
        self.model.eval()
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        q_value = self.model(state)
        action = q_value.max(1)[1].data.cpu().numpy()[0]
        # action = action.data
        # action = action.cpu().numpy()
        # action = action[0]
        return action

    def run_greedy_episode(self, env, state):
        score = 0
        done = False
        while not done:
            action = self.greedy(state)
            state, reward, done, info = env.step(action)
            reward = reward if not done or score == 499 else -100
            score += reward
        score = score if score == 500 else score + 100
        return score

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.memory, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    # def get_epsilon(self):
    #     return

    def train_model(self):
        self.model.train()
        if self.epsilon > self.epsilon_final :
            self.epsilon *= self.epsilon_decay

        states, actions, rewards, next_states, dones = self.sample(self.batch_size)
        # mini_batch = random.sample(self.memory, self.batch_size)

        states = Variable(torch.FloatTensor(np.float32(states)))
        next_states = Variable(torch.FloatTensor(np.float32(next_states)), volatile=True)
        actions = Variable(torch.LongTensor(actions))
        rewards = Variable(torch.FloatTensor(rewards))
        dones = Variable(torch.FloatTensor(dones))
        # states = np.zeros((self.batch_size, self.state_size))
        # next_states = np.zeros((self.batch_size, self.state_size))
        # actions, rewards, dones = [], [], []
        # for i in range(self.batch_size):
        #     states[i] = mini_batch[i][0]
        #     actions.append(mini_batch[i][1])
        #     rewards.append(mini_batch[i][2])
        #     next_states[i] = mini_batch[i][3]
        #     dones.append(mini_batch[i][4])
        # states = Variable(torch.FloatTensor(states))
        # next_states = Variable(torch.FloatTensor(next_states))

        q_values = self.model(states)
        next_q_values = self.model(next_states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + self.discount_factor * next_q_value * (1 - dones)
        # target = self.model(states).data.numpy()
        # target_val = self.model(next_states).data.numpy()
        #
        # for i in range(self.batch_size):
        #     if dones[i]:
        #         target[i][actions[i]] = rewards[i]
        #     else:
        #         target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_val[i])
        # expected_q_value = Variable(torch.FloatTensor(target))
        # q_value = self.model(states)

        # loss = self.criterion(outputs, target)
        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        for W in self.model.parameters():
            loss += self.reg * W.norm(2)
        # loss = F.smooth_l1_loss(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.model.parameters():
        #     print(param.grad)
        self.optimizer.step()

        return loss

class CNNQNet(nn.Module):
    def __init__(self, output_size, drop_p):
        super(CNNQNet, self).__init__()

        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        return self.layers(x)

class DropCNNQNet(nn.Module):
    def __init__(self, output_size, drop_p):
        super(DropCNNQNet, self).__init__()
        self.output_size = output_size

        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    # 환경 준비
    env = Environment(parser.game, record=False, seed=parser.seed)
    # state_size = env.observation_space.shape[0]
    action_size = env.action_space

    # 실험 저장 폴더 생성
    exp_dirpath = './exp/{}_{}_{}'.format(parser.model, parser.game, parser.prefix)
    print('Experiment save to ', exp_dirpath)
    if not os.path.exists(exp_dirpath):
        os.makedirs(exp_dirpath)

    # Tensorboard 준비
    run_dirpath = './runs/{}_{}_{}'.format(parser.model, parser.game, parser.prefix)
    writer = SummaryWriter(log_dir=run_dirpath)

    # Logging
    logger = logging.getLogger('DropBayesDQN')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler('./exp/{}_{}_{}.log'.format(parser.model, parser.game, parser.prefix))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # DQN 에이전트 생성
    agent = DQNAgent(parser, action_size)

    scores, episodes = [], []
    ratio_epsilons, ratio_thomsons = [], []

    for e in range(parser.episode):
        done = False
        score = 0
        loss = float("inf")
        # env 초기화
        state = env.reset()
        # state = np.reshape(state, [1, state_size])
        ratio_thomson = 0
        ratio_epsilon = 0
        cnt_frame = 0

        while not done:
            # if parser.mode == 'play':
            #     env.render()

            # 현재 상태로 행동을 선택
            action_thomson = agent.thomson_sampling(state)
            action_epsilon = agent.epsilon_greedy(state)
            action_greedy = agent.greedy(state)
            if action_thomson == action_greedy : ratio_thomson += 1
            if action_epsilon == action_greedy : ratio_epsilon += 1
            cnt_frame += 1

            action = action_thomson #if e < 200 else action_greedy
            # action = action_epsilon

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observation, reward, done, info = env.step(action)
            next_state = env.get_screen()
            # next_state = np.reshape(next_state, [1, state_size])

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.push(state, action, reward, next_state, done)

            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                loss = agent.train_model()

            score += reward
            state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()

                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)

                # Greedy policy 기준으로 한 episode를 돌림
                state = env.reset()
                greedy_score = agent.run_greedy_episode(env, state)

                print("episode:", e, "  score:", score, "  greedy score:", greedy_score, "  loss:", loss, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)
                ratio_thomson = ratio_thomson/float(cnt_frame)
                ratio_epsilon = ratio_epsilon/float(cnt_frame)
                ratio_thomsons.append(ratio_thomson)
                ratio_epsilons.append(ratio_epsilon)
                print("ratio_thomson:", ratio_thomson, "  ratio_epsilon:", ratio_epsilon)

                # 에피소드마다 결과 tensorboard에 저장
                writer.add_scalar('data/loss', loss, e)
                writer.add_scalar('data/score', score, e)
                writer.add_scalar('data/greedy_score', score, e)
                writer.add_scalar('data/memlength', len(agent.memory), e)
                writer.add_scalar('data/epsilon', agent.epsilon, e)
                # writer.add_scalar('data/ratios', {'thomson':ratio_thomson, 'epsilon':ratio_epsilon}, e)
                writer.add_scalar('data/ratio_thomson', ratio_thomson, e)
                writer.add_scalar('data/ratio_epsilon', ratio_epsilon, e)
                # params = agent.model.parameters()
                for name, param in agent.model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), e)

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                # if np.mean(scores[-min(10, len(scores)):]) > 490:
                    # agent.model.save_weights("./save_model/cartpole_dqn.h5")
                    # sys.exit()

    with open(exp_dirpath + "/scores.txt", 'w') as f:
        for item in scores:
            f.write("%s\n" % item)
    with open(exp_dirpath + "/ratio_thomsons.txt", 'w') as f:
        for item in ratio_thomsons:
            f.write("%s\n" % item)
    with open(exp_dirpath + "/ratio_epsilons.txt", 'w') as f:
        for item in ratio_epsilons:
            f.write("%s\n" % item)