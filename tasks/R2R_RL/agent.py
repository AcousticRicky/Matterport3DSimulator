''' Agents: stop/random/shortest/seq2seq  '''

import json
import os
import sys
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.distributions as D
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
from utils import padding_idx
from model import EncoderLSTM
from eval import Evaluation
from actor_critic import EncoderHistory
from actor_critic import A2CAgent
from torch.distributions import Categorical
from collections import namedtuple

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.iteritems()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def rollout(self):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        #print 'Testing %s' % self.__class__.__name__
        looped = False
        while True:
            for traj in self.rollout():
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj['path']
            if looped:
                break

    
class StopAgent(BaseAgent):  
    ''' An agent that doesn't move! '''

    def rollout(self):
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in self.env.reset()]
        return traj


class RandomAgent(BaseAgent):
    ''' An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]

        self.steps = random.sample(range(-11,1), len(obs))
        ended = [False] * len(obs)
        for t in range(30):
            actions = []
            for i,ob in enumerate(obs):
                if self.steps[i] >= 5:
                    actions.append((0, 0, 0)) # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] < 0:
                    actions.append((0, 1, 0)) # turn right (direction choosing)
                    self.steps[i] += 1
                elif len(ob['navigableLocations']) > 1:
                    actions.append((1, 0, 0)) # go forward
                    self.steps[i] += 1
                else: 
                    actions.append((0, 1, 0)) # turn right until we can go forward
            obs = self.env.step(actions)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
        return traj


class ShortestAgent(BaseAgent):
    ''' An agent that always takes the shortest path to goal. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))
        while True:
            actions = [ob['teacher'] for ob in obs]
            obs = self.env.step(actions)
            for i,a in enumerate(actions):
                if a == (0, 0, 0):
                    ended[i] = True
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break
        return traj


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    env_actions = [
      (0,-1, 0), # left
      (0, 1, 0), # right
      (0, 0, 1), # up
      (0, 0,-1), # down
      (1, 0, 0), # forward
      (0, 0, 0), # <end>
      (0, 0, 0), # <start>
      (0, 0, 0)  # <ignore>
    ]
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.encoder = encoder
        self.decoder = decoder
        self.episode_len = episode_len
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.model_actions.index('<ignore>'))

    @staticmethod
    def n_inputs():
        return len(Seq2SeqAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Seq2SeqAgent.model_actions)-2 # Model doesn't output start or ignore

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1] # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(), \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        feature_size = obs[0]['feature'].shape[0]
        features = np.empty((len(obs),feature_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature']
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _teacher_action(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            ix,heading_chg,elevation_chg = ob['teacher']
            if heading_chg > 0:
                a[i] = self.model_actions.index('right')
            elif heading_chg < 0:
                a[i] = self.model_actions.index('left')
            elif elevation_chg > 0:
                a[i] = self.model_actions.index('up')
            elif elevation_chg < 0:
                a[i] = self.model_actions.index('down')
            elif ix > 0:
                a[i] = self.model_actions.index('forward')
            elif ended[i]:
                a[i] = self.model_actions.index('<ignore>')
            else:
                a[i] = self.model_actions.index('<end>')
        return Variable(a, requires_grad=False).cuda()

    def rollout(self):
        obs = np.array(self.env.reset())
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        # Initial action
        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'), 
                    requires_grad=False).cuda()
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        env_action = [None] * batch_size
        for t in range(self.episode_len):

            f_t = self._feature_variable(perm_obs) # Image features from obs
            h_t,c_t,alpha,logit = self.decoder(a_t.view(-1, 1), f_t, h_t, c_t, ctx, seq_mask)
            # Mask outputs where agent can't move forward
            for i,ob in enumerate(perm_obs):
                if len(ob['navigableLocations']) <= 1:
                    logit[i, self.model_actions.index('forward')] = -float('inf')             

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            self.loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher': 
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax': 
                _,a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logit, dim=1)
                m = D.Categorical(probs)
                a_t = m.sample()            # sampling an action from model
            else:
                sys.exit('Invalid feedback option')

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i]#.item()#data[0]
                if action_idx == self.model_actions.index('<end>'):
                    ended[i] = True
                env_action[idx] = self.env_actions[action_idx]

            obs = np.array(self.env.step(env_action))
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            # Early exit if all ended
            if ended.all(): 
                break

        self.losses.append(self.loss.data[0] / self.episode_len)
        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        super(Seq2SeqAgent, self).test()

    def train(self, encoder_optimizer, decoder_optimizer, n_iters, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        for iter in range(1, n_iters + 1):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            self.rollout()
            self.loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

    def save(self, encoder_path, decoder_path):
        ''' Snapshot models '''
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))


class ActorCriticAgent(BaseAgent):

    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    env_actions = [
        (0,-1, 0), # left
        (0, 1, 0), # right
        (0, 0, 1), # up
        (0, 0,-1), # down
        (1, 0, 0), # forward
        (0, 0, 0), # <end>
        (0, 0, 0), # <start>
        (0, 0, 0)  # <ignore>
    ]

    SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'step'])
    eps = np.finfo(np.float32).eps.item()

    def __init__(self, env, vocab_size, results_path, batch_size, episode_len=20):
        super(ActorCriticAgent, self).__init__(env, results_path)

        #For evaluation
        self.ev = Evaluation(['train'])

        #For navigation
        self.episode_len = episode_len
        self.losses = []

        ''' Define instruction encoder '''
        word_embedding_size = 256
        hidden_size = 512
        bidirectional = False
        dropout_ratio = 0.5

	enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
	self.encoder = EncoderLSTM(vocab_size, word_embedding_size, enc_hidden_size, padding_idx, dropout_ratio, bidirectional=bidirectional).cuda()

        context_size = 1024
        self.hist_encoder = EncoderHistory(len(self.model_actions), 32, 2048, context_size).cuda()
        self.a2c_agent = A2CAgent(enc_hidden_size, context_size, len(self.model_actions) - 2).cuda()
        self.saved_actions = []

        params = list(self.encoder.parameters()) + list(self.hist_encoder.parameters()) + list(self.a2c_agent.parameters())
	self.losses = []
        self.optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=1e-5)


    def _sort_batch(self, obs):
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1] # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(), \
               list(seq_lengths), list(perm_idx)


    def _feature_variable(self, obs):
        feature_size = obs[0]['feature'].shape[0]
        features = np.empty((len(obs),feature_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature']
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()


    def _teacher_action(self, obs, ended):
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            ix,heading_chg,elevation_chg = ob['teacher']
            if heading_chg > 0:
                a[i] = self.model_actions.index('right')
            elif heading_chg < 0:
                a[i] = self.model_actions.index('left')
            elif elevation_chg > 0:
                a[i] = self.model_actions.index('up')
            elif elevation_chg < 0:
                a[i] = self.model_actions.index('down')
            elif ix > 0:
                a[i] = self.model_actions.index('forward')
            elif ended[i]:
                a[i] = self.model_actions.index('<ignore>')
            else:
                a[i] = self.model_actions.index('<end>')
        return Variable(a, requires_grad=False).cuda()


    def rollout(self, guide_prob):
        #For navigation
        obs = np.array(self.env.reset())
        batch_size = len(obs)

        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'), requires_grad=False).cuda()

        ended = np.array([False] * len(obs))
        env_action = [None] * batch_size

        h_n, c_n = self.hist_encoder.init_hidden(batch_size)

        for t in range(self.episode_len):
            f_t = self._feature_variable(perm_obs)

            enc_data, h_n, c_n =self.hist_encoder(a_t, f_t, h_n, c_n)
            action_prob, critic_value = self.a2c_agent(ctx, seq_lengths, enc_data)

            guided = np.random.choice(2, batch_size, p=[1.0 - guide_prob, guide_prob])

            demo = self._teacher_action(perm_obs, ended)

            if guided[0] == 1:
                a_t = demo
            else:

                if len(perm_obs[0]['navigableLocations']) <= 1:
                    action_prob[0, self.model_actions.index('forward')] = -float('inf')

                action_prob = F.softmax(action_prob, dim=1)

                m = Categorical(action_prob)
                a_t = m.sample()
                if not ended[0]:
                    self.saved_actions.append(self.SavedAction(m.log_prob(a_t), critic_value, t))

            for i, (idx, ob) in enumerate(zip(perm_idx, perm_obs)):
                action_idx = a_t[i]
                if action_idx == self.model_actions.index('<end>'):
                    ended[i] = True
                env_action[idx] = self.env_actions[action_idx]

            obs = np.array(self.env.step(env_action))
            perm_obs = obs[perm_idx]

            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            if ended.all():
                break

        return traj


    def clear_saved_actions(self):
        del self.saved_actions[:]


    def test(self, guide_prob):
        self.encoder.eval()
        self.hist_encoder.eval()
        self.a2c_agent.eval()

	self.env.reset_epoch()
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        #print 'Testing %s' % self.__class__.__name__
        looped = False
        while True:
            for traj in self.rollout(guide_prob):
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj['path']
            if looped:
                break

        self.clear_saved_actions()


    def train(self, n_iters, guide_prob):
        self.encoder.train()
        self.hist_encoder.train()
        self.a2c_agent.train()

        policy_losses = []
        value_losses = []
	self.losses = []

        total_num = 0
        success_num = 0
        for iter in range(1, n_iters + 1):
            traj = self.rollout(guide_prob)
            for i, t in enumerate(traj):
                nav_error, oracle_error, trajectory_step, trajectory_length = self.ev._score_item(t['instr_id'], t['path'])
                reward = 1.0 if nav_error < 3.0 else 0.0

                total_num += 1.0
                success_num += reward

                for log_prob, value, step in self.saved_actions:
                    discounted_reward = pow(0.99, trajectory_step - step) * reward
                    advantage = discounted_reward - value.item()
                    policy_losses.append(-log_prob * advantage)
                    value_losses.append(F.smooth_l1_loss(value, Variable(torch.tensor([[discounted_reward]]).cuda(), requires_grad=False)))

            data_len = len(policy_losses)
            if data_len > 64:
                self.optimizer.zero_grad()
                value_loss = torch.stack(value_losses).sum()
                policy_loss = torch.stack(policy_losses).sum() 
                loss = value_loss + policy_loss
		self.losses.append(value_loss.item() / data_len)
		#print('sub iter [%d/%d], Average Value Loss: %.4f' %(iter, n_iters, value_loss.item() / data_len))
                loss.backward()
                self.optimizer.step()
                self.clear_saved_actions()
                policy_losses = []
                value_losses = []

        data_len = len(policy_losses)
        if data_len > 0:
            self.optimizer.zero_grad()
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
            self.losses.append(loss.item() / data_len)
            loss.backward()
            self.optimizer.step()
            self.clear_saved_actions()

        print('guide prob: %.2f, train value loss: %.4f, success: %.2f' % (guide_prob, np.average(np.array(self.losses)), (success_num / total_num)))
