############################################################
## Packages ################################################
import copy
import itertools
import logging
import os
import random
from collections import namedtuple, deque
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler

from . import CFG

Transition = namedtuple('Transition',  ('state', 'action', 'next_state', 'reward'))

#################################################################
## DQN algorithm ################################################
class DQN(nn.Module):

    @classmethod
    def construct_path(_, iWorld, policy_type, algorithm):
        fname = f'{algorithm}_{policy_type}.torch'
        project_dir = os.getcwd().split('MARL')[0] + 'MARL\\'
        file_path = project_dir + f'results\\IDQN_W{iWorld}\\{fname}'
        return file_path

    @classmethod
    def load(_, iWorld, policy_type, algorithm, verbose=True):

        try:
            file_path = DQN.construct_path(iWorld, policy_type, algorithm)
            q = torch.load(file_path)
        except:
            logging.warning('Inverted policy and algorithm name')
            file_path = DQN.construct_path(iWorld, algorithm, policy_type)
            q = torch.load(file_path)
        file_size = os.path.getsize(file_path)
        if verbose: print(
            f'\nLOADED EXISTING QNet \n\t| Path: {file_path} \n\t| Size: {file_size} \n\t| Device: {q.tensor_type["device"]}')
        return q

    @classmethod
    def save(_, module, iWorld, algorithm, policy_type, axes = None,verbose=True):
        if axes is not None: module.axes = axes #copy.deepcopy(axes)
        file_path = DQN.construct_path(iWorld, policy_type, algorithm)
        torch.save(module, file_path)
        file_size = os.path.getsize(file_path)
        if verbose: print(f'\nQNet SAVED \n\t| Path: {file_path} \n\t| Size: {round(file_size / 1000)}MB')

    @classmethod
    def preview(_, iWorld, policy_type, algorithm):
        q = DQN.load(iWorld, policy_type, algorithm)
        assert q.axes is not None, 'tried previewing DQN figure that does not exist'
        plt.ioff()
        plt.show()

        # if verbose: print(
        #     f'\nLOADED EXISTING QNet \n\t| Path: {file_path} \n\t| Size: {file_size} \n\t| Device: {q.tensor_type["device"]}')
        # return q

    def __init__(self,run_config=None):
        super(DQN, self).__init__()
        this_CFG = run_config if run_config is not None else copy.deepcopy(CFG)
        self.n_jointA =  this_CFG.Environment.n_jointA
        self.n_egoA = this_CFG.Environment.n_egoA
        self.n_obs = this_CFG.Environment.n_obs
        self.n_agents =  this_CFG.Environment.n_agents
        self.rationality =  this_CFG.Environment.rationality
        self.ToM = this_CFG.Environment.ToM
        self.tensor_type = {'device': this_CFG.torch.device, 'dtype': this_CFG.torch.dtype}
        self.run_config = this_CFG
        self.env = None

        # Precalculate lookups for efficiency
        self.ijoint, self.solo2joint, self.joint2solo = self.init_Alookup_mats()

        # Create Q-Net
        for k in range(self.n_agents):
            setattr(self, 'agent_{}'.format(k),
                    nn.Sequential(nn.Linear(self.n_obs, 128), nn.ReLU(),
                                  nn.Linear(128, 128), nn.ReLU(),
                                  nn.Linear(128, self.n_jointA))
                    )

    def init_Alookup_mats(self):
        joint2solo = np.array(list(itertools.product(*[np.arange(5), np.arange(5)])), dtype=int)
        solo2joint = np.zeros([5, 5], dtype=int)
        for aJ, joint_action in enumerate(joint2solo):
            aR, aH = joint_action
            solo2joint[aR, aH] = aJ
        ijoint = np.zeros([2, 5, 25], dtype=np.float32)
        for k in range(self.n_agents):
            for ak in range(5):
                idxs = np.array(np.where(joint2solo[:, k] == ak)).flatten()
                ijoint[k, ak, idxs] = 1
        ijoint = torch.as_tensor(ijoint, **self.tensor_type)
        solo2joint = torch.as_tensor(solo2joint, **self.tensor_type)
        joint2solo = torch.as_tensor(joint2solo, **self.tensor_type)
        return ijoint,solo2joint,joint2solo


    def forward(self, obs):
        q_values = [getattr(self, 'agent_{}'.format(k))(obs).unsqueeze(1) for k in range(self.n_agents)]
        return torch.cat(q_values, dim=1)

    def QRE(self,qAk,get_pd = True,get_q=False):
        """
        Iterate through recursive sophistication (ToM) levels for both agents.
        << can be improved using caching >>
        Vars:
        pdAjointk:  prob of agent k taking joint action a_ij
        qAego:      quality of agent k taking controllable action a_i
        pdAegok:    prob of agent k taking controllable action a_i
        """
        ToM         = self.ToM
        n_agents    = self.n_agents
        n_joint_act = self.n_jointA
        n_egoA      = self.n_egoA
        rationality = self.rationality
        n_batch     = qAk.shape[0]

        pdAjointk   = torch.ones([n_batch, n_agents, n_joint_act], **self.tensor_type) / n_joint_act
        qAego       = torch.empty([n_batch, n_agents, n_egoA], **self.tensor_type)
        pdAegok     = torch.empty([n_batch, n_agents, n_egoA], **self.tensor_type)
        for isoph in range(ToM):
            new_pdAjointk = torch.zeros([n_batch, n_agents, n_joint_act]) # temp pdAjointk (!! IMPORTANT !!)
            for k in range(n_agents):
                ijoint_batch = self.ijoint[k, :, :].T.repeat([n_batch, 1, 1]) # ego lookup
                qAJ_conditioned = qAk[:, k, :] * pdAjointk[:, int(not k), :] # condition on belief of partner on joint action
                qAego[:, k, :] = torch.bmm(qAJ_conditioned.unsqueeze(1), ijoint_batch).squeeze() # expected val over controllable actions (joint=>ego)
                pdAegok[:, k, :] = torch.special.softmax(rationality * qAego[:, k, :], dim=-1) # inference given ego rationality
                new_pdAjointk[:, k, :] = torch.bmm(pdAegok[:, k, :].unsqueeze(1), torch.transpose(ijoint_batch, dim0=1, dim1=2)).squeeze() # (ego=>joint)
            pdAjointk = new_pdAjointk.clone().detach()  # update pdAjointk
        if get_pd and get_q: return pdAegok,qAego
        elif get_pd:  return pdAegok#.detach().long()  # pdAjointk
        elif get_q: return qAego
        else: raise Exception('Unknown QRE parameter')

    def sample_best_action(self,obs,agent=2):
        """
        pAnotk: probability of partner (-k) actions given k's MM of -k
        Qk_exp: expected quality of k controllable action
        """
        iR, iH, iBoth = 0, 1, 2
        if agent == 0: kslice = 0
        elif agent == 1: kslice = 1
        elif agent == 2: kslice = slice(0,2)
        else: raise Exception('unknown agent sampling')

        n_batch = obs.shape[0]
        ak = torch.empty([n_batch, self.n_agents], dtype=torch.int64)
        aJ = torch.zeros([n_batch, 1], device=self.tensor_type['device'], dtype=torch.int64)
        Qk_exp = torch.zeros([n_batch, self.n_agents], **self.tensor_type)
        pAnotk = torch.empty([n_batch, self.n_agents, self.n_egoA], **self.tensor_type)

        with torch.no_grad():  # <=== CAUSED MEMORY ERROR WITHOUT ===
            qAjointk = self.forward(obs) # agent k's quality over joint actions
            pdAegok, qegok = self.QRE(qAjointk, get_pd=True, get_q=True)

            for ibatch in range(n_batch):
                # Each agen chooses the best controllable action => aJ = a_k X a_-k
                # for themselves conditioned on partner action
                aR = torch.argmax(pdAegok[ibatch, iR, :])
                aH = torch.argmax(pdAegok[ibatch, iH, :])
                aJ[ibatch] = self.solo2joint[aR, aH] # lookup table

                # Store batch stats
                ak[ibatch,iR] = aR
                pAnotk[ibatch,iR,:] = pdAegok[ibatch, int(not (iR)), :]
                Qk_exp[ibatch, iR ] = torch.sum(qegok[ibatch, iR] * pdAegok[ibatch, iR, :])

                ak[ibatch,iH] = aH
                pAnotk[ibatch, iH,:] = pdAegok[ibatch, int(not (iH)), :]
                Qk_exp[ibatch, iH ] = torch.sum(qegok[ibatch, iH] * pdAegok[ibatch, iH, :])

                # Assume perfect coordination (Pareto) (!! UNTESTED !!!)
                # aJ = torch.argmax(torch.mean(qAjointk[ibatch,:,:],dim=1))
                # aR,aH = self.joint2solo[aJ] # lookup table

        return aJ[:,kslice], Qk_exp[:,kslice]

    def sample_action(self, obs, epsilon, agent=2):
        """
        (ToM) Sophistocation 0: n/a
        (ToM) Sophistocation 1: I assume you move with uniform probability
        (ToM) Sophistocation 2: I assume that (you assume I move with uniform probability)
        (ToM) Sophistocation 3: I assume that [you assume that (I assume you move with uniform probability)]
        :param obs:
        :return:
        """
        iR, iH, iBoth = 0, 1, 2
        if agent == 0:    kslice = 0
        elif agent == 1:  kslice = 1
        elif agent == 2:  kslice = slice(0, 2)
        else: raise Exception('unknown agent sampling')

        n_batch = obs.shape[0]
        ak = torch.empty([n_batch, self.n_agents], dtype=torch.int64)
        aJ = torch.zeros([n_batch, 1], device=self.tensor_type['device'], dtype=torch.int64)
        Qk_exp = torch.zeros([n_batch, self.n_agents], **self.tensor_type)
        pAnotk = torch.empty([n_batch, self.n_agents, self.n_egoA], **self.tensor_type)

        if torch.rand(1) < epsilon:
            aJ = torch.randint(0, self.n_jointA, [n_batch, 1], device=self.tensor_type['device'], dtype=torch.int64)
        else:
            with torch.no_grad():  # <=== CAUSED MEMORY ERROR WITHOUT ===
                qAjointk = self.forward(obs)  # agent k's quality over joint actions
                pdAegok, qegok = self.QRE(qAjointk, get_pd=True, get_q=True)

                for ibatch in range(n_batch):
                    # Noisy rational sample both agent actions
                    aR = list(WeightedRandomSampler(pdAegok[ibatch, 0, :], 1, replacement=True))[0]
                    aH = list(WeightedRandomSampler(pdAegok[ibatch, 1, :], 1, replacement=True))[0]
                    aJ[ibatch] = self.solo2joint[aR, aH]

                    # Store batch stats
                    ak[ibatch, iR] = aR
                    pAnotk[ibatch, iR, :] = pdAegok[ibatch, int(not (iR)), :]
                    Qk_exp[ibatch, iR] = torch.sum(qegok[ibatch, iR] * pdAegok[ibatch, iR, :])

                    ak[ibatch, iH] = aH
                    pAnotk[ibatch, iH, :] = pdAegok[ibatch, int(not (iH)), :]
                    Qk_exp[ibatch, iH] = torch.sum(qegok[ibatch, iH] * pdAegok[ibatch, iH, :])

        if agent == iBoth:  return aJ
        else:               return ak[:,kslice], pAnotk[:,kslice]

    def simulate(self,env,epsilon):
        observations = []
        with torch.no_grad():
            state = env.reset()  # Initialize the environment and get it's state
            for t in itertools.count():
                action = self.sample_action(state, epsilon)
                next_state, reward, done, _ = env.step(action.squeeze())
                if done: observations.append([state, action, None, reward]); break
                else: observations.append([state, action, next_state, reward])
                state = next_state.clone().detach()
        return observations




#################################################################
## Auxiliary DQN functions ######################################
def test_policy(env, num_episodes, policy_net):
    with torch.no_grad():
        length  = 0
        psucc   = 0
        score   = np.zeros(env.n_agents)
        for episode_i in range(num_episodes):
            state = env.reset()
            for t in count():
                action = policy_net.sample_action(state, epsilon=0)
                next_state, reward, done, _ = env.step(action.squeeze())
                score += reward.detach().flatten().cpu().numpy()
                state = next_state.clone()
                if done: break
            if env.check_caught(env.current_positions): psucc +=1
            length += env.step_count
    final_score     = list(score/ num_episodes)
    final_length    = length/num_episodes
    final_psucc     = psucc/ num_episodes
    return final_score,final_length,final_psucc


def soft_update(policy_net,target_net,TAU):
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize_model(policy_net,target_net,optimizer,
                   memory,GAMMA,BATCH_SIZE,
                   update_iterations=1):
    if len(memory) < 2*BATCH_SIZE: return
    n_agents = 2
    losses = []
    for _ in range(update_iterations):
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=policy_net.tensor_type['device'], dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)#.to(policy_net.tensor_type['device'])
        action_batch = torch.cat(batch.action)#.to(policy_net.tensor_type['device'])
        reward_batch = torch.cat(batch.reward).reshape([-1,2]) #.to(policy_net.tensor_type['device'])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # state_action_values = policy_net(state_batch).gather(1, action_batch)
        state_action_values = policy_net(state_batch).gather(2, action_batch.unsqueeze(1).repeat([1, 2, 1]))


        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros([BATCH_SIZE,n_agents], device=policy_net.tensor_type['device'])
        with torch.no_grad():
            aJ, qA_sprime = target_net.sample_best_action(non_final_next_states)
            next_state_values[non_final_mask] = qA_sprime.squeeze()
            # next_state_values[non_final_mask] = qA_sprime

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(-1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
        losses.append(loss.item())

    ## LR Scheduler is more complicated than its worth
    # # mean_loss = loss.item()
    # if lr_scheduler is not None:
    #     mean_loss = sum(losses)/len(losses)
    #     lr_scheduler.step(mean_loss)


###############################################################################
################# !! DEPRECATED !! ############################################
###############################################################################
# def optimize_model(policy_net,target_net,optimizer, memory,GAMMA,BATCH_SIZE):
#     if len(memory) < BATCH_SIZE: return
#     n_agents = 2
#     transitions = memory.sample(BATCH_SIZE)
#     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation). This converts batch-array of Transitions
#     # to Transition of batch-arrays.
#     batch = Transition(*zip(*transitions))
#
#     # Compute a mask of non-final states and concatenate the batch elements
#     # (a final state would've been the one after which simulation ended)
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
#                                   device=policy_net.tensor_type['device'], dtype=torch.bool)
#     non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
#
#     #
#     # non_final_mask = torch.tensor(tuple(map(lambda _done: not _done, batch.next_state)),
#     #                               device=policy_net.tensor_type['device'], dtype=torch.bool)
#     # non_final_next_states = torch.cat([batch.next_state[si] for si in batch.next_state.numel() if not batch.done[si]])
#
#
#     state_batch = torch.cat(batch.state)#.to(policy_net.tensor_type['device'])
#     action_batch = torch.cat(batch.action)#.to(policy_net.tensor_type['device'])
#     reward_batch = torch.cat(batch.reward).reshape([-1,2]) #.to(policy_net.tensor_type['device'])
#
#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken. These are the actions which would've been taken
#     # for each batch state according to policy_net
#     # state_action_values = policy_net(state_batch).gather(1, action_batch)
#     state_action_values = policy_net(state_batch).gather(2, action_batch.unsqueeze(1).repeat([1, 2, 1]))
#
#
#     # Compute V(s_{t+1}) for all next states.
#     # Expected values of actions for non_final_next_states are computed based
#     # on the "older" target_net; selecting their best reward with max(1)[0].
#     # This is merged based on the mask, such that we'll have either the expected
#     # state value or 0 in case the state was final.
#     next_state_values = torch.zeros([BATCH_SIZE,n_agents], device=policy_net.tensor_type['device'])
#     with torch.no_grad():
#         # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
#         aJ = target_net.sample_action(non_final_next_states,epsilon=0)
#         qA_sprime = target_net(non_final_next_states)
#         next_state_values[non_final_mask] = qA_sprime.gather(2,aJ.unsqueeze(1).repeat(1,2,1)).squeeze()
#
#
#     # Compute the expected Q values
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#
#     # Compute Huber loss
#     criterion = nn.SmoothL1Loss()
#     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(-1))
#
#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     # In-place gradient clipping
#     torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
#     optimizer.step()