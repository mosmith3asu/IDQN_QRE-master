import logging
import math
from itertools import count

import torch
from gym import Env
from torch.utils.data.sampler import WeightedRandomSampler

from .learning_utils import CPT_Handler
from .make_worlds import WorldDefs
from .. import CFG


class PursuitEvastionGame(Env):

    def __init__(self,iWorld,device,dtype,seed=None):
        world = WorldDefs.world[iWorld]
        # self.device = device  self.dtype = torch.float32

        self.r_catch = CFG.Environment.r_catch
        self.r_penalty = CFG.Environment.r_penalty
        self.p_penalty = CFG.Environment.p_penalty
        self.n_moves = CFG.Environment.n_moves
        self.prey_rationality = CFG.Environment.prey_rationality
        self.prey_dist_power = CFG.Environment.prey_dist_power
        self.enable_penalty = CFG.Environment.enable_penalty
        self.enable_prey_move = CFG.Environment.enable_prey_move
        self.save_rewards_for_end = CFG.Environment.save_rewards_for_end
        self.enable_rand_init = CFG.Environment.enable_rand_init
        self.n_jointA = CFG.Environment.n_jointA
        self.n_egoA = CFG.Environment.n_egoA
        self.n_agents = CFG.Environment.n_agents
        self.rationality = CFG.Environment.rationality
        self.ToM = CFG.Environment.ToM
        self.n_obs = CFG.Environment.n_obs
        self.grid_sz = CFG.Environment.grid_sz


        # Define constant parameters
        # self.r_catch = 20
        # self.r_catch = 25; logging.warning(f'R catch modified = {self.r_catch}')
        # self.r_penalty = -3
        # self.p_penalty = 0.5
        # self.n_moves = 20
        # self.prey_rationality = 1
        # self.prey_dist_power = 2
        # self.enable_penalty = True
        # self.enable_prey_move = True
        # self.enable_rand_init = False
        # self.n_joint_actions = 25
        # self.n_ego_actions = 5
        # self.n_agents = 2



        self.tensor_type = {'device':device,'dtype':dtype}
        self.tensor_type_idx = {'device': device, 'dtype': torch.long}
        self.scale_rcatch = 1.0
        self.scale_penalty = 1.0


        grid_shape           = torch.tensor(world.grid_shape,**self.tensor_type)
        self._all_walls      = torch.tensor(world.walls, **self.tensor_type)
        self._walls          = torch.tensor([[2., 2.], [2., 4.], [4., 2.], [4., 4.]], **self.tensor_type)
        self._start_obs      = torch.tensor(world.start_obs, **self.tensor_type)
        self._penalty_states = torch.tensor(world.penalty_states, **self.tensor_type)
        self._max_dist       = torch.dist(torch.zeros(2,**self.tensor_type), grid_shape)
        self._max_pos        = 5
        self._min_pos        = 1

        self.AGENT_COLOR    = world.AGENT_COLOR
        self.PREY_COLOR     = world.PREY_COLOR
        self.CELL_SIZE      = world.CELL_SIZE
        self.WALL_COLOR     = world.WALL_COLOR
        self.PEN_COLOR      = world.PEN_COLOR

        # Define Conversion Functions
        self.action2idx = {'down': 0, 'left': 1, 'up': 2, 'right': 3, 'wait': 4}
        self.idx2action = {v: k for k, v in self.action2idx.items()}
        self.explicit_actions = {'down':  torch.tensor([1, 0], **self.tensor_type),
                                 'left':  torch.tensor([0, -1], **self.tensor_type),
                                 'up':    torch.tensor([-1, 0], **self.tensor_type),
                                 'right': torch.tensor([0, -1], **self.tensor_type),
                                 'wait':  torch.tensor([0, 0], **self.tensor_type)}

        # Define dynamic variables
        self.step_count = None
        self.current_positions = None
        self.prey_alive = None
        self.done = None
        self.viewer = None

        self.CPT = CPT_Handler() # load optimal (no CPT transform)
        self.iCPT_agent = 1 # specify which agent is risk-sensitive
        if seed is not None: self.seed(seed)

    def import_settings(self,config):

        for key, val in config.load_dict().items():

            if key not in self.__dict__.keys():
                logging.warning(f'Failed to import environment setting [{key}={val}]')
            else:
                self.__dict__[key] = val

    ##############################################
    # CORE GYM FUNCTIONS #########################
    ##############################################

    def step(self, joint_action: torch.Tensor):
        assert (self.step_count is not None), "Call reset before using step method."
        info = None
        self.step_count +=1

        rewards = torch.zeros([self.n_agents, 1], **self.tensor_type)
        if self.done:
            logging.warning('Called step when already done')
            return self.current_positions, rewards, self.done, info

        # Simultanous move of agents
        self.current_positions = self.step_agents(joint_action)
        rewards += self.get_action_reward(self.current_positions)
        self.done = self.check_done()

        # Sequential move of prey
        self.current_positions = self.step_prey()
        rewards += self.get_aux_reward(self.current_positions)
        self.done = self.check_done()

        # Construct env observation
        if self.save_rewards_for_end:
            self.cum_rewards += rewards.clone().detach()
            if self.done:
                rewards = self.cum_rewards.clone().detach()
                rewards = torch.maximum(rewards,torch.zeros(rewards.shape))
            else:
                rewards = torch.zeros([self.n_agents, 1], **self.tensor_type)
        return self.observation,rewards,self.done,info


    def reset(self):
        if self.enable_rand_init: self.current_positions = self.rand_init()
        else: self.current_positions = self._start_obs.clone().detach()
        self.done = False
        self.step_count = 0
        self.cum_rewards = torch.zeros([self.n_agents, 1], **self.tensor_type)
        return self.observation

    ##############################################
    # REWARD AND DECISION FUNCTIONS ##############
    ##############################################
    def step_agents(self,joint_action):
        # Take joint action
        new_positions = self.current_positions.detach()
        ego_actions = self.joint2ego_action(joint_action)
        for k, ego_action in enumerate(ego_actions):
            new_positions[k] = self.move(ego_action, self.current_positions[k])
        return new_positions

    def step_prey(self):
        q_inadmissable = -1e3
        if self.done: return self.current_positions
        new_positions =  self.current_positions.clone()

        # Decide Prey action
        qA = torch.zeros(self.n_egoA)
        for ia in range(self.n_egoA):
            next_position = self.move(ia, self.current_positions[-1],force_valid_move=False)
            if self.check_valid_state(next_position):
                # dist2k = torch.zeros([self.n_agents])
                # for k in range(self.n_agents):
                #     dist2k[k] = torch.dist(self.current_positions[k], next_position)
                # qA[ia] = torch.linalg.norm(torch.abs(torch.pow(dist2k, self.prey_dist_power)))
                dist2k = torch.zeros([self.n_agents])
                for k in range(self.n_agents):
                    d = torch.dist(self.current_positions[k], next_position)
                    dist2k[k] = self._max_dist**3 - (self._max_dist-d)**3
                # qA[ia] = torch.linalg.norm(torch.abs(torch.pow(dist2k, self.prey_dist_power)))
                qA[ia] = torch.mean(dist2k)
            else:  qA[ia] = q_inadmissable

        pA = torch.special.softmax(self.prey_rationality * qA,dim=0)
        move = list(WeightedRandomSampler(pA, 1, replacement=True))[0]
        if not self.enable_prey_move: move = 4

        # Apply action
        new_positions[-1]   = self.move(move,new_positions[-1])
        return new_positions

    def agent2prey_dist(self,*args):
        positions = args[0] if len(args)>=1 else self.current_positions
        dist2k = torch.zeros([self.n_agents])
        for k in range(self.n_agents):
            dist2k[k] = torch.dist(positions[k], positions[-1])
        return dist2k

    def get_aux_reward(self,positions):
        """Reward if prey gets itself caught"""
        rewards = torch.zeros([self.n_agents, 1], **self.tensor_type)
        if self.done: return rewards
        pgain = 1
        r_catch = self.r_catch * self.scale_rcatch
        for k in range(self.n_agents):
            rgain = (r_catch - self.step_count) if self.check_caught(positions) else 0
            if self.iCPT_agent == k: rg_hat, pg_hat = self.CPT.transform(rgain,pgain)
            else: rg_hat, pg_hat = rgain, pgain
            expected_reward = rg_hat  # expected reward
            rewards[k] = expected_reward
        return rewards

    def get_action_reward(self,positions):
        """Reward from agent taking actions"""
        rewards = torch.zeros([self.n_agents, 1], **self.tensor_type)
        if self.done: return rewards

        # Get stage reward
        r_catch = self.r_catch * self.scale_rcatch
        r_pen = self.r_penalty * self.scale_penalty  if self.enable_penalty else 0
        p_pen = self.p_penalty

        for k in range(self.n_agents):
            rgain = ((r_catch - self.step_count) if self.check_caught(positions) else 0)
            rloss = rgain + (r_pen if self.check_is_penalty(positions[k]) else 0)
            if self.iCPT_agent == k:
                rg_hat, pg_hat = self.CPT.transform(rgain, 1 - p_pen)
                rl_hat, pl_hat = self.CPT.transform(rloss, p_pen)
            else:
                rg_hat, pg_hat = rgain, 1 - p_pen
                rl_hat, pl_hat = rloss, p_pen
            expected_reward = (rg_hat * pg_hat) + (rl_hat * pl_hat)  # expected reward
            rewards[k] = expected_reward
        return rewards


    ##############################################
    # CHECK STATE FUNCTIONS ######################
    ##############################################
    def check_done(self,positions=None):
        positions = self.current_positions if positions is None else positions
        if self.check_caught(positions) or (self.step_count >= self.n_moves): return True
        return False
    def check_caught(self,positions):
        d0 = torch.dist(positions[0], positions[-1])
        d1 = torch.dist(positions[1], positions[-1])
        return torch.all(torch.tensor([d0,d1],**self.tensor_type) <= 1)
    def check_is_penalty(self,pos):
        assert pos.numel()==2,'incorrect size of position in penalty check'
        return  torch.any(torch.all(pos == self._penalty_states,dim=1))
    def check_is_wall(self,pos):
        assert pos.numel() == 2, 'incorrect size of position in penalty check'
        return torch.any(torch.all(pos == self._walls, dim=1))
    def check_in_bounds(self,pos):
        assert pos.numel() == 2, 'incorrect size of position in penalty check'
        return  torch.all(pos >= self._min_pos) and torch.all(pos <= self._max_pos)
    def check_valid_state(self,pos):
        return self.check_in_bounds(pos) and not self.check_is_wall(pos)

    ##############################################
    # TRANSITION UTILITIES #######################
    ##############################################
    def move(self, ego_action, curr_pos, force_valid_move=True):
        if isinstance(ego_action,torch.Tensor): ego_action = int(ego_action)
        assert ego_action in range(self.n_egoA),'Unknown ego action in env.move()'
        action_name = self.idx2action[int(ego_action)]
        next_pos = curr_pos + self.explicit_actions[action_name]
        if force_valid_move:
            is_valid = self.check_valid_state(next_pos)
            if not is_valid:  next_pos = curr_pos
        return next_pos

    def joint2ego_action(self,joint_action):
        aR = math.floor(joint_action / self.n_egoA)
        aH =  int(joint_action % self.n_egoA)
        assert aR <= self.n_egoA-1, f'robot action {aR} out of range'
        assert aH <= self.n_egoA-1, f'human action {aH} out of range'
        # print(f'A{[aR, aH]} { [self.idx2action[a] for a in [aR,aH]]} { [self.explicit_actions[self.idx2action[a]] for a in [aR,aH]]}')
        return [aR, aH]

    def rand_init(self):
        for attempt in count():
            _obs = torch.randint(low=1, high=5, size=self._start_obs.shape)
            _obs = _obs.to(torch.float32)
            is_valid = all([self.check_valid_state(s) for s in _obs])
            not_trivial = torch.any(self.agent2prey_dist(_obs) >= 2)
            # print(f'{is_valid} and {not_trivial} \t {_obs.numpy().flatten()}')
            if is_valid and not_trivial: break
            if attempt > 1e6:  raise Exception('env: rand init failed')
        return _obs

    @ property
    def observation(self):
        return self.current_positions.detach().flatten().unsqueeze(0).clone().detach()


def main():
    device = 'cpu'
    dtype = torch.float32
    env = PursuitEvastionGame(1,device,dtype)

    state = env.reset()

    # pos = torch.tensor([[1,1],[0,0],[0,0]],**env.tensor_type)
    # # print(f'bounds: {env.check_in_bounds(pos[0])}')
    # # print(f'Wall: {env.check_is_wall(pos[0])}')
    # print(f'VALID: {env.check_valid_state(pos[0])}\n')
    # print(f'Caught: {env.check_caught(pos)}')
    # print(f'Penalty[k=0]: {env.check_is_penalty(pos[0])}')
    # print(f'Move {env.move(0,pos[0])}')


    aJ = 5 #torch.tensor(1,**env.tensor_type)
    for epi in range(10):
        print(f'state: {env.current_positions.flatten()}')
        next_state,reward,done,_ = env.step(aJ)



def subfun():
    pass


if __name__ == "__main__":
    main()
