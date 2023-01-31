############################################################
## Packages ################################################
import copy
import logging
import random

import numpy as np
import torch
import torch.optim as optim

from . import CFG  # INITIALIZED IN IDQN.__init__
from .DeepQNet import DQN
from .DeepQNet import optimize_model, ReplayMemory, soft_update, test_policy
from .utilities.learning_utils import EpisodeTimer, RL_Logger
from .utilities.make_env import PursuitEvastionGame

# Set fixed params
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)



#################################################################
## Training algorithm ###########################################
################################################################
def run_IDQN(iWorld,**kwargs):

    #############################################################
    # Unpack Config Params
    #############################################################
    # Update any run-specific params
    this_CFG = copy.deepcopy(CFG)
    for key, val in kwargs.items(): this_CFG.set_option(key,val)

    # Global (except WORLDS) ---------------------------------
    algorithm_name = this_CFG.algorithm_name
    policy_type = this_CFG.policy_type
    # Training -------------------------------------------------
    num_episodes = this_CFG.Training.num_episodes
    update_interval = this_CFG.Training.update_interval
    report_interval =  this_CFG.Training.report_interval
    test_interval =  this_CFG.Training.test_interval
    test_episodes = this_CFG.Training.test_episodes
    rand_init_episodes = this_CFG.Training.rand_init_episodes
    warmup_samples = this_CFG.Training.warmup_samples
    EPS_START = this_CFG.Training.eps_schedule.START
    EPS_END = this_CFG.Training.eps_schedule.END
    EPS_DECAY = this_CFG.Training.eps_schedule.DECAY
    EPS_EXPLORE = this_CFG.Training.eps_schedule.EXPLORE
    # Learning Params -------------------------------------------
    LR = this_CFG.Learning.LR
    tau = this_CFG.Learning.tau
    gamma = this_CFG.Learning.gamma
    batch_size = this_CFG.Learning.batch_size
    memory_size = this_CFG.Learning.memory_size
    update_iterations = this_CFG.Learning.update_iterations
    # Torch Settings -------------------------------------------
    dtype = this_CFG.torch.dtype
    device = this_CFG.torch.device
    torch.set_default_dtype(this_CFG.torch.dtype)
    # Report setup ---------------------------------------------
    print(this_CFG)

    ###################################################
    # CREATE LEARNING OBJECTS #########################
    policy_net = DQN(run_config=this_CFG).to(device)
    target_net = DQN(run_config=this_CFG).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(memory_size)

    # Create environment -------------------------
    env = PursuitEvastionGame(iWorld,device,dtype)
    env.import_settings(this_CFG.Environment)

    test_env = PursuitEvastionGame(iWorld, device, dtype)
    test_env.import_settings(this_CFG.Environment)

    # Set up utilities -------------------------------
    Logger = RL_Logger(iWorld,algorithm_name,policy_type)
    epi_timer = EpisodeTimer()

    # Schedule Epsilon Decay ------------------------------
    episodes = np.hstack([np.zeros(EPS_EXPLORE), np.arange(num_episodes - EPS_EXPLORE)])
    epi_epsilons = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * episodes / EPS_DECAY)


    #############################################################################
    # BEGIN EPISODES ############################################################
    for i_episode in range(num_episodes):
        # --------------------------------------------------------------------
        # Open episode  ------------------------------------------------------
        # --------------------------------------------------------------------
        Logger.flush() # interative plot handling
        epi_timer.sample() # sample episode completion time
        warmup_finished = (len(memory) > warmup_samples)
        epsilon = epi_epsilons[i_episode] # get decaying epsilon
        env.enable_rand_init = (i_episode < rand_init_episodes) # random start?
        if Logger.end_button_state==1: # Exit if plot button pressed
            logging.warning(f'EXITING FROM INTERFACE\n\n'); break

        # --------------------------------------------------------------------
        # Sample 1 round of play ---------------------------------------------
        # --------------------------------------------------------------------
        observations = policy_net.simulate(env,epsilon = epsilon)
        for state, action, next_state, reward in observations:
            memory.push(state, action, next_state, reward)

        # --------------------------------------------------------------------
        # Get periodic subroutine flags --------------------------------------
        # --------------------------------------------------------------------
        time2update = (i_episode % update_interval == 0) if update_interval!=0 else True
        time2report = (i_episode % report_interval == 0)
        time2test = (i_episode % test_interval == 0)

        time2update = time2update if warmup_finished else False
        # time2report = time2report if warmup_finished else False
        # time2test =  time2test if warmup_finished else False

        # Perform (one) step of the optimization (on the policy network) ------
        if time2update:
            optimize_model(policy_net, target_net,
                           optimizer=optimizer,
                           memory=memory,
                           GAMMA=gamma,
                           BATCH_SIZE=batch_size,
                           update_iterations=update_iterations)
            if update_interval == 0: soft_update(policy_net, target_net, TAU=tau) # Soft update every iteration
            else: target_net.load_state_dict(policy_net.state_dict()) # periodic hard update

        # Report ----------------------------------------------------------------
        if time2report:
            test_score, test_length, test_psucc = test_policy(test_env, test_episodes, policy_net) # Test policy N times
            Logger.log_episode(test_score, test_length, test_psucc, buffered=False, episode=i_episode) # Update episode learning tracker and draw plot
            Logger.draw() # draw learning plot

            # Report in terminal
            disp = ''
            disp += f'[{epi_timer.remaining_time(i_episode,num_episodes)}] '
            disp += f'[W{iWorld} {policy_type}] '
            if not warmup_finished: disp += ' WARMUP\t'
            disp += '{:<20}'.format(f'Epi[{i_episode}/{num_episodes}]')
            disp += "stats: [s/epi:{:<4} ".format(np.round(epi_timer.mean_dur, 2)) + f'eps:{round(epsilon,2)} Memory:{len(memory)} ] \t'
            disp += "test score: {:<35}".format( f'[Σr_k(t):{np.round(test_score, 1)} epi_len:{np.round(test_length, 1)} p(catch):{np.round(test_psucc, 1)}]\t')
            print(disp)


        elif time2test: # perform low-requirement test (no draw or console print)
            test_score, test_length, test_psucc = test_policy(test_env, test_episodes, policy_net)
            Logger.log_episode(test_score, test_length, test_psucc, buffered=False, episode=i_episode)

        torch.clear_autocast_cache()

    print('Complete')
    # DQN.save(policy_net,iWorld,policy_type ,algorithm_name  + '_extended' if continued else '',axes=Logger.axs)
    # DQN.save(policy_net,iWorld,policy_type ,algorithm_name,axes=Logger.axs )#
    # Logger.blocking_preview()
    # DQN.save(policy_net,iWorld,policy_type = policy_type ,algorithm = algorithm_name,axes=Logger.fig)#
    # Logger.save()
    # Logger.close()


    Logger.blocking_preview()



