import datetime as _dt
import logging
import math
import os
import warnings
from datetime import datetime
from itertools import count
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from scipy import signal, stats
import time

#################################################################
## Replay Memory ################################################
#################################################################



class EpisodeTimer(object):
    def __init__(self,max_buffer=100):
        self.durations = []
        self.max_buffer = max_buffer
        self.last_sample_time = None
        self.n_del = max_buffer/10

    def sample(self):
        if self.last_sample_time is not None:
            self.durations.append(time.time() - self.last_sample_time)
            if len(self.durations) > self.max_buffer: self.durations.pop(0)
                # self.durations = self.durations[self.n_del:]
        self.last_sample_time = time.time()

    @property
    def mean_dur(self):
        if len(self.durations)>0: return np.mean(self.durations)
        else: return 0

    def remaining_time(self,i_epi,num_episodes):
        return str(_dt.timedelta(seconds=round(self.mean_dur * (num_episodes - i_epi))))
        # return str(datetime.timedelta(seconds=round(self.mean_dur * (num_episodes - i_epi))))


class CPT_Handler(object):
    @classmethod
    def rand(cls,assume=None,verbose=True):
        cpt = CPT_Handler(rand=True,assume=assume)
        if verbose: cpt.preview_params()
        return cpt


    def __init__(self,rand=False,assume=None):
        self.b, self.b_bounds = 0., (0, 5)
        self.lam, self.lam_bounds = 1., (1/5, 5)
        self.yg, self.yg_bounds = 1., (0.4, 1.0)
        self.yl, self.yl_bounds = 1., (0.4, 1.0)
        self.dg, self.dg_bounds = 1., (0.4, 1.0)
        self.dl, self.dl_bounds = 1., (0.4, 1.0)
        self.rationality, self.rationality_bounds = 1., (1., 1.)

        self.symm_reward_sensitivity = False
        self.symm_probability_sensitivity = False

        self.assumption_attempts = 500
        self.n_test_sensitivity = 100
        self.p_thresh_sensitivity = 0.2
        self.r_range_sensitivity = (-15, 15)
        self.r_pen = -3
        self.p_pen = 0.5

        self.n_draw_sensitivity = 100
        self.n_draw_ticks = 5

        self.paccept_sensitivity = 0
        self.attribution = 'insensitive'

        if rand: self.rand_params(assume=assume)

    def __str__(self):
        return self.flat_preview()
    def flat_preview(self):
        sigdig = 2
        # self.preview_params()
        disp = []
        for key in ['b','lam','yg','yl','dg','dl','rationality']:
            disp.append(round(self.__dict__[key], sigdig))

        dfavor = self.get_favor()
        favor = 'more gain' if dfavor >0 else 'more loss'

        return f'CPT({round(self.paccept_sensitivity,sigdig)}):{disp} => [{favor}]={dfavor}%'# G{TG} x L{TL}'

    def _get_optimal(self):
        b, lam, yg, yl, dg, dl, rationality = 0., 1., 1., 1., 1., 1., 1.
        return b, lam, yg, yl, dg, dl, rationality

    @property
    def is_optimal(self):
        check = True
        if self.b != 0.: check = False
        if self.lam != 1.: check = False
        if self.yg != 1.: check = False
        if self.yl != 1.: check = False
        if self.dg != 1.: check = False
        if self.dl != 1.: check = False
        return check

    def preview_params(self,sigdig=2):
        print(f'### [CPT Parameters] ### <==',end='')
        # print('\t|',end='')
        for key in ['b','lam','yg','yl','dg','dl','rationality']:
            print(' {:<1}: [{:<1}]'.format(key,round(self.__dict__[key],sigdig)),end='')
        print(f'\t sensitivity: [{round(self.paccept_sensitivity,sigdig)}] \t attribution: [{self.attribution}]',end='')
        print(f'')
    def transform(self, *args):
        if len(args)==2:
            r, p = args
            if self.is_optimal: return r,p
            b, lam, yg, yl, dg, dl = self.b, self.lam, self.yg, self.yl, self.dg, self.dl
        elif len(args)==8:
            r, p, b, lam, yg, yl, dg, dl = args
        else: raise Exception("UNKOWN NUMBER OF CPT TRANSFORM ARGUMENTS")

        is_cert = (p==1)
        if (r - b) >= 0:
            rhat = pow(r - b, yg)
            phat = pow(p, dg) / pow(pow(p, dg) + pow(1 - p, dg), dg)
        else:
            rhat = -lam * pow(abs(r - b), yl)
            phat = pow(p, dl) / pow(pow(p, dl) + pow(1 - p, dl), dl)

        if is_cert: phat=1
        return rhat, phat

    def plot_indifference_curve(self, ax=None):
        N = self.n_draw_sensitivity
        if ax is None: fig, ax = plt.subplots(1, 1)
        n_ticks = self.n_draw_ticks
        rmin, rmax = self.r_range_sensitivity
        r_cert = np.linspace(rmin, rmax, N)  # + r_pen/2
        r_gain = np.linspace(rmin, rmax, N)  # [0,20]

        attribution, p_accept = self._get_sensitivity(self.b, self.lam,
                                                      self.yg, self.yl,
                                                      self.dg, self.dl,
                                                      self.rationality,
                                                      return_paccept=True, N=N)

        ax.matshow(p_accept - 0.5, cmap='bwr', origin='lower')
        ax.set_title(
            f'Preference Map [{attribution}]\n (White: indifferent)(Red: prefer gamble)(Blue: prefer certainty)')
        ax.set_xlabel('$\mathbb{C}[R_{gamble} \; | \; r_{\\rho}=-3,p_{\\rho}=0.5 ]$')
        # ax.set_xlabel('$\mathbb{C}[R_{gamble}] = (1-p_{pen})R_{gain}+p_{pen}(R_{gain}-r_{pen})$')

        ax.set_xticks(np.linspace(0, 100, n_ticks))
        ax.set_yticks(np.linspace(0, 100, n_ticks))
        ax.set_xticklabels(np.round(np.linspace(r_gain[0], r_gain[-1], n_ticks), 1))

        ax.set_ylabel('$\mathbb{C}[R_{certainty}]$')
        ax.set_yticklabels(np.round(np.linspace(r_cert[0], r_cert[-1], n_ticks), 1))

    def _get_sensitivity(self, b, lam, yg, yl, dg, dl, rationality, return_paccept=False, N=None):
        iaccept = 0
        N = self.n_test_sensitivity if N is None else N
        rmin, rmax = self.r_range_sensitivity
        r_cert = np.linspace(rmin, rmax, N)  + self.r_pen/2
        r_gain = np.linspace(rmin, rmax, N)  # [0,20]
        r_loss = r_gain + self.r_pen
        p_thresh = self.p_thresh_sensitivity

        # p_accept = np.empty([N, N])
        # for r in range(N):
        #     for c in range(N):
        #         rg = r_gain[c]
        #         rl = r_loss[c]
        #         rc = r_cert[r]
        #
        #         rg_hat, pg_hat = self.transform(rg, 1 - self.p_pen, b, lam, yg, yl, dg, dl)
        #         rl_hat, pl_hat = self.transform(rl, self.p_pen, b, lam, yg, yl, dg, dl)
        #         Er_gamble = (rg_hat * pg_hat) + (rl_hat * pl_hat)
        #         Er_cert = rc - b
        #         Er_choices = np.array([Er_gamble, Er_cert])
        #
        #         pCPT = softmax(rationality * Er_choices)
        #         p_accept[r, c] = pCPT[iaccept]
        #
        # p_sum = np.mean(p_accept - 0.5)
        # if abs(p_sum) < p_thresh: attribution = 'insensitive'
        # elif p_sum >= p_thresh: attribution = 'seeking'
        # elif p_sum <= -p_thresh: attribution = 'averse'
        # else: raise Exception('Unknown CPT attribution')

        p_accept = 0
        p_sum = self.get_favor()

        if abs(p_sum) < p_thresh: attribution = 'insensitive'
        elif p_sum >= p_thresh and abs(p_sum)<2: attribution = 'seeking'
        elif p_sum <= -p_thresh and abs(p_sum)<2: attribution = 'averse'
        else:
            # raise Exception('Unknown CPT attribution')
            attribution = 'Unknown CPT attribution'
        if return_paccept: return attribution, p_accept
        else: return attribution

    def get_favor(self):
        p = 0.5

        dfavors = np.zeros(int(10))
        for r in range(dfavors.size):
            rhatG, phatG = np.array(self.transform((r + 1), p))
            rhatL, phatL = np.array(self.transform(-(r + 1), p))
            dfavors[r] = (rhatG * phatG + rhatL * phatL) / (r + 1)

        dfavor = np.nan_to_num(np.mean(dfavors)).round(1)  # round(rel_diff[0]-rel_diff[1],1)
        return dfavor


    def _sample_random_params(self, n_samples):
        b = np.random.choice(np.linspace(self.b_bounds[0], self.b_bounds[1], n_samples))
        lam_seeking = np.linspace(self.lam_bounds[0], 1, int(n_samples / 2))
        lam_averse = np.linspace(self.lam_bounds[0] + 1, self.lam_bounds[1], int(n_samples / 2))
        lam = np.random.choice(np.hstack([lam_seeking, lam_averse]))
        yg = np.random.choice(np.linspace(self.yg_bounds[0], self.yg_bounds[1], n_samples))
        yl = np.random.choice(np.linspace(self.yl_bounds[0], self.yl_bounds[1], n_samples))
        dg = np.random.choice(np.linspace(self.dg_bounds[0], self.dg_bounds[1], n_samples))
        dl = np.random.choice(np.linspace(self.dl_bounds[0], self.dl_bounds[1], n_samples))
        rationality = np.random.choice(
            np.linspace(self.rationality_bounds[0], self.rationality_bounds[1], n_samples))
        if self.symm_reward_sensitivity: yl = yg
        if self.symm_probability_sensitivity: dl = dg
        return b, lam, yg, yl, dg, dl, rationality

    def rand_params(self, assume=None, n_samples=100):
        assert assume.lower() in ['averse', 'seeking', 'insensitive','baseline', None], f'CPT parameter assumption unknown: {assume}'
        if assume is not None:
            if assume.lower() == 'baseline':
                b, lam, yg, yl, dg, dl, rationality = self._get_optimal()
                self.b, self.lam = b, lam
                self.yg, self.yl = yg, yl
                self.dg, self.dl = dg, dl
                self.rationality = rationality
            else:
                for attempt in range(self.assumption_attempts):
                    b, lam, yg, yl, dg, dl, rationality = self._sample_random_params(n_samples)
                    self.b, self.lam = b, lam
                    self.yg, self.yl = yg, yl
                    self.dg, self.dl = dg, dl
                    self.rationality = rationality

                    attribution,p_accept = self._get_sensitivity(b, lam, yg, yl, dg, dl, rationality, return_paccept=True)
                    self.attribution = attribution
                    # self.paccept_sensitivity = np.mean(p_accept - 0.5)

                    if attribution.lower() == assume.lower(): break
                    if attempt>=self.assumption_attempts-1: logging.warning(f"CPT unable to generate assumed {assume} parameters")



def schedule(start,end,N_start,N_end,N_total,slope=1.0):
    if start == end: return start * np.ones(N_total)
    warmup = start * np.ones(N_start)
    perform = end * np.ones(N_end)
    N_transition = N_total - N_start - N_end
    iinv = np.power(1 / (np.linspace(1, 10,N_transition) - 0.1) - 0.1, slope)
    improve = (start + end) * iinv + end
    epi_schedule = np.hstack([warmup, improve, perform])
    return epi_schedule


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def spawn_env_worker(env, policy_net, epsilon,que, num_iterations = 4):
    # processes = []
    # manager = mp.Manager()
    # que = manager.Queue()
    # num_proc = 4
    # plt.ioff()
    # for iproc in range(num_proc):
    #     p = mp.Process(target=spawn_env_worker,args=(copy.deepcopy(env), policy_net, epsilon,que,))
    #     p.start()
    #     processes.append(p) #spawn_env_worker(iWorld, policy_net, epsilon)
    #
    # for p in processes:
    #     p.join()
    # que.put(None)
    # plt.ion()
    #
    # for _ in count():
    #     sample = que.get()
    #     if sample is None:  break
    #     else:
    #         # print(data.shape)
    #         state = sample[0:6]
    #         action = sample[6]
    #         next_state = None if torch.all(sample[7:13]==0) else sample[7:13]
    #         reward = sample[13:14]
    #         memory.push(state, action, next_state, reward)


    # print(f'start_proc')
    # for i in range(num_iterations):
    #     que.put(i*torch.ones(1, 10))
    # mp.Event().wait()
    #
    nVars = 15
    history = torch.empty([0,nVars])#.share_memory_()
    observations = torch.empty([0, nVars])

    # history = []
    for iter in range(num_iterations):
        # iter_history = []
        state = env.reset()  # Initialize the environment and get it's state
        done = False
        for t in count():
            action = policy_net.sample_action(state, epsilon)
            next_state, reward, done, _ = env.step(action.squeeze())
            # obs = [state, action, None if done else next_state, reward]
            obs = torch.cat([state.flatten(), action.flatten(), torch.zeros(6).flatten() if done else next_state.flatten(), reward.flatten()])
            observations = torch.cat([observations, obs.reshape([1, nVars])], dim=0)

            # obs.share_memory_()
            # que = torch.cat([que,  obs.reshape([1, nVars])], dim=0)
            # que.put(obs.s)

            # history = torch.cat([history,obs.reshape([1,nVars])],dim=1)


            if done:                break

    #return history
    # print(observations.shape)
    # history = torch.cat([history, observations], dim=0)
    que.put(observations)

class ExecutionTimer(object):
    def __init__(self,max_samples=10_000,sigdig = 3,enable=True):
        # self.profiles = []
        self.profiles = {}
        self.max_samples = max_samples
        self.sigdig = sigdig
        self.tstarts = {}
        self.main_iter = 0
        self.ENABLE = enable

    def __call__(self, *args, **kwargs):
        if not self.ENABLE: return None
        return self.add_profile(*args, **kwargs)

    def preview_percent(self):
        if not self.ENABLE: return None
        if len(self.profiles['main']['dur']) > 0:
            mean_execution_time = np.mean(self.profiles['main']['dur'])
            print(f'Execution Times [Percent]: ############')
            for name in self.profiles.keys():
                profile_durs = np.array(self.profiles[name]['dur'])
                disp = '\t| {:<10}: {:<5}%'.format(name, (np.mean(profile_durs) / mean_execution_time).round(self.sigdig))
                print(disp)
            # print(f'\t| {name}: {(np.mean(ave_durs)/mean_execution_time).round(self.sigdig)}%')
    # def preview_percent(self):
    #     if not self.ENABLE: return None
    #     mean_execution_time =  np.mean(self.profiles['main']['dur'])
    #     # if len(self.profiles['main']['dur']) > 0:
    #     print(f'Execution Times [Percent]: ############')
    #     for name in self.profiles.keys():
    #         profile_durs = np.array(self.profiles[name]['dur'])
    #         profile_imains = np.array(self.profiles[name]['main_iter'])
    #         ave_durs = []
    #         for imain in  self.profiles['main']['main_iter']:
    #            idurs = profile_durs[np.where(profile_imains == imain)]
    #            if len(idurs) > 0: ave_durs.append(np.mean(idurs))
    #
    #         disp = '\t| {:<10}: {:<5}%'.format(name,(np.mean(ave_durs)/mean_execution_time).round(self.sigdig))
    #         print(disp)
    #         #print(f'\t| {name}: {(np.mean(ave_durs)/mean_execution_time).round(self.sigdig)}%')

    def preview_all(self):
        if not self.ENABLE: return None
        print(f'Execution Times: ############')
        for name in self.profiles.keys():
            print(f'\t| {name}: {np.mean(self.profiles[name]).round(self.sigdig)}')

    def mean_duration(self,name):
        if not self.ENABLE: return None
        return np.mean(self.profiles[name])


    def add_profile(self,name,status):
        if not self.ENABLE: return None
        assert status == 'start' or status == 'stop', f'incorrect input arg [{status}]'
        if status == 'start':
            self.tstarts[name] = time.time()
            if name not in self.profiles.keys():
                self.profiles[name] = {'dur': [], 'main_iter': []}
        elif status == 'stop':
            self.profiles[name]['dur'].append(time.time() - self.tstarts[name])
            self.profiles[name]['main_iter'].append(self.main_iter)

    def main_profile(self,status):
        if not self.ENABLE: return None

        assert status == 'start' or status == 'stop', f'incorrect input arg [{status}]'
        name = 'main'
        if status == 'start':
            self.tstarts[name] = time.time()
            if name not in self.profiles.keys():
                self.profiles[name] = {'dur':[],'main_iter':[]}
        elif status == 'stop':
            self.profiles[name]['dur'].append(time.time() - self.tstarts[name])
            self.profiles[name]['main_iter'].append(self.main_iter)
            self.main_iter += 1



#################################################################
## RL Logger (Plotting) #########################################
#################################################################


class RL_Logger(object):
    fig = None
    axs = None
    lw_small = 0.25
    lw_med = 1
    agent_colors = ['r', 'b']
    epi_length_color = 'g'
    epi_psucc_color = 'm'
    ls_raw = '--'
    ls_filt = '-'
    enable_legend = True


    reward_raw_config0  = {'lw': lw_small,   'c': agent_colors[0], 'ls': ls_raw,'label':['raw: $r_{R}$','raw: $r_{H}$'][0]}
    reward_filt_config0 = {'lw': lw_med,     'c': agent_colors[0], 'ls': ls_filt,'label':['filtered: $r_{R}$','filtered: $r_{H}$'][0]}
    reward_raw_config1  = {'lw': lw_small,   'c': agent_colors[1], 'ls': ls_raw, 'label': ['raw: $r_{R}$', 'raw: $r_{H}$'][1]}
    reward_filt_config1 = {'lw': lw_med,     'c': agent_colors[1], 'ls': ls_filt, 'label': ['filtered: $r_{R}$', 'filtered: $r_{H}$'][1]}


    length_raw_config = {'lw': lw_small, 'c': epi_length_color, 'ls': ls_raw, 'label': 'raw: epi len'}
    length_filt_config = {'lw': lw_med, 'c': epi_length_color, 'ls': ls_filt, 'label': 'filtered: epi len'}
    psucc_raw_config = {'lw': lw_small, 'c': epi_psucc_color, 'ls': ls_raw, 'label': 'raw: P(Success)'}
    psucc_filt_config = {'lw': lw_med, 'c': epi_psucc_color, 'ls': ls_filt, 'label': 'filtered: P(Success)'}


    # line_reward = None
    # line_mreward = None
    raw_reward_lines = [None,None]
    filt_reward_lines = [None,None]

    line_len = None
    line_mlen = None
    line_psucc = None
    line_mpsucc = None

    ax_button = None
    close_button = None


    def __init__(self,iWorld,algorithm_name,policy_type,fname_notes=''):
        plt.ion()
        if RL_Logger.fig is None:
            self.new_plot()
        self.Epi_Reward = []
        self.Epi_Length = []
        self.Epi_Psuccess = []
        # self.Epi_Reward = np.zeros([0,2],dtype=np)
        # self.Epi_Length = np.zeros([0,1])
        # self.Epi_Psuccess = np.zeros([0,1])

        self.max_memory_size = 4000
        self.max_memory_resample = 3
        self.keep_n_early_memory = 1800
        self.is_resampled = False
        self.current_episode = 0
        self.psuccess_window = 7
        self.filter_window = 100
        self.auto_draw = False
        self.itick = 0
        self.refresh_rate = 1

        self.end_button_state = 0

        self._psuccess_buffer = []

        self.nepi_since_last_draw = 0
        self.draw_every_n_episodes = 0

        self.xdata = np.arange(2)
        self.timestamp = datetime.now().strftime("--%b%d--h%H-m%M")
        self.fname_notes = fname_notes
        self.project_root = os.getcwd().split('MARL')[0]+'MARL\\'
        self.save_dir = self.project_root + f'results\\IDQN_{self.fname_notes}\\'
        self.file_name = 'Fig_IDQN'

        self.file_name = f'Fig_{algorithm_name}_{policy_type}'  # + '_extended' if continued else ''
        self.update_save_directory(self.project_root + f'results\\IDQN_W{iWorld}\\')
        self.update_plt_title(f'[W{iWorld}-{policy_type}] {algorithm_name.replace("_", " ")} Training Results')
        self.make_directory();
        self.filter_window = 10
        self.refresh_rate = 10


    def end_button_callback(self,event):
        self.end_button_state = 1

    def new_plot(self):
        # plt.close(RL_Logger.fig)
        dummy_data = np.zeros([3,1])
        # plt.clf()

        # if RL_Logger.fig is not None:
            # plt.clf()
            # plt.close(RL_Logger.fig)
        RL_Logger.fig, RL_Logger.axs = plt.subplots(3, 1,constrained_layout=True)
        RL_Logger.fig.set_size_inches(11, 8.5)

        # RL_Logger.raw_reward_lines = RL_Logger.axs[0].plot(np.tile(dummy_data,[1,2]),**RL_Logger.reward_raw_config)
        # RL_Logger.filt_reward_lines = RL_Logger.axs[0].plot(np.tile(dummy_data,[1,2]),**RL_Logger.reward_filt_config)
        RL_Logger.raw_reward_lines[0] = RL_Logger.axs[0].plot(dummy_data, **RL_Logger.reward_raw_config0)[0]
        RL_Logger.filt_reward_lines[0] = RL_Logger.axs[0].plot(dummy_data, **RL_Logger.reward_filt_config0)[0]
        RL_Logger.raw_reward_lines[1] = RL_Logger.axs[0].plot(dummy_data, **RL_Logger.reward_raw_config1)[0]
        RL_Logger.filt_reward_lines[1] = RL_Logger.axs[0].plot(dummy_data, **RL_Logger.reward_filt_config1)[0]

        RL_Logger.line_len, = RL_Logger.axs[1].plot(np.zeros(2), **RL_Logger.length_raw_config)
        RL_Logger.line_mlen, = RL_Logger.axs[1].plot(np.zeros(2), **RL_Logger.length_filt_config)
        RL_Logger.line_psucc, = RL_Logger.axs[2].plot(np.zeros(2), **RL_Logger.psucc_raw_config)
        RL_Logger.line_mpsucc, = RL_Logger.axs[2].plot(np.zeros(2),**RL_Logger.psucc_filt_config)

        if RL_Logger.enable_legend:
            for i in range(len(RL_Logger.axs)): RL_Logger.axs[i].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        RL_Logger.axs[0].set_title('IDQN Training Results')
        RL_Logger.axs[0].set_ylabel('Epi Reward')
        RL_Logger.axs[1].set_ylabel('Epi Length')
        RL_Logger.axs[1].set_ylim([-0.1, 20.1])
        RL_Logger.axs[2].set_ylabel('P(Success)')
        RL_Logger.axs[-1].set_xlabel('Episode')


        w = 0.15
        h = 0.075
        RL_Logger.ax_button = RL_Logger.fig.add_axes([1-1.1*w, 0.75*h, w, h])
        RL_Logger.close_button = Button(RL_Logger.ax_button, 'end training')
        RL_Logger.close_button.on_clicked(self.end_button_callback)


    def update_save_directory(self,dir): self.save_dir = dir #f'results/IDQN_{fname_notes}/'
    def update_plt_title(self,title): self.axs[0].set_title(title)


    def make_directory(self):
        print(f'Initializing Data Storage')
        try: os.mkdir(self.save_dir),print(f'\t| Making root results directory [{self.save_dir}]...')
        except: print(f'\t| Root results directory already exists [{self.save_dir}]...')

        # subdir = self.save_dir + 'recordings/'
        # try: os.mkdir(subdir),print(f'\t| Making sub directory [{subdir}]...')
        # except: print(f'\t| Sub directory already exists [{subdir}]...')

        # subdir = self.save_dir + 'recordings/idqn'
        # try: os.mkdir(subdir), print(f'\t| Making sub directory [{subdir}]...')
        # except:  print(f'\t| Sub directory already exists [{subdir}]...')
    def draw(self,verbose=False):

        if self.auto_draw:
            if self.nepi_since_last_draw >= self.draw_every_n_episodes:
                if verbose: print(f'[Plotting...]')
                self.update_plt_data()
                self.fig.canvas.flush_events()
                self.fig.canvas.draw()
                self.nepi_since_last_draw = 0
            else: self.nepi_since_last_draw += 1
        else:
            if verbose: print(f'[Plotting...]', end='')
            self.update_plt_data()
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
            self.nepi_since_last_draw = 0



    def update_data(self,line, xdata, ydata,yscale=None):
        line.axes.relim()
        line.axes.autoscale_view()
        if yscale is not None:
            line.axes.set_ylim(yscale)
        line.set_xdata(xdata), line.set_ydata(ydata)

    def update_plt_data(self):
        warnings.filterwarnings("ignore")
        x = self.xdata

        rewardsK = np.array(self.Epi_Reward)
        for il in range(len(self.raw_reward_lines)):
            self.update_data(self.raw_reward_lines[il],x, rewardsK[:,il])
            self.update_data(self.filt_reward_lines[il], x, self.filter(rewardsK[:,il]))
        self.update_data(self.line_len, x, self.Epi_Length,yscale=[-0.1,20.1])
        self.update_data(self.line_mlen, x, self.filter(self.Epi_Length),yscale=[-0.1,20.1])
        self.update_data(self.line_psucc, x, self.Epi_Psuccess,yscale=[-0.1,1.1])
        self.update_data(self.line_mpsucc, x, self.filter( self.Epi_Psuccess),yscale=[-0.1,1.1])

        warnings.filterwarnings("default")


    def log_episode(self,agent_reward,episode_length,was_success,buffered=True,episode=None):
        self.check_resample()
        self.Epi_Reward.append(agent_reward)
        self.Epi_Length.append(episode_length)


        # update probability of success
        if buffered:
            self._psuccess_buffer.append(int(was_success))
            if len(self._psuccess_buffer) > self.psuccess_window: self._psuccess_buffer.pop(0)
            psuccess = np.mean(self._psuccess_buffer)
            self.Epi_Psuccess.append(psuccess)
        else:
            self.Epi_Psuccess.append(was_success)

        if episode is None:  self.current_episode += 1
        else: self.current_episode = episode
        # self.xdata = np.linspace(0, self.current_episode, len(self.Epi_Reward))

        if self.is_resampled:
            n_keep =self.keep_n_early_memory
            xdata_keep = np.arange(n_keep)
            xdata_resampled = np.linspace(n_keep, self.current_episode, len(self.Epi_Reward)-n_keep)
            self.xdata = np.hstack([xdata_keep,xdata_resampled])
        else:
            self.xdata = np.linspace(0, self.current_episode, len(self.Epi_Reward))

        if self.auto_draw: self.draw()


    def check_resample(self):
        if len(self.Epi_Reward) > self.max_memory_size:
            n_keep = self.keep_n_early_memory
            n_resample = self.max_memory_resample
            _Epi_Reward = list(self.filter(self.Epi_Reward, window=max(n_resample, 3)))
            _Epi_Length = list(self.filter(self.Epi_Length, window=max(n_resample, 3)))
            _Epi_Psuccess = list(self.filter(self.Epi_Psuccess, window=max(n_resample, 3)))

            self.Epi_Reward     = _Epi_Reward[:n_keep] + _Epi_Reward[n_keep::n_resample]
            self.Epi_Length     = _Epi_Length[:n_keep] + _Epi_Length[n_keep::n_resample]
            self.Epi_Psuccess   = _Epi_Psuccess[:n_keep] + _Epi_Psuccess[n_keep::n_resample]
            self.is_resampled = True
            # print(f'[Resampling logger...]')

    def filter(self,data,window=None):
        if window is None: window = self.filter_window
        if window % 2 == 0: window += 1
        if len(data) > window:
            buff0 = np.mean(data[ceil(window / 4):]) * np.ones(window)
            buff1 = np.mean(data[:-ceil(window / 4)]) * np.ones(window)
            tmp_data = np.hstack([buff0, data, buff1])
            filt = stats.norm.pdf(np.arange(window), loc=window / 2, scale=window / 5)
            new_data = signal.fftconvolve(tmp_data, filt / np.sum(filt), mode='full')
            ndiff = np.abs(len(data) - len(new_data))
            new_data = new_data[int(ndiff / 2):-int(ndiff / 2)]
        else: new_data = data
        # filt = signal.gaussian(window, std=3)
        # filt = filt/np.sum(filt)
        # new_data = signal.fftconvolve(data, filt, mode='same')
        return new_data

    def flush(self):
        RL_Logger.fig.canvas.flush_events()

    def tick(self):
        if self.itick % self.refresh_rate==0:
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
        self.itick += 1
    def save(self):
        # path = self.save_dir + self.file_name +self.fname_notes+ self.timestamp
        path = self.save_dir + self.file_name
        plt.savefig( path)
        print(f'Saved logger figure in [{path}]')
        # print("date and time:", self.save_dir + self.file_name + self.timestamp)

    def close(self):
        self.end_button_state = 0
        self.Epi_Reward = []
        self.Epi_Length = []
        self.Epi_Psuccess = []
        self._psuccess_buffer = []
        self.timestamp = datetime.now().strftime("--%b%d--h%H-m%M")
        self.project_root = os.getcwd().split('MARL')[0] + 'MARL\\'
        self.save_dir = self.project_root + f'results\\IDQN_{self.fname_notes}\\'
        self.file_name = 'Fig_IDQN'

        # self.new_plot()
        dummy_y = np.zeros([3, 1])
        dummy_x = np.arange(3).reshape(dummy_y.shape)

        for il in range(len(self.raw_reward_lines)):
            self.update_data(self.raw_reward_lines[il], dummy_x, dummy_y)
            self.update_data(self.filt_reward_lines[il], dummy_x, dummy_y)
        self.update_data(self.line_len, dummy_x, dummy_y, yscale=[-0.1, 20.1])
        self.update_data(self.line_mlen, dummy_x, dummy_y, yscale=[-0.1, 20.1])
        self.update_data(self.line_psucc, dummy_x, dummy_y, yscale=[-0.1, 1.1])
        self.update_data(self.line_mpsucc, dummy_x, dummy_y, yscale=[-0.1, 1.1])


        # plt.ioff()
        # plt.show()

    def blocking_preview(self):
        plt.ioff()
        plt.show()



if __name__ == "__main__":
    import time

    Logger = RL_Logger()
    for trial in range(3):
        for epi in range(15):
            print(f'Trial {trial} {epi}')
            agent_rewards = [math.sin(epi),math.cos(epi)]
            length = epi % 20
            psucc = epi
            Logger.log_episode(agent_rewards,length,psucc)
            Logger.draw()
            time.sleep(0.1)
        Logger.close()
