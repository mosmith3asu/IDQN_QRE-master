import logging
from dataclasses import dataclass

import torch
########################################################################################
## Import learning config from yaml ####################################################
########################################################################################
import yaml

import IDQN


@dataclass
class Config:
    def __init__(self,path=None,depth=1,**kwargs):
        self._depth = depth
        self._is_root = True if depth==1 else False
        if self._is_root and path==None:
            default_config_path = "\\".join(str(__file__).split('\\')[:-1] + ['config.yaml'])
            path = default_config_path

        if path is not None:
            with open(path, 'r') as file:
                kwargs = yaml.safe_load(file)
        for key in kwargs.keys():
            val = kwargs[key]

            # Special Cases -----------------------
            if key == 'dtype':
                val = torch.__dict__[val]
            # -------------------------------------

            self.__dict__[key] =  Config(depth=depth+1,**val) if isinstance(val,dict) else val
    def __repr__(self):
        res = ''
        for key in self.__dict__:
            if key[0] != "_":
                tabs = "".join(['\t' for _ in range(self._depth)])
                res+=f'\n{tabs}| {key}: {self.__dict__[key]}'
        # Root case -----------------------------------------------------------------------
        if self._is_root: res = '\nConfiguration:' + res + '\n'
        # ----------------------------------------------------------------------------------
        return res

    def __getitem__(self, key):
        return self.__dict__[key]


    def load_dict(self):
        res = {}
        for key, val in self.__dict__.items():
            if key[0]!="_":  res[key] = val
        return res
    def set_option(self,target_key,target_val):
        """
        Searches structure for target attribute name
            EX:  CFG.set_option('num_episodes',1)
            NOT !!! CFG.set_option('Training.num_episodes',1)
            :param target_key: attribute name (no header)
            :param target_val:
            :return: (bool) found target attribute
        """
        found_target = False
        for key in self.__dict__.keys():
            attr = self.__dict__[key]
            if isinstance(attr,IDQN.Config):
                found_target = attr.set_option(target_key,target_val) # CLASS
            elif key==target_key:
                self.__dict__[key] = target_val
                found_target = True
            if found_target: break
        # Root case -----------------------------------------------------------------------
        if self._is_root and not found_target:
            logging.warning(f'Config attribute not found - Unable to set {target_key}={target_val}')
        # ----------------------------------------------------------------------------------
        return found_target

CFG = Config()


########################################################################################
## Run Training Algorithm ##############################################################
########################################################################################
def run():
    from .train import run_IDQN
    for iworld in CFG.WORLDS:
        run_IDQN(iworld)

