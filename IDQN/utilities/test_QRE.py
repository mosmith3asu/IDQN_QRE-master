import itertools
import logging

import numpy as np
import torch

device = 'cpu'
dtype = torch.float32
n_agents = 2

joint2solo = np.array(list(itertools.product(*[np.arange(5), np.arange(5)])), dtype=int)
solo2joint = np.zeros([5, 5], dtype=int)
for aJ, joint_action in enumerate(joint2solo):
    aR, aH = joint_action
    solo2joint[aR, aH] = aJ

ijoint = np.zeros([2, 5, 25], dtype=np.float32)

for k in range(n_agents):
    for ak in range(5):
        idxs = np.array(np.where(joint2solo[:, k] == ak)).flatten()
        ijoint[k, ak, idxs] = 1

ijoint = torch.as_tensor(ijoint, device=device)
solo2joint = torch.as_tensor(solo2joint, device=device)
joint2solo = torch.as_tensor(joint2solo, device=device)
tensor_type = {'device': device,'dtype':dtype}


def QRE(qAk):
    sophistocation = 5

    n_agents = 2
    n_joint_act = 25
    n_ego_act = 5
    rationality = 1
    n_batch = qAk.shape[0]

    pdAjointk = torch.ones([n_batch, n_agents, n_joint_act], **tensor_type) / n_joint_act

    qAego = torch.empty([n_batch, n_agents, n_ego_act], **tensor_type)
    pdAegok = torch.empty([n_batch, n_agents, n_ego_act], **tensor_type)


    for isoph in range(sophistocation):
        new_pdAjointk = torch.zeros([n_batch,n_agents,n_joint_act])
        for k in range(n_agents):
            this_k = k
            notk = int(not this_k)
            ijoint_batch = ijoint[this_k, :, :].T.repeat([n_batch, 1, 1])
            qAJ_conditioned = qAk[:, this_k, :] * pdAjointk[:, notk, :]
            qAego[:, this_k, :] = torch.bmm(qAJ_conditioned.unsqueeze(1), ijoint_batch).squeeze()
            pdAegok[:, this_k, :] = torch.special.softmax(rationality * qAego[:, this_k, :], dim=-1)
            new_pdAjointk[:, this_k, :] = torch.bmm(pdAegok[:,this_k, :].unsqueeze(1), torch.transpose(ijoint_batch/5, dim0=1, dim1=2)).squeeze()
            # print(f'p{this_k}:{new_pdAegok[0,this_k,:].detach().numpy().round(3)}')
            if not torch.all(torch.abs(1-torch.sum(new_pdAjointk[:,this_k,:],dim=-1)) < 0.01):
                logging.warning(f'Joint probs not sum to 1 in QRE [sum={torch.sum(new_pdAjointk[:,this_k,:],dim=-1)}]')
            if not torch.all(torch.abs(1 - torch.sum(pdAegok[:, this_k, :], dim=-1)) < 0.01):
                logging.warning(f'Ego probs not sum to 1 in QRE [sum={torch.sum(pdAegok[:,this_k,:],dim=-1)}]')
        pdAjointk = new_pdAjointk.clone().detach()

    return pdAegok  # .detach().long()  # pdAjointk


def main():
    sophistocation = 2
    n_batch = 3
    n_agents = 2
    n_joint_action = 25
    qAk = torch.zeros([n_batch,n_agents, n_joint_action])

    qAk[:, 0, 0] = 5
    qAk[:, 0, 24] = 10
    #
    qAk[:, 1, 0] = 10
    qAk[:, 1, 24] = 5

    # ia = 0
    # qAk[:, 0, :] = 5 * ijoint[0, ia, :]
    # # qAk[:, 0, :] = 100 * ijoint[0, ia, :]
    # qAk[:, 1, :] = 5 * ijoint[1, ia, :]
    # # qAk[:, 1, :] = 100 * ijoint[1, ia, :]


    pdAk = QRE(qAk)
    for k in range(n_agents): print(f'p{k}:{np.round(list((pdAk[0,k]).detach()),3)}')


def subfun():
    pass


if __name__ == "__main__":
    main()
