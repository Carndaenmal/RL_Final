import itertools
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from .components.environments import GFlowNetEnv, GraphEnv
from .components.evaluation import (
    compare_graph_distribution,
    compare_graphs,
    compare_graphs_bayesian_cover,
    compare_graphs_bayesian_shd,
    compute_graphs_bayesian_diversity,
    compute_graphs_sparsity,
)

from .components.structural_equations import (
    HyperStructuralEquationModel,
    LinearStructuralEquationModel,
)

from .components.energy import (
    PerNodeSimpleAnalyticBayesVelocityEnergy,
    SimpleAnalyticBayesVelocityEnergy,
)

from .parallel_energy_gfn_module import PerNodeParallelTrainableCausalGraphGFlowNetModule


class PerNodeDynGFN(PerNodeParallelTrainableCausalGraphGFlowNetModule):
    
    def __init__(
        self,
        dm_conf,
        bias: bool = True,
        env_batch_size: int = 64,
        eval_batch_size: int = 1000,
        uniform_backwards: float = False,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        alpha: float = 0.0,
        temperature: float = 1.0,
        temper_period: int = 1.0,
        prior_lambda: float = 1.0,
        beta: float = 1e-4,
        gfn_freq: int = 10,
        energy_freq: int = 10,
        load_pretrain: bool = False,
        pretraining_epochs: int = 15,
        full_posterior_eval: bool = False,
        debug_use_shd_energy: bool = False,
        analytic_use_simple_mse_energy: bool = False,
        **kwargs,
    ) -> None:


        super().__init__(
            dm_conf.p,
            LinearStructuralEquationModel(dm_conf.p, bias=bias),
            env_batch_size,
            eval_batch_size,
            uniform_backwards,
            hidden_dim,
            lr,
            alpha,
            temperature,
            temper_period,
            prior_lambda,
            beta,
            gfn_freq,
            energy_freq,
            load_pretrain,
            pretraining_epochs,
            full_posterior_eval,
            debug_use_shd_energy,
            analytic_use_simple_mse_energy,
            **kwargs
        )

        self.n_dim = dm_conf.p
        self.energy = SimpleAnalyticBayesVelocityEnergy(self.n_dim, beta, prior_lambda, temperature)
        self.node_energy = PerNodeSimpleAnalyticBayesVelocityEnergy(self.n_dim, beta, prior_lambda, temperature)
        
        # construct GFN models
        self.encoder = nn.ModuleList([nn.Sequential(
            nn.Linear(self.n_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            ) for i in range(self.n_dim)
        ])

        self.Pf = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, self.n_dim + 1))
            for i in range(self.n_dim)
        ])

        self.Pf_ = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, self.n_dim + 1))
            for i in range(self.n_dim)
        ])
        self.Pf_.load_state_dict(self.Pf.state_dict())

        self.save_hyperparameters(ignore=["structural_eq_model"], logger=False)
        self.automatic_optimization = False


    def configure_optimizers(self):
        gfn_opt = torch.optim.Adam(
            [
                {"params": self.Pf.parameters()},
                {"params": self.encoder.parameters()},
            ],
            lr=self.hparams.lr
        )
        return gfn_opt
    
    def training_step(self, batch: Any, batch_idx: int):

        def fmask(s):
            mask = torch.ones_like(s)
            mask[:, :-1] = 1 - s[:, :-1]                                     # existing edges cannot be added
            mask[:, :-1] *= 1 - s[:, -1].expand(self.n_dim, -1).T      # if terminal next action has to be terminal
            return mask

        def bmask(s):  
            mask = torch.ones_like(s)
            mask[:, :-1] = s[:, :-1]                                           # only existing edges can be removed   
            mask[:, :-1] *= 1 - s[:, -1].expand(self.n_dim, -1).T        # if terminal prev action has to be terminal               
            return mask

        bs = batch[0].size(0)

        G = torch.empty(bs, self.n_dim, self.n_dim, device=next(self.encoder.parameters()).device)
        for i in range(self.n_dim):
            
            state = torch.zeros(bs, self.n_dim + 1, device=next(self.encoder.parameters()).device)
            log_reward = torch.zeros(bs, device=next(self.encoder.parameters()).device)
            time = 0
            
            while not torch.all(state[:, -1]):
                
                # -- Actor -------
                # Compute forward prob from current state
                x = self.encoder[i](state[:, :-1])
                unnorm_pf = self.Pf[i](x)
                pf = F.softmax(unnorm_pf - 1e8 * (1 - fmask(state)), dim=1)
                log_pf = F.log_softmax(unnorm_pf - 1e8 * (1 - fmask(state)), dim=1)
                # ----------------
                
                # Sample action
                action = F.one_hot(torch.multinomial(pf, num_samples=1).squeeze(), num_classes = self.n_dim + 1)
                next_state = (state + action).clamp(0, 1)
                next_x = self.encoder[i](next_state[:, :-1])

                # -- Critic ------
                # Compute backward prob from next state
                log_pb = F.log_softmax(torch.ones(bs, self.n_dim + 1, device=next(self.encoder.parameters()).device) - 1e8 * (1 - bmask(next_state)), dim=1)
                # Reward of next state
                next_log_reward = - self.node_energy(next_state[:, :-1].unsqueeze(1), batch, node_idx=i)
                # Compute terminating probs
                log_terminating_prob = log_pf[:, -1]
                with torch.no_grad():
                    next_log_pf = F.log_softmax(self.Pf_[i](next_x) - 1e8 * (1 - fmask(next_state)), dim=1)
                    next_log_terminating_prob = next_log_pf[:, -1]
                # Compute detailed balance loss
                error = next_log_reward - log_reward
                error += (log_pb * action).sum(1) - next_log_terminating_prob
                error -= (log_pf * action).sum(1) - log_terminating_prob
                db_loss = F.huber_loss(error, target=torch.zeros_like(error), delta=1.0, reduction="none")
                entropy = - (pf * log_pf).sum(1)
                loss = db_loss - self.hparams.alpha * entropy
                # ---------------

                self.optimizers().zero_grad()
                self.manual_backward(loss.sum())
                self.optimizers().step()
                if time % self.hparams.gfn_freq == 0:
                    self.Pf_[i].load_state_dict(self.Pf[i].state_dict())
                
                state = next_state
                log_reward = next_log_reward
                time += 1

            G[:, i] = state[:, :-1]

        loss = self.energy(G, batch, return_mse=True).sum()
        self.log_dict(
            {
                "train/loss": loss,
                "train/reward": log_reward.mean(), 
                "train/avg_shd": torch.sum(torch.abs(batch[2][0] - G.mean(0))),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_eval_start(self):

        for i in range(self.n_dim):
            self.gfn_model[i].rep.load_state_dict(self.encoder[i].state_dict())

            w, b = self.Pf[i][0].state_dict()['weight'], self.Pf[i][0].state_dict()['bias']
            self.gfn_model[i].forward_prob[0].load_state_dict({'weight': w[:-1] , 'bias': b[:-1]})
            self.gfn_model[i].stop[0].load_state_dict({'weight': w[-1:] , 'bias': b[-1:]})

        super().on_eval_start()
