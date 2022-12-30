from collections import OrderedDict
from typing import Any, List, Optional, Union, Tuple
import concurrent.futures
import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import decaf.logger as log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_nonlin(name: str) -> nn.Module:
    if name == "none":
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "softmax":
        return nn.Softmax(dim=-1)
    else:
        raise ValueError(f"Unknown nonlinearity {name}")


class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, data: torch.Tensor) -> torch.Tensor:
        E = torch.linalg.matrix_exp(data)
        f = torch.trace(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=data.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (E,) = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input


trace_expm = TraceExpm.apply

#
# class EmbeddingBlock(nn.Module):
#     def __init__(self, h_dim, feature_types: Optional[List[int]], msk: Optional[torch.Tensor]=None):
#         super(EmbeddingBlock, self).__init__()
#         # noise is a feature
#         ft = feature_types
#         if msk is None:
#             msk = [1] * len(ft)
#         self.msk = msk
#         self.ft = ft
#         self.h_dim = h_dim
#         self.encoders = nn.ModuleList([
#             nn.Sequential(
#                 # nn.Linear(1 if t == -1 else t, 32), nn.ReLU(inplace=True),
#                 nn.Linear(1, 8), nn.ReLU(inplace=True),
#                 nn.Linear(8, 2), nn.ReLU(inplace=True)
#             ) if msk[i] else None for i, t in enumerate(ft)])
#         self.fc = nn.Sequential(
#             nn.Linear(2 * len(ft) + 1, h_dim),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x: torch.Tensor, z: torch.Tensor, msk=None, biased_edges=None):
#         # mask
#         if msk is not None:
#             assert (self.msk != msk).sum() == 0
#             x = x * msk
#         # debiasing
#         if biased_edges is not None:
#             for j in biased_edges:
#                 x_j = x[:, j]
#                 perm = torch.randperm(len(x_j))
#                 x[:, j] = x_j[perm]
#         # embed
#         # out = B x Num_features x Hidden_dims_per_feature -> B x Hidden_dims
#         out = torch.zeros((x.size(0), 2 * len(self.ft) + 1), device=DEVICE)
#
#         def fn(item):
#             idx, enc, xx = item
#             return idx, enc(xx)
#
#         with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
#             # Use map() to apply the process_item function to each item in the data
#             results = list(executor.map(fn, [(i, self.encoders[i], x[:, i].unsqueeze(1))
#                                              for i in range(len(self.ft)) if msk[i] == 1]))
#         for i, o in results:
#             out[:, (i << 1):(i << 1) + 2] = o
#         out[:, -1] = z
#         out = self.fc(out)
#         return out
#
#
# class GaussianEncoder(nn.Module):
#     def __init__(self, h_dim=4):
#         super(GaussianEncoder, self).__init__()
#         self.h_dim = h_dim
#         self.fc = nn.Sequential(
#             nn.Linear(1, 16), nn.ReLU(inplace=True),
#             nn.Linear(16, 8)
#         )
#
#     # return o = B x 2 x h_dim, where o[:, 0] = mu; o[:, 1] = sigma
#     def forward(self, x):
#         return self.fc(x).view(x.size(0), 2, self.h_dim)

#
# class GaussianEmbedding(nn.Module):
#     def __init__(self, x_dim, h_dim, msk):
#         super(GaussianEmbedding, self).__init__()
#         if msk is None:
#             msk = [1] * x_dim
#         self.msk = msk
#         self.h_dim = h_dim
#         self.encoders = nn.ModuleList([GaussianEncoder() if msk[i] else None for i in range(len(msk))])
#         self.fc = nn.Sequential(
#             nn.Linear(4 * x_dim, h_dim),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x: torch.Tensor, z: torch.Tensor, msk=None, biased_edges=None):
#         # mask
#         if msk is not None:
#             assert (self.msk != msk).sum() == 0
#             x = x * msk
#         # debiasing
#         if biased_edges is not None:
#             for j in biased_edges:
#                 x_j = x[:, j]
#                 perm = torch.randperm(len(x_j))
#                 x[:, j] = x_j[perm]
#         # embed
#         # out = B x Num_features x Hidden_dims_per_feature -> B x Hidden_dims
#         out = torch.zeros((x.size(0), 4 * len(self.ft) + 1), device=DEVICE)
#
#         def fn(item):
#             idx, enc, xx = item
#             return idx, enc(xx)
#
#         with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
#             # Use map() to apply the process_item function to each item in the data
#             results = list(executor.map(fn, [(i, self.encoders[i], x[:, i].unsqueeze(1))
#                                              for i in range(len(self.ft)) if msk[i] == 1]))
#         for i, o in results:
#             out[:, (i << 2):(i << 2) + 4] = o
#         out[:, -1] = z
#         out = self.fc(out)
#         return out

#


class Generator_causal(nn.Module):
    def __init__(
            self,
            z_dim: int,
            x_dim: int,
            h_dim: int,
            f_scale: float = 0.1,
            dag_seed: list = [],
            nonlin_out: Optional[List] = None,
            feature_types: Optional[List[int]] = None,
            choice='1hot',
            g_dim=4,
    ) -> None:
        super().__init__()

        if nonlin_out is not None:
            out_dim = 0
            for act, length in nonlin_out:
                out_dim += length
            if out_dim != x_dim:
                raise RuntimeError("Invalid nonlin_out")

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.g_dim = g_dim
        self.nonlin_out = nonlin_out

        def block(in_feat: int, out_feat: int, normalize: bool = False) -> list:
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers

        if choice == 'gaussian':
            self.shared_enc = nn.Sequential(
                nn.Linear(h_dim, h_dim << 1), nn.ReLU(inplace=True),
                nn.Linear(h_dim << 1, h_dim << 1), nn.ReLU(inplace=True),
                nn.Linear(h_dim << 1, self.g_dim << 1)
            ).to(DEVICE)
            self.shared_dec = nn.Sequential(
                nn.Linear(g_dim, h_dim), nn.ReLU(inplace=True),
                # nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),
                nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),
            ).to(DEVICE)

        else:
            self.shared = nn.Sequential(*block(h_dim, h_dim), *block(h_dim, h_dim)).to(
                DEVICE
            )

        if len(dag_seed) > 0:
            # M_init = torch.zeros(x_dim, x_dim)
            # M_init[torch.eye(x_dim, dtype=bool)] = 0
            M_init = torch.zeros(x_dim, x_dim)
            for pair in dag_seed:
                M_init[pair[0], pair[1]] = 1

            M_init = M_init.to(DEVICE)
            self.M = torch.nn.parameter.Parameter(M_init, requires_grad=False).to(
                DEVICE
            )
        else:
            M_init = torch.rand(x_dim, x_dim) * 0.2
            M_init[torch.eye(x_dim, dtype=bool)] = 0
            M_init = M_init.to(DEVICE)
            self.M = torch.nn.parameter.Parameter(M_init).to(DEVICE)

        self.choice = choice

        if choice == '1hot':
            self.fc_i = nn.ModuleList(
                # [EmbeddingBlock(h_dim, feature_types, self.M[:, i]) for i in range(self.x_dim)]
            )
        elif choice == 'gaussian':
            self.fc_i = nn.ModuleList(
                [nn.Linear(x_dim, h_dim) for _ in range(self.x_dim)]
            )
        else:
            self.fc_i = nn.ModuleList(
                [nn.Linear(x_dim + 1, h_dim) for _ in range(self.x_dim)]
            )
        self.fc_f = nn.ModuleList([nn.Linear(h_dim, 1) for _ in range(self.x_dim)])

        def _fn(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.weight.data *= f_scale

        if self.choice == 'gaussian':
            self.shared_enc.apply(_fn)
            self.shared_dec.apply(_fn)
        else:
            self.shared.apply(_fn)

        for i, layer in enumerate(self.fc_i):
            if choice in ['original', 'gaussian']:
                torch.nn.init.xavier_normal_(layer.weight)
                layer.weight.data *= f_scale
                layer.weight.data[:, i] = 1e-16
            else:
                layer.apply(_fn)

        for i, layer in enumerate(self.fc_f):
            layer.apply(_fn)

    def sequential(
            self,
            x: torch.Tensor,
            z: torch.Tensor,
            gen_order: Union[list, dict, None] = None,
            biased_edges: dict = {},
            requires_kl=False
    ) -> torch.Tensor:
        out = x.clone().detach()
        z = torch.randn((x.size(0), x.size(1), 1), device=DEVICE)

        if gen_order is None:
            gen_order = list(range(self.x_dim))

        kl_sum = torch.tensor(0.).to(DEVICE)
        cached_means = torch.zeros((self.x_dim, x.size(0), self.g_dim), device=DEVICE)
        cached_sigmas = torch.zeros((self.x_dim, x.size(0), self.g_dim), device=DEVICE)
        flag = 0
        for i in gen_order:
            if self.choice == '1hot':
                x = out.clone()
                out_i = self.fc_i[i](x, z[:, i],
                                     self.M[:, i], biased_edges[i] if i in biased_edges else None)
            else:
                x_masked = out.clone() * self.M[:, i]
                x_masked[:, i] = 0.0
                # sample from distribution means...
                # permutation
                if i in biased_edges:
                    if self.choice == 'gaussian':
                        for j in biased_edges[i]:
                            z = cached_means[j] + torch.randn_like(cached_sigmas[j]) * cached_sigmas[j]
                            x_masked[:, j] = torch.sigmoid(self.fc_f[j](self.shared_dec(z))).squeeze()
                        # for j in biased_edges[i]:
                        #     z = torch.randn((x.size(0), self.g_dim), device=DEVICE)
                        #     x_masked[:, j] = torch.sigmoid(self.fc_f[j](self.shared_dec(z))).squeeze()
                            # if flag == 0:
                            #     flag = 1
                            #     print('DE BIAS!!!')
                        # for j in biased_edges[i]:
                        #     x_j = x_masked[:, j]
                        #     perm = torch.randperm(len(x_j))
                        #     x_masked[:, j] = x_j[perm]
                        # if flag == 0:
                        #     flag = 1
                        #     print('B')
                    else:
                        # idx = torch.tensor(biased_edges[i], dtype=torch.long)
                        # x_b = x_masked[:, idx]
                        # perm = torch.randperm(len(x_b))
                        # x_masked[:, idx] = x_b[perm]
                        for j in biased_edges[i]:
                            x_j = x_masked[:, j]
                            perm = torch.randperm(len(x_j))
                            x_masked[:, j] = x_j[perm]

                if self.choice == 'gaussian':
                    out_i = self.fc_i[i](x_masked)
                else:
                    out_i = self.fc_i[i](torch.cat([x_masked, z[:, i, :]], dim=1))
                out_i = F.relu(out_i, inplace=True)
            if self.choice == 'gaussian':
                out_i = self.shared_enc(out_i)
                mean, tao = out_i[:, :self.g_dim], out_i[:, self.g_dim:]
                sigma = torch.exp(tao * 0.5)
                cached_means[i] = mean.detach().clone()
                cached_sigmas[i] = sigma.detach().clone()
                # sigma^2 = exp(tao)
                # log sigma^2 = tao
                if requires_kl:
                    kl = 1. * (mean * mean + torch.exp(tao) - tao - 1.).sum(dim=1)
                    kl_sum += kl.mean()
                out_i = mean + torch.randn_like(sigma) * sigma
                out_i = self.shared_dec(out_i)
            else:
                out_i = self.shared(out_i)
            out_i = torch.sigmoid(self.fc_f[i](out_i)).squeeze()
            out[:, i] = out_i

        # if self.nonlin_out is not None:
        #     split = 0
        #     for act_name, step in self.nonlin_out:
        #         activation = get_nonlin(act_name)
        #         out[..., split: split + step] = activation(
        #             out[..., split: split + step]
        #         )
        #
        #         split += step
        #
        #     if split != out.shape[-1]:
        #         raise ValueError("Invalid activations")
        if requires_kl and self.choice == 'gaussian':
            return out, kl_sum / len(gen_order)
        return out


class Discriminator(nn.Module):
    def __init__(self, x_dim: int, h_dim: int) -> None:
        super().__init__()
        h_dim = 200
        self.model = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, 1),
        )

        for layer in self.model.parameters():
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer)

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        return self.model(x_hat)


class DECAF(pl.LightningModule):
    def __init__(
            self,
            input_dim: int,
            dag_seed: list = [],
            h_dim: int = 200,
            lr: float = 1e-3,
            b1: float = 0.5,
            b2: float = 0.999,
            batch_size: int = 32,
            lambda_gp: float = 10,
            lambda_privacy: float = 1,
            eps: float = 1e-8,
            alpha: float = 1,
            rho: float = 1,
            weight_decay: float = 1e-2,
            grad_dag_loss: bool = False,
            l1_g: float = 0,
            l1_W: float = 1,
            nonlin_out: Optional[List] = None,
            feature_types: Optional[List[int]] = None,
            choice='1hot'
    ):
        super().__init__()
        self.save_hyperparameters()

        self.iterations_d = 0
        self.iterations_g = 0

        log.info(f"dag_seed {dag_seed}")

        self.x_dim = input_dim
        self.z_dim = self.x_dim

        log.info(
            f"Setting up network with x_dim = {self.x_dim}, z_dim = {self.z_dim}, h_dim = {h_dim}"
        )
        # networks
        self.generator = Generator_causal(
            z_dim=self.z_dim,
            x_dim=self.x_dim,
            h_dim=h_dim,
            dag_seed=dag_seed,
            nonlin_out=nonlin_out,
            feature_types=feature_types,
            choice=choice
        ).to(DEVICE)
        self.discriminator = Discriminator(x_dim=self.x_dim, h_dim=h_dim).to(DEVICE)

        self.dag_seed = dag_seed

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.generator(x, z)

    def gradient_dag_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the gradient of the output wrt the input. This is a better way to compute the DAG loss,
        but fairly slow atm
        """
        x.requires_grad = True
        z.requires_grad = True
        gen_x = self.generator(x, z)
        dummy = torch.ones(x.size(0))
        dummy = dummy.type_as(x)

        W = torch.zeros(x.shape[1], x.shape[1])
        W = W.type_as(x)

        for i in range(x.shape[1]):
            gradients = torch.autograd.grad(
                outputs=gen_x[:, i],
                inputs=x,
                grad_outputs=dummy,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            W[i] = torch.sum(torch.abs(gradients), axis=0)

        h = trace_expm(W ** 2) - self.hparams.x_dim

        return 0.5 * self.hparams.rho * h * h + self.hparams.alpha * h

    def compute_gradient_penalty(
            self, real_samples: torch.Tensor, fake_samples: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1)
        alpha = alpha.expand(real_samples.size())
        alpha = alpha.type_as(real_samples)
        # Get random interpolation between real and fake samples
        interpolates = (
                alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones(real_samples.size(0), 1)
        fake = fake.type_as(real_samples)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def privacy_loss(
            self, real_samples: torch.Tensor, fake_samples: torch.Tensor
    ) -> torch.Tensor:
        return -torch.mean(
            torch.sqrt(
                torch.mean((real_samples - fake_samples) ** 2, dim=1)
                + self.hparams.eps
            )
        )

    def get_W(self) -> torch.Tensor:
        return self.generator.M

    def dag_loss(self) -> torch.Tensor:
        W = self.get_W()
        h = trace_expm(W ** 2) - self.x_dim
        l1_loss = torch.norm(W, 1)
        return (
                0.5 * self.hparams.rho * h ** 2
                + self.hparams.alpha * h
                + self.hparams.l1_W * l1_loss
        )

    def sample_z(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.z_dim, device=DEVICE)

    @staticmethod
    def l1_reg(model: nn.Module) -> float:
        l1 = torch.tensor(0.0, requires_grad=True)
        for name, layer in model.named_parameters():
            if "weight" in name:
                l1 = l1 + layer.norm(p=1)
        return l1

    def gen_synthetic(self, x: torch.Tensor, biased_edges: dict = {}) -> torch.Tensor:
        self.generator = self.generator.to(DEVICE)
        x = x.to(DEVICE)
        gen_order = self.get_gen_order()
        return self.generator.sequential(
            x,
            self.sample_z(x.shape[0]).type_as(x),
            gen_order=gen_order,
            biased_edges=biased_edges,
            requires_kl=False
        )

    def get_dag(self) -> np.ndarray:
        return np.round(self.get_W().detach().cpu().numpy(), 3)

    def get_gen_order(self) -> list:
        dense_dag = np.array(self.get_dag())
        dense_dag[dense_dag > 0.5] = 1
        dense_dag[dense_dag <= 0.5] = 0
        G = nx.from_numpy_matrix(dense_dag, create_using=nx.DiGraph)
        gen_order = list(nx.algorithms.dag.topological_sort(G))
        return gen_order

    def training_step(
            self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int
    ) -> OrderedDict:
        # sample noise
        z = self.sample_z(batch.shape[0])
        z = z.type_as(batch)
        generated_batch = self.generator.sequential(batch, z, self.get_gen_order(),
                                                    requires_kl=self.hparams.choice == 'gaussian')
        kl = 0
        if self.hparams.choice == 'gaussian':
            generated_batch, kl = generated_batch

        # train generator
        if optimizer_idx == 0:
            self.iterations_d += 1
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            real_loss = torch.mean(self.discriminator(batch))
            fake_loss = torch.mean(self.discriminator(generated_batch.detach()))

            # discriminator loss
            d_loss = fake_loss - real_loss

            # add the gradient penalty
            d_loss += self.hparams.lambda_gp * self.compute_gradient_penalty(
                batch, generated_batch
            )
            if torch.isnan(d_loss).sum() != 0:
                raise ValueError("NaN in the discr loss")

            return d_loss
        elif optimizer_idx == 1:
            # sanity check: keep track of G updates
            self.iterations_g += 1

            # adversarial loss (negative D fake loss)
            g_loss = -torch.mean(
                self.discriminator(generated_batch)
            )  # self.adversarial_loss(self.discriminator(self.generated_batch), valid)

            # add privacy loss of ADS-GAN
            # g_loss += self.hparams.lambda_privacy * self.privacy_loss(
            #     batch, generated_batch
            # )

            # add l1 regularization loss
            if self.hparams.l1_g > 1e-8:
                g_loss += self.hparams.l1_g * self.l1_reg(self.generator)

            # add kl loss
            if self.hparams.choice == 'gaussian':
                g_loss += 0.1 * kl

            if len(self.dag_seed) == 0:
                if self.hparams.grad_dag_loss:
                    g_loss += self.gradient_dag_loss(batch, z)
            if torch.isnan(g_loss).sum() != 0:
                raise ValueError("NaN in the gen loss")

            return g_loss
        else:
            raise ValueError("should not get here")

    def configure_optimizers(self) -> tuple:
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        weight_decay = self.hparams.weight_decay

        opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=lr,
            betas=(b1, b2),
            weight_decay=weight_decay,
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=lr,
            betas=(b1, b2),
            weight_decay=weight_decay,
        )
        return (
            {"optimizer": opt_d, "frequency": 5},
            {"optimizer": opt_g, "frequency": 1},
        )
