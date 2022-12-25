import argparse

import networkx as nx
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch

from utils import gen_data_nonlinear, load_adult, load_adult_ex
from sklearn.model_selection import train_test_split

from decaf import DECAF
from decaf.data import DataModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h_dim", type=int, default=240)
    parser.add_argument("--lr", type=float, default=0.5e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--rho", type=float, default=2)
    parser.add_argument("--l1_W", type=float, default=1e-4)
    parser.add_argument("--logfile", type=str, default="default_log.txt")
    parser.add_argument("--name", type=str, default="decaf")
    parser.add_argument("--embed", type=str, default="original")
    parser.add_argument("--datasize", type=int, default=1000)
    args = parser.parse_args()

    # causal structure is in dag_seed
    dag_seed = [[0, 4], [0, 8], [0, 10], [0, 3], [0, 1], [0, 2], [0, 5],
                [1, 10],
                [2, 4], [2, 8], [2, 10], [2, 1], [2, 5],
                [3, 4], [3, 8], [3, 10], [3, 1], [3, 5], [3, 2],
                [4, 10],
                [5, 10],
                [6, 4], [6, 10], [6, 8], [6, 2], [6, 3],
                [7, 4], [7, 3], [7, 10], [7, 8], [7, 1], [7, 2], [7, 5],
                [8, 10],
                [9, 3], [9, 8], [9, 2], [9, 1], [9, 10], [9, 5]]

    # edge removal dictionary
    bias_dict = {10: [7]}  # This removes the edge into 6 from 3.

    # DATA SETUP according to dag_seed
    G = nx.DiGraph(dag_seed)
    # data = gen_data_nonlinear(G, SIZE=2000)
    data = load_adult_ex()
    data_train, data_test = train_test_split(data, test_size=2000, stratify=data['label'])
    dm = DataModule(data_train.values, num_workers=0)
    data_tensor = dm.dataset.x

    # sample default hyperparameters
    x_dim = dm.dims[0]
    z_dim = x_dim  # noise dimension for generator input. For the causal system, this should be equal to x_dim
    lambda_privacy = 0  # privacy used for ADS-GAN, not sure if necessary for us tbh
    lambda_gp = 10  # gradient penalisation used in WGAN-GP
    l1_g = 0  # l1 reg on sum of all parameters in generator
    weight_decay = 1e-2  # used by AdamW to regularise all network weights. Similar to L2 but for momentum-based optimization

    # causality settings
    grad_dag_loss = False

    # number_of_gpus = 1

    # model initialisation and train
    model = DECAF(
        dm.dims[0],
        dag_seed=dag_seed,
        h_dim=args.h_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        lambda_privacy=lambda_privacy,
        lambda_gp=lambda_gp,
        alpha=args.alpha,
        rho=args.rho,
        weight_decay=weight_decay,
        grad_dag_loss=grad_dag_loss,
        l1_g=l1_g,
        l1_W=args.l1_W,
        feature_types=[-1, 7, 16, 7, 14, 6, 5, 1, -1, 41, 1],
        choice=args.embed
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.epochs,
        # progress_bar_refresh_rate=1,
        profiler=False,
        callbacks=[],
    )
    if not os.path.exists(f'{args.name}.pth'):
        trainer.fit(model, dm)
        torch.save(model, f'{args.name}.pth')
    else:
       model = torch.load(f'{args.name}.pth')
    synth_dataset = (
        model.gen_synthetic(data_tensor, biased_edges={}).detach().cpu().numpy()
    )
    synth_dataset[:, -1] = synth_dataset[:, -1].astype(np.int8)

    synth_dataset = pd.DataFrame(synth_dataset,
                                 index=data_train.index,
                                 columns=data_train.columns)
    synth_dataset['sex'] = np.round(synth_dataset['sex'])
    synth_dataset['label'] = np.round(synth_dataset['label'])

    from eval_utils import eval_model
    print(eval_model(synth_dataset, data_test))
    print("Data generated successfully!")
