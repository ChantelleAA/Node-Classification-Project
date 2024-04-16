#!/bin/sh

echo "GCN"

echo "Cora"
echo "===="

python gcn.py --dataset='../data/'  --optimizer=Adam --logger=GCN-Cora1-Adam

python gcn.py --dataset='../data/'  --optimizer=Adam --hyperparam=gamma --logger=GCN-Cora1-Adam

python gcn.py --dataset='../data/'  --optimizer=Adam --preconditioner=KFAC --hyperparam=eps --logger=GCN-Cora1-Adam-KFAC

python gcn.py --dataset='../data/'  --optimizer=Adam --preconditioner=KFAC --hyperparam=gamma --logger=GCN-Cora1-Adam-KFAC


