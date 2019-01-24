'''Data Loader parameter'''
# Multiple threads loading data
workers_tr = 4
workers_va = 4
# Data augmentation
flip_prob = 0.5
crop_size = 0

'''GNN parameter'''
use_gnn = True
gnn_iterations = 3
gnn_k = 64
mlp_num_layers = 1

'''Model parameter'''
use_bootstrap_loss = False
bootstrap_rate = 0.25
use_gpu = True
class_weights = [0.0] + [1.0 for i in range(13)]
nclasses = len(class_weights)

'''Optimizer parameter'''
base_initial_lr = 5e-4
gnn_initial_lr = 1e-3
betas = [0.9, 0.999]
eps = 1e-08
weight_decay = 1e-4
lr_schedule_type = 'exp'
lr_decay = 0.9
lr_patience = 10