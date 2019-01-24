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

ENCODER_PARAMS = [
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 32,
        'output_channels': 64,
        'downsample': True,
        'dropout_prob': 0.01
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 64,
        'output_channels': 64,
        'downsample': False,
        'dropout_prob': 0.01
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 64,
        'output_channels': 64,
        'downsample': False,
        'dropout_prob': 0.01
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 64,
        'output_channels': 64,
        'downsample': False,
        'dropout_prob': 0.01
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 64,
        'output_channels': 64,
        'downsample': False,
        'dropout_prob': 0.01
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 64,
        'output_channels': 128,
        'downsample': True,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 2
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 4
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 8
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 16
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 2
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 4
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 8
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    },
    {
        'internal_scale': 4,
        'use_relu': True,
        'asymmetric': False,
        'dilated': False,  # 16
        'input_channels': 128,
        'output_channels': 128,
        'downsample': False,
        'dropout_prob': 0.1
    }
]

DECODER_PARAMS = [
    {
        'input_channels': 128,
        'output_channels': 128,
        'upsample': False,
        'pooling_module': None
    },
    {
        'input_channels': 128,
        'output_channels': 64,
        'upsample': True,
        'pooling_module': None
    },
    {
        'input_channels': 64,
        'output_channels': 64,
        'upsample': False,
        'pooling_module': None
    },
    {
        'input_channels': 64,
        'output_channels': 64,
        'upsample': False,
        'pooling_module': None
    },
    {
        'input_channels': 64,
        'output_channels': 32,
        'upsample': True,
        'pooling_module': None
    },
    {
        'input_channels': 32,
        'output_channels': 32,
        'upsample': False,
        'pooling_module': None
    }
]