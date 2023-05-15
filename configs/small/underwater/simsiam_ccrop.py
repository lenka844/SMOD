# python DDP_simsiam_ccrop.py path/to/this/config

# model
# dim, pred_dim = 512, 128
dim, pred_dim = 1024, 512
# model = dict(type='ResNet', depth=50, num_classes=dim, maxpool=False, zero_init_residual=True)
model = dict(type='CbamSimAMnam', depth=50, num_classes=dim, maxpool=True, zero_init_residual=True)
simsiam = dict(dim=dim, pred_dim=pred_dim)
loss = dict(type='CosineSimilarity', dim=1)

# data
root = '/home/ljh/self-detection/data/UNDERWATER/1w_underwater/'
# root = '/home/ljh/self-detection/data/UNDERWATER/datatset'
mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)
batch_size = 2
num_workers = 4
data = dict(
    train=dict(
        ds_dict=dict(
            type='Underwater_boxes',
            root=root,
            train=True,
        ),
        rcrop_dict=dict(
            type='underwater_train_rcrop',
            mean=mean, std=std
        ),
        ccrop_dict=dict(
            type='underwater_train_ccrop',
            alpha=0.1,
            mean=mean, std=std
        ),
    ),
    eval_train=dict(
        ds_dict=dict(
            type='UnderWater',
            root=root,
            train=True,
        ),
        trans_dict=dict(
            type='underwater_test',
            mean=mean, std=std
        ),
    ),
)

# boxes
warmup_epochs = 100
loc_interval = 100
box_thresh = 0.10

# training optimizer & scheduler
epochs = 500
lr = 0.05
# lr = 0.5
fix_pred_lr = True
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-4)
lr_cfg = dict(  # passed to adjust_learning_rate(cfg=lr_cfg)
    type='Cosine',
    steps=epochs,
    lr=lr,
    decay_rate=0.1,
    # decay_steps=[100, 150]
    warmup_steps=0,
    # warmup_from=0.01
)


# log & save
log_interval = 20
save_interval = 100
work_dir = '/home/ljh/self-detection/ContrastiveCrop/checkpoints/small/underwater/Coordinate_attention'  # rewritten by args
resume = None
# resume = '/home/ljh/self-detection/ContrastiveCrop/checkpoints/small/underwater/simsiam_ccrop/epoch_190.pth'
load = None
port = 10001
