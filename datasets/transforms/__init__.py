from .cifar import cifar_train_ccrop, cifar_train_rcrop, cifar_linear, cifar_test
from .imagenet import imagenet_pretrain_rcrop, imagenet_pretrain_ccrop, imagenet_linear_train, \
    imagenet_val, imagenet_eval_boxes
from .underwater import underwater_train_ccrop, underwater_train_rcrop, underwater_linear, underwater_test
from .underwater_full import underwater_train_full_ccrop, underwater_train_full_rcrop, underwater_full_linear, underwater_full_test
from .build import build_transform
