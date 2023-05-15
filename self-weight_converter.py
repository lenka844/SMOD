from os import replace
import pickle as pkl
import sys
import torch

checkpoint = torch.load('/home/ljh/self-detection/ContrastiveCrop/checkpoints/small/underwater/simsiam_ccrop_pretrain_4w/epoch_600.pth', map_location='cpu')
# state_dict = checkpoint['state_dict']
state_dict = checkpoint['simsiam_state']
for k in list(state_dict.keys()):
    # print(k)
    if k.endswith('num_batches_tracked'):
        del state_dict[k]
        continue
    # print(k)
    if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
        state_dict[k[len('module.encoder.'):]] = state_dict[k]
        # state_dict['backbone.'+k[len('module.encoder.'):]] = state_dict[k]
        # print('backbone.'+k[len('module.encoder.'):])
    del state_dict[k]


print(state_dict.keys())
torch.save(state_dict, "/home/ljh/self-detection/mmdetection/checkpoints/UN4W_cocopre.pth", _use_new_zipfile_serialization=False)
