from torch.utils.serialization import load_lua

x = load_lua('/home/elijha/Documents/Data/MSCOCO/coco_gru18_bs64_cls0.5_ngf128_ndf128_a10_c512_80_net_T.t7',
             unknown_classes=True)
print()
