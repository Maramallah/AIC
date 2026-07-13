# Inference Notes

Default inference is the locked v6-conservative exact reproduction path.

Do not change:
- strict checkpoint loading
- direct switch_to_deploy()
- eval().cuda()
- force_cuda()
- torch.no_grad()
- preprocessing CUDA behavior

Expected released-checkpoint health check:
- Matched: 74293/74293
- EMA updates: 21517
- Lost recoveries: 133
