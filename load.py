import torch
path='/home/han006/experiment_v3/spie/kmeans/model_best.pth.tar'
checkpoint=torch.load(path)
model=checkpoint['best_acc1']
print(model)