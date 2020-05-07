import pandas as pd
import torch
import main

# Hyper parameters
BATCH_SIZE = 20
LEARNING_RATE = 0.001  # The optimal learning rate for the Adam optimizer
BEST_EPOCH = 5
DROPOUT_RATE = 0.1

val_dataset = main.Dataset('dataset/prep_unbiased/val.txt')
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

net = main.Net(DROPOUT_RATE)
net.load_state_dict(torch.load('models/split_10_90_batchsize_' + str(BATCH_SIZE) + '_lr_' + str(LEARNING_RATE) + '_dropout_'+ str(DROPOUT_RATE) + '_epoch_' + str(BEST_EPOCH) +'.pkl'))

print(main.compute_loss(net, val_dataloader))
print()