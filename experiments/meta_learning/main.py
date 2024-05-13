import torch
from torchvision.transforms import transforms
from maml_parallel import MAML
import torchvision
import torch.optim as optim
import torch.nn as nn
from functools import partial

import utils
from models.cnn import CNN
from models.mlp import MLP

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

RUN_NAME = "omniglot_cnn"
logfile = "maml_omniglot"

if __name__ == '__main__':
    iterations = 60000
    n_way = 5 
    shots = 5
    batch_size = 16
    num_adaption_steps = 5
    num_test_adaption_steps = 5
    seed = 42
    greyscale = True
    inner_lr = 0.1 # 0.1 for omniglot, 1e-3 for miniimagenet
    meta_lr = 1e-3
    log = []
    

    utils.set_seed(seed)

    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
    ])

    train_loader, val_loader, test_loader = utils.load_data(name="Omniglot",
                                                            shots=shots,
                                                            n_ways=n_way,
                                                            batch_size=batch_size,
                                                            root='data',
                                                            num_workers=4,
                                                            train_transform=transform,
                                                            test_transform=transform)
    

    # model = torchvision.models.resnet50(norm_layer=partial(nn.BatchNorm2d, track_running_stats=False))
    # model.fc = nn.Linear(model.fc.in_features, n_way)
    # model.to(device)

    model = CNN(in_channels=1, output_size=n_way) 
    model.to(device)

    maml = MAML()

    model_meta_opt = optim.Adam(model.parameters(), lr=1e-3)

    train_loader = iter(train_loader)

    for epoch in range(iterations):
        loss, acc = maml.train(train_loader, model, device, model_meta_opt, epoch, num_adaption_steps, inner_lr)

        log.append(
        {
            "epoch": epoch,
            "loss": loss,
            "acc": acc,
            "mode": "train",
        })

        if epoch % 1000 == 0:
            loss, acc = maml.test(val_loader, model, device, num_adaption_steps, inner_lr)
            
            print(f"val_loss: {loss} - val_acc: {acc}")
            log.append(
            {
                "epoch": epoch,
                "loss": loss,
                "acc": acc,
                "mode": "val",
            })


    loss, acc = maml.test(test_loader, model, device, epoch, num_adaption_steps)

    print(f"test_loss: {loss} - test_acc: {acc}")
    log.append(
    {
        "epoch": epoch,
        "loss": loss,
        "acc": acc,
        "mode": "test",
    })


    torch.save(model.state_dict(), f'trained/{RUN_NAME}.pt')
    utils.plot(log, logfile)
    