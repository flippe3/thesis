import torch
import torchmeta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def plot(log, filename):
    # Generally you should pull your plotting code out of your training
    # script but we are doing it here for brevity.
    df = pd.DataFrame(log)

    fig, ax = plt.subplots(figsize=(6, 4))
    train_df = df[df["mode"] == "train"]
    val_df = df[df["mode"] == "val"]
    test_df = df[df["mode"] == "test"]
    ax.plot(train_df["epoch"], train_df["acc"], label="Train")
    ax.plot(val_df["epoch"], val_df["acc"], label="Val")
    ax.plot(test_df["epoch"], test_df["acc"], label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(70, 100)
    fig.legend(ncol=2, loc="lower right")
    fig.tight_layout()
    print(f"--- Plotting accuracy to {filename}")
    fig.savefig("plots/"+filename)
    plt.close(fig)


def load_data(name, shots, n_ways, batch_size, train_transform, test_transform, root, num_workers=4):
    train_dataset = torchmeta.datasets.__dict__[name](
        root,
        num_classes_per_task=n_ways,
        meta_train=True,
        download=True,
        transform=train_transform, 
        target_transform=torchmeta.transforms.Categorical(num_classes=n_ways),
    )

    val_dataset = torchmeta.datasets.__dict__[name](
        root,
        num_classes_per_task=n_ways,
        meta_val=True,
        download=True,
        transform=test_transform, 
        target_transform=torchmeta.transforms.Categorical(num_classes=n_ways),
    )

    test_dataset = torchmeta.datasets.__dict__[name](
        root,
        num_classes_per_task=n_ways,
        meta_test=True,
        download=True,
        transform=test_transform, 
        target_transform=torchmeta.transforms.Categorical(num_classes=n_ways),
    )

    train_dataset = torchmeta.transforms.ClassSplitter(train_dataset, shuffle=True, num_train_per_class=shots, num_test_per_class=shots)
    val_dataset = torchmeta.transforms.ClassSplitter(val_dataset, shuffle=True, num_train_per_class=shots, num_test_per_class=shots)
    test_dataset = torchmeta.transforms.ClassSplitter(test_dataset, shuffle=True, num_train_per_class=shots, num_test_per_class=shots)

    train_dataloader = torchmeta.utils.data.BatchMetaDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torchmeta.utils.data.BatchMetaDataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = torchmeta.utils.data.BatchMetaDataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader