import torch
import predata
import model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch.utils.data
import torch.nn
import random
import torch.backends.cudnn


def set_random_seed(val=42):
    """freeze random, set cuda to deterministic mode"""
    random.seed(val)
    np.random.seed(val)
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    set_random_seed()
    data_root = input(str("data dir:"))
    path_model_save = input(str('path to save model'))
    # create dataframes
    df_train, df_validate, df_test = predata.prepare_data(data_root)

    # prepare data
    BATCH_SIZE = 10

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # prepare datasets
    train_dataset = predata.FlowerDataset(data_root, df_train, transforms=train_transforms)
    val_dataset = predata.FlowerDataset(data_root, df_validate, transforms=val_test_transforms)
    test_dataset = predata.FlowerDataset(data_root, df_test, transforms=val_test_transforms)
    # create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=BATCH_SIZE)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                 num_workers=BATCH_SIZE)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers=BATCH_SIZE)

    print("info: train data [batches:{}, files{}\n"
          "validation data [batches:{}, files{}\n"
          "test data [batches:{}, files{}\n".format(train_dataloader, train_dataset,
                                                    val_dataloader, val_dataset,
                                                    test_dataloader, test_dataset))

    # create model and set params
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.flower_net50()
    model = model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_history_loss = []
    train_history_acc = []
    val_history_loss = []
    val_history_acc = []

    num_epochs = 100
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)
        scheduler.step()

        running_loss = 0.
        running_acc = 0.

        for inputs, labels in tqdm(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                predicts = model(inputs)
                loss_value = loss(predicts, labels)
                predicts_class = predicts.argmax(dim=1)
                loss_value.backward()
                optimizer.step()

            running_loss += loss_value.item()
            running_acc += (predicts_class == labels.data).float().mean()

        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = running_acc / len(train_dataloader)

        train_history_loss.append(epoch_loss)
        train_history_acc.append(epoch_acc)
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc), flush=True)
        if epoch % 10 == 0:
            model.save_checkpoint(epoch, model, optimizer, epoch_loss, path_model_save)

    print('end train')


