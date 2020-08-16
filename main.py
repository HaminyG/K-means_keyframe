from videoDataset import VideoDataset
import torch
import cv2
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import shutil
import copy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
global val_total
from torch.optim.lr_scheduler import StepLR
sum_path = "runs/slr_cnn3d_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
writer = SummaryWriter(sum_path)
n_gpu = torch.cuda.device_count()
device = torch.device("cuda")


def train(model, device, train_loader, optimizer, epoch):
    total = 0
    correct = 0
    losses=[]
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader), 0):
        # l2_reg=torch.FloatTensor(1).clone().detach().requires_grad_(True).to(device)
        dataset, label = data
        dataset = dataset.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(dataset.float())
        output =output
        # for param in model.parameters():
        #     l2_reg=l2_reg+torch.pow(param,2).sum()
        loss = criterion(output, label)  # +l2_reg
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        training_acc = 100. * correct / total
        training_loss = sum(losses) / len(losses)
        writer.add_scalars('train Loss', {'train': training_loss}, epoch + 1)
        writer.add_scalars('train Accuracy', {'train': training_acc}, epoch + 1)
        print(
            'Train Epoch: {} [{}/{}]  Loss: {:.6f} Acc: {:.4f}% Correct: {} Total: {}'.format(
                epoch, batch_idx * len(dataset), len(training_dataloader.dataset), loss.item(), training_acc, correct, total
            )
        )
    return training_acc

def val(model, device, val_loader):
    model.eval()
    # calculate accuracy on validation set
    top1_val_correct,top5_val_correct, val_loss = 0, 0, 0
    with torch.no_grad():
        for val_data, val_label in val_loader:
            val_data = val_data.to(device)
            val_label = val_label.to(device)
            output = model(val_data.float())

            maxk = max((1, 5))
            y_resize = val_label.view(-1, 1)
            _, pred = output.topk(maxk, 1, True, True)
            top5_val_correct += torch.eq(pred, y_resize).sum().float().item()

            val_loss += criterion(output, val_label).item()
            _, val_predicted = torch.max(output, 1)
            top1_val_correct += (val_predicted == val_label).sum().item()
        val_mean_loss = val_loss / len(val_loader.dataset)
        top1_val_accuracy = 100. * top1_val_correct / len(val_loader.dataset)
        top5_val_accuracy = 100. * top5_val_correct / len(val_loader.dataset)
        writer.add_scalars('val Loss', {'val': val_mean_loss}, epoch + 1)
        writer.add_scalars('top1val Accuracy', {'val': top1_val_accuracy}, epoch + 1)
        writer.add_scalars('top5val Accuracy', {'val': top5_val_accuracy}, epoch + 1)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} top1({:.4f}%) top5({:.4f}%)\n'.format(
            val_mean_loss, top1_val_correct, len(val_loader.dataset), top1_val_accuracy,top5_val_accuracy
        )
        )
    return val_mean_loss, top1_val_accuracy

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':

    import argparse
    import os
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from models.new_model import r2plus1d_18_new
    from models.Conv3D import resnet18,r2plus1d_18
    from models.resnet2dlstm import ResCRNN
    #from r2plus1d import r2plus1d_18
    best_acc1 = 0
    global model
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/han006/experiment_v3/CSL5000_100class/dataset/CSL25000_500classes/xf500_color_video',
        type=str, help='Data path for training')
    parser.add_argument('--json_path', default='/home/han006/experiment_v3/CSL5000_100class/dataset/CSL25000_500classes/video_json_word',
        type=str, help='json path for training')
    parser.add_argument('--model', default='3dresnet18',required=False,
        type=str, help='3dresnet18, 3dresnet34, 3dresnet50，3dresnet101')
    parser.add_argument('--num_classes', default=500,
        type=int, help='Number of classes for testing')
    parser.add_argument('--batch_size', default=128,
        type=int, help='Batch size for training')
    parser.add_argument('--sample_size', default=112,
        type=int, help='Sample size for training')
    parser.add_argument('--epoch', default=1000,
        type=int, help='epoch number')
    parser.add_argument('--lr', default=0.0001,
        type=int, help='learning rate')
    args = parser.parse_args()

    videodir=args.data_path
    jsondir=args.json_path
    lr=args.lr
    epoch_number =args.epoch
    num_classes = args.num_classes
    batch_size = args.batch_size
    sample_size = args.sample_size


    train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((sample_size, sample_size)),
                                      transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
                                      #transforms.Normalize(mean=[0.6062, 0.6063, 0.615], std=[0.141, 0.136, 0.148])]
                                     )
    val_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((sample_size, sample_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
                                      #transforms.Normalize(mean=[0.6062, 0.6063, 0.615], std=[0.141, 0.136, 0.148])]
                                     )
    train_data = VideoDataset(videodir,jsondir, 3, 16, mode='train', transform=train_transform)
    val_data = VideoDataset(videodir, jsondir, 3, 16, mode='val', transform=val_transform)

    training_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=16, pin_memory=True)
    validation_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True,num_workers=16, pin_memory=True)

    if args.model == 'r2plus1d_18':
        model = cnn3d = r2plus1d_18(pretrained=False, num_classes=500)
    elif args.model == 'r2plus1d_18_new':
        model = r2plus1d_18_new(pretrained=False, progress=False)
    elif args.model == '3dresnet18':
        model = resnet18(pretrained=False, sample_size=sample_size,
            sample_duration=16, num_classes=500)
    elif args.model == 'ResCRNN':
        model=ResCRNN(sample_size=112, sample_duration=16, num_classes=100, arch="resnet18")
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3], dim=0).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    best_epoch = 0
    for epoch in tqdm(range(epoch_number)):
        acc = train(model, device, training_dataloader, optimizer, epoch)
        val_loss, val_acc = val(model, device, validation_dataloader)
        scheduler.step(epoch)
        is_best = val_acc > best_acc1
        if is_best:
            best_epoch = copy.deepcopy(epoch)
        best_acc1 = max(val_acc, best_acc1)
        save_checkpoint({
        'epoch': best_epoch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
        }, is_best)
