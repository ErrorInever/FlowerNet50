from torch import nn
import torch
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_drop_out=False, d_out_p=0.5):
        super(Bottleneck, self).__init__()
        self.use_drop_out = use_drop_out
        self.p = d_out_p
        # activation function
        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.d_out_1 = nn.Dropout2d(self.p)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.d_out_2 = nn.Dropout2d(self.p)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.d_out_3 = nn.Dropout2d(self.p)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.use_drop_out:
            out = self.d_out_1(out)
        out = self.act(out)
        out = self.bn2(self.conv2(out))
        if self.use_drop_out:
            out = self.d_out_2(out)
        out = self.act(out)
        out = self.bn3(self.conv3(out))
        if self.use_drop_out:
            out = self.d_out_3(out)
        out += self.shortcut(x)
        out = self.act(out)
        return out


class FlowerNet50(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_drop_out=False, d_out_p=0.5):
        super(FlowerNet50, self).__init__()
        self.use_drop_out = use_drop_out
        self.p = d_out_p
        self.in_planes = 64

        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.d_out_1 = nn.Dropout2d(self.p)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                use_drop_out=self.use_drop_out, d_out_p=0.15))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.use_drop_out:
            out = self.d_out_1(out)
        out = self.act(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    @staticmethod
    def save_checkpoint(epoch, model, optimizer, loss, path):
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, path)

    @staticmethod
    def load_checkpoint(model_inst, optimizer_inst, file_path, train=False):
        """
        :param model_inst: TheModelClass(*args, **kwargs)
        :param optimizer_inst: TheOptimizerClass(*args, **kwargs)
        :param file_path: path where model stored
        :param train: set model to train mode or evaluate
        """
        model = model_inst
        optimizer = optimizer_inst
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        if train:
            model.train()
        else:
            model.eval()

        return model, optimizer, epoch, loss


def flower_net50():
    return FlowerNet50(Bottleneck, [3, 4, 6, 3])


def flower_net50_d_out_15():
    return FlowerNet50(Bottleneck, [3, 4, 6, 3], use_drop_out=True, d_out_p=0.15)
