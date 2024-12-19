import torch
from torch import Tensor
from torch import optim
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18

from src.mlConfig.Const import mode_UsePretrainedWeight

class CRNN(nn.Module):

    def __init__(self, num_chars: int, mode_ImgRgb: bool, rnn_hidden_size=256, dropout=0.1):
        super().__init__()
        self.num_chars = num_chars
        self.rnn_hidden_size = rnn_hidden_size
        self.dropout = dropout

        # CNN Part 1
        resnet = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)  # must put outside?.. // placing outside drops faster in first few epochs?...
        if not mode_ImgRgb:
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet_modules = list(resnet.children())[:-3]
        self.cnn_p1 = nn.Sequential(*resnet_modules)

        # CNN Part 2
        self.cnn_p2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 6), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # RNN
        # // feels crnn is using encoder decoder though .... just that output -- context vector passing right? dk why asked gpt before said no .... //dde
        # self.linear1 = nn.Linear(1024, 256)
        self.linear1 = nn.Linear(1536, 256)
        self.rnn1 = nn.GRU(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, bidirectional=True, batch_first=True)
        self.rnn2 = nn.GRU(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, bidirectional=True, batch_first=True)
        self.linear2 = nn.Linear(self.rnn_hidden_size * 2, num_chars)

        # #>>
        if mode_UsePretrainedWeight:
            self.dict_module_ExcludeWeightInit: dict[nn.Module, str] = dict()
            for name, module in self.cnn_p1.named_modules():
                if not mode_ImgRgb and (module == resnet.conv1):
                    continue
                self.dict_module_ExcludeWeightInit[module] = name

    def forward(self, x: Tensor):
        # print(x.size())  # @old vs @new # torch.Size([32, 3, 32, 256]) # torch.Size([32, 3, 64, 512])

        x = self.cnn_p1(x)
        # print(x.size())  # torch.Size([-1, 256, 4, 13]) # torch.Size([32, 256, 4, 32])
        # //? idk the feat extraction,
        # 1. the feature map is only of height 4, is it too small? what it extracted?
        # 1. the width of the kernel is longer than the height. what does that mean for the sequential writing of the words?

        x = self.cnn_p2(x)  # [batch_size, channels, height, width]
        # print(x.size())  # torch.Size([-1, 256, 4, 10]) # torch.Size([32, 256, 4, 29])

        x = x.permute(0, 3, 1, 2)  # [batch_size, width, channels, height]
        # print(x.size())  # torch.Size([-1, 10, 256, 4]) # torch.Size([32, 29, 256, 4])

        batch_size = x.size(0)
        T = x.size(1)
        x = x.view(batch_size, T, -1)  # [batch_size, T==width, num_features==channels*height]
        # print(x.size())  # torch.Size([-1, 10, 1024]) # torch.Size([32, 29, 1024])

        x = self.linear1(x)
        # # print(x.size())  # torch.Size([-1, 10, 256]) # torch.Size([32, 29, 256])

        x, hidden = self.rnn1(x)
        feature_size = x.size(2)
        x = x[:, :, : feature_size // 2] + x[:, :, feature_size // 2 :]
        # print(x.size())  # torch.Size([-1, 10, 256]) # torch.Size([32, 29, 256])

        x, hidden = self.rnn2(x)
        # print(x.size())  # torch.Size([-1, 10, 512]) # torch.Size([32, 29, 512])

        x = self.linear2(x)
        # print(x.size())  # torch.Size([-1, 10, 20]) # torch.Size([32, 29, 95])

        x = x.permute(1, 0, 2)  # [T==10, batch_size, num_classes==num_features]
        # print(x.size())  # torch.Size([10, -1, 20]) # torch.Size([29, 32, 95])

        return x

    def weights_init(self, module: nn.Module):
        if mode_UsePretrainedWeight:
            moduleName_Exclude = self.dict_module_ExcludeWeightInit.get(module)
            if moduleName_Exclude is not None:
                # print(f"{moduleName_Exclude:<20} {module._get_name():<20} << This is in the pretrained Resnet:")
                return
            else:
                # print(f"{'':<20} {module._get_name():<20}")
                pass

        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)


# def weights_init(m: nn.Linear | nn.Conv2d | nn.Conv1d | nn.BatchNorm2d | nn.BatchNorm1d | nn.Module):
#     classname = m.__class__.__name__
#     print(f"name: {m._get_name()}")
#     print(type(m))
#     print(classname)
#     if isinstance(m, nn.Sequential):
#         print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
#     for cls in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.BatchNorm2d, nn.BatchNorm1d):
#         print(isinstance(m, cls), end="; ")
#     print()
#     if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
#         torch.nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             m.bias.data.fill_(0.01)
#     elif classname.find("BatchNorm") != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)