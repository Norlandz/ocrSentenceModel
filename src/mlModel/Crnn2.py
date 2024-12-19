# Building a Captcha OCR in TF2.0
# https://www.kaggle.com/code/aakashnain/building-a-captcha-ocr-in-tf2-0
# 
# CAPTCHA : 2 solutions to break them [>99%]
# https://www.kaggle.com/code/arnrob/captcha-2-solutions-to-break-them-99#Model-Performance
# 
# Handwritten Text Recognition-IAM
# https://www.kaggle.com/code/hamiddamadi/handwritten-text-recognition-iam/notebook#Building-the-Handwritten-Text-Recognition-Model-(HTR)
# 
# class CRNN(nn.Module):
# 
#     def __init__(self, num_chars, rnn_hidden_size=256, dropout=0.2):
#         super().__init__()
#         self.num_chars = num_chars
#         self.rnn_hidden_size = rnn_hidden_size
#         self.dropout = dropout
# 
#         # CNN Part 1
#         self.cnn_p1 = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1),
#             # nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2, 2)),  # // dkdkdkdk padding
#         )
# 
#         # CNN Part 2
#         self.cnn_p2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
#             # nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2, 2)),
#         )
# 
#         # RNN
#         self.linear1 = nn.Linear(768, 64)  # 50 (height after pooling), 768 total width reduced to 64.
# 
#         # RNN layers
#         self.rnn1 = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True, dropout=0.25)
#         self.rnn2 = nn.LSTM(input_size=128*2, hidden_size=64, bidirectional=True, batch_first=True, dropout=0.25)
# 
#         self.linear2 = nn.Linear(64 * 2, num_chars)
# 
#     def forward(self, x: Tensor):
#         # print(x.size())  # @old vs @new # torch.Size([32, 3, 32, 256]) ###
#         x = self.cnn_p1(x)
#         # print(x.size())  # torch.Size([-1, 256, 4, 13]) ###
#         x = self.cnn_p2(x)  # [batch_size, channels, height, width]
#         # print(x.size())  # torch.Size([-1, 256, 4, 10]) ###
# 
#         # print('linear')
#         x = x.permute(0, 3, 1, 2)  # [batch_size, width, channels, height]
#         # print(x.size())  # torch.Size([-1, 10, 256, 4]) ###
#         batch_size = x.size(0)
#         T = x.size(1)
#         x = x.view(batch_size, T, -1)  # [batch_size, T==width, num_features==channels*height]
#         # print(x.size())  # torch.Size([-1, 10, 1024]) ###
# 
#         x = self.linear1(x)
#         x = F.relu(x)
#         x = F.dropout(x, self.dropout)
#         # print(x.size())  # torch.Size([-1, 10, 256]) ###
# 
#         # print('rnn')
#         x, hidden = self.rnn1(x)
#         feature_size = x.size(2)
#         # x = x[:, :, : feature_size // 2] + x[:, :, feature_size // 2 :]
#         # print(x.size())  # torch.Size([-1, 10, 256]) ###
# 
#         x, hidden = self.rnn2(x)
#         # print(x.size())  # torch.Size([-1, 10, 512]) ###
# 
#         x = self.linear2(x)
#         # print(x.size())  # torch.Size([-1, 10, 20]) ###
# 
#         x = x.permute(1, 0, 2)  # [T==10, batch_size, num_classes==num_features]
#         # print(x.size())  # torch.Size([10, -1, 20]) ###
# 
#         return x