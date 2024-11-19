# import torch
# import numpy as np
# import os
# from torch.nn import Parameter
# from torch.autograd import Variable
# from torch.nn import functional as F
# from SE_module import SELayer
# from ATT_module import ATTLayer
# from WTConv import *
# from mamba import *
# from Total import *
# class SelectE(torch.nn.Module):
#     def __init__(self, logger, num_emb, embedding_dim=300, input_drop=0.4, hidden_drop=0.3, feature_map_drop = 0.3,
#                 k_w = 10, k_h = 20, output_channel = 20, filter1_size = (1,5), filter2_size = (3,3), filter3_size = (1,9)):
#         super(SelectE, self).__init__()
#
#         current_file_name = os.path.basename(__file__)
#         logger.info( "[Model Name]: " + str(current_file_name))
#
#         # 定义模型
#         self.emb = torch.nn.Embedding(num_emb, embedding_dim)
#         self.logger = logger
#         self.se1 = SELayer(output_channel, reduction = int(0.5*output_channel))
#         self.se3 = SELayer(output_channel, reduction = int(0.5*output_channel))
#         self.se5 = SELayer(output_channel, reduction = int(0.5*output_channel))
#         self.att = ATTLayer(output_channel,reduction = int(0.5*output_channel))
#         self.embedding_dim = embedding_dim
#         self.perm = 1
#         self.k_w = k_w
#         self.k_h = k_h
#         self.loss = torch.nn.CrossEntropyLoss()
#         self.device = torch.device('cuda')
#         #self.net = NdMamba2(2400,200,64).cuda()
#         # 定义尺寸
#         self.chequer_perm = self.get_chequer_perm()
#         self.reshape_H = 20
#         self.reshape_W = 20
#         self.in_channel = 1 # 输入通道数
#         self.out_1 = output_channel # 第一个卷积核的输出通道数
#         self.out_2 = output_channel # 第二个卷积核的输出通道数
#         self.out_3 = output_channel # 第三个卷积核的输出通道数
#         self.emb = torch.nn.Embedding(num_emb, embedding_dim)
#         # 卷积核
#         self.filter1_size = filter1_size
#         self.filter2_size = filter2_size
#         self.filter3_size = filter3_size
#         self.h1 = self.filter1_size[0]
#         self.w1 = self.filter1_size[1]
#         self.h2 = self.filter2_size[0]
#         self.w2 = self.filter2_size[1]
#         self.h3 = self.filter3_size[0]
#         self.w3 = self.filter3_size[1]
#
#         filter1_dim = self.in_channel * self.out_1 * self.h1 * self.w1  # 1*20*1*3=60
#         self.filter1 = torch.nn.Embedding(num_emb, filter1_dim, padding_idx=0)
#         filter2_dim = self.in_channel * self.out_2 * self.h2 * self.w2  # 1*20*3*3=180
#         self.filter3 = torch.nn.Embedding(num_emb, filter2_dim, padding_idx=0)
#         filter3_dim = self.in_channel * self.out_3 * self.h3 * self.w3 # 1*20*1*5=100
#         self.filter5 = torch.nn.Embedding(num_emb, filter3_dim, padding_idx=0)
#
#         # 定义dropout和batchnorm
#         self.input_drop = torch.nn.Dropout(input_drop)
#         self.hidden_drop = torch.nn.Dropout(hidden_drop)
#         self.feature_map_drop = torch.nn.Dropout2d(feature_map_drop)
#         self.bn0 = torch.nn.BatchNorm2d(self.in_channel)
#         self.bn1 = torch.nn.BatchNorm2d(self.out_1 + self.out_2 + self.out_3)
#         self.bn1_1 = torch.nn.BatchNorm2d(self.out_1)
#         self.bn1_2 = torch.nn.BatchNorm2d(self.out_2)
#         self.bn1_3 = torch.nn.BatchNorm2d(self.out_3)
#         self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
#
#         fc_length = self.reshape_H * self.reshape_W * (self.out_1 + self.out_2 + self.out_3)
#         #fc_length = 13824
#         self.fc = torch.nn.Linear(fc_length, embedding_dim)
#         self.register_parameter('b', Parameter(torch.zeros(num_emb)))
#
#     def to_var(self, x, use_gpu=True):
#         if use_gpu:
#             return Variable(torch.from_numpy(x).long().cuda())
#
#     def init(self):
#         torch.nn.init.xavier_normal_(self.emb.weight.data)
#         torch.nn.init.xavier_normal_(self.filter1.weight.data)
#         torch.nn.init.xavier_normal_(self.filter3.weight.data)
#         torch.nn.init.xavier_normal_(self.filter5.weight.data)
#
#     def get_chequer_perm(self):
#         ent_perm = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)]) # 返回一个随机排列
#         rel_perm = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)])
#         comb_idx = []
#         for k in range(self.perm):
#             temp = []
#             ent_idx, rel_idx = 0, 0
#
#             for i in range(self.k_h):
#                 for j in range(self.k_w):
#                     if k % 2 == 0:
#                         if i % 2 == 0:
#                             temp.append(ent_perm[k, ent_idx])
#                             ent_idx += 1
#                             temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
#                             rel_idx += 1
#                         else:
#                             temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
#                             rel_idx += 1
#                             temp.append(ent_perm[k, ent_idx])
#                             ent_idx += 1
#                     else:
#                         if i % 2 == 0:
#                             temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
#                             rel_idx += 1
#                             temp.append(ent_perm[k, ent_idx])
#                             ent_idx += 1
#                         else:
#                             temp.append(ent_perm[k, ent_idx])
#                             ent_idx += 1
#                             temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
#                             rel_idx += 1
#
#             comb_idx.append(temp)
#
#         chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
#         return chequer_perm
#
#     def forward(self, e1, rel):
#         e1 = self.to_var(e1)
#         rel = self.to_var(rel) #将头实体和关系的numpy数组转换为pytorch张量，形状为(batch_size, )
#         e1_embedded = self.emb(e1)
#         rel_embedded = self.emb(rel) #转换为嵌入向量，embedding_dim=200,因此形状为(batch_size,200)
#         comb_emb = torch.cat([e1_embedded, rel_embedded], dim=1)  #在列维度上拼接，形状是(batch_size,400)
#         chequer_perm = comb_emb[:, self.chequer_perm] #棋盘重新排列，batch_size变成通道维度，200*1的e1_embedded和rel_embedded被转换成20个1*10的小向量，交替拼接成一个(10*20*2)的二维矩阵
#         stack_inp = chequer_perm.reshape((-1, self.perm, 2*self.k_w, self.k_h)) #（batch_size,1,20,20)
#         x = self.bn0(stack_inp)
#         x = self.input_drop(x)
#         #x = F.interpolate(x, size=(24, 24), mode='bilinear', align_corners=False)
#         #conv_layer = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=1, stride=1).to('cuda')
#         #x = conv_layer(x) #1500,20,20,20
#         #block = WTConv2d(1,1).to('cuda')
#         #x = block(x)
#         x = x.permute(1, 0, 2, 3) #(1,batch_size,20,20)
#         # H = HLFD(dim=24).to('cuda')
#         # x = H(x)
#
#         f1 = self.filter1(rel) # (1500,60)   将关系映射为60维，作为1*3卷积核的参数，相当于20个1*3卷积核       (1500,160)
#         f1 = f1.reshape(e1_embedded.size(0) * self.in_channel * self.out_1, 1, self.h1, self.w1) # (1500*1*20,1,1,3)  1500的batch_size,卷积核的通道数是1（每个样本单独卷积，输入通道是1），卷积核数量是20个，大小为1*3 (48000,4,1,5)
#         f3 = self.filter3(rel) # (1500,180)          (1500,720)
#         f3 = f3.reshape(e1_embedded.size(0) * self.in_channel * self.out_2, 1, self.h2, self.w2) # (1500*1*20,1,3,3)    (120000,4,3,3)
#         f5 = self.filter5(rel) # (1500,100)          (1500,288)
#         f5 = f5.reshape(e1_embedded.size(0) * self.in_channel * self.out_3, 1, self.h3, self.w3) # (1500*1*20,1,1,5)    (48000,4,1,9)
#         # (1,1500,20,20)  (48000,1,1,5)
#
#
#         x1 = F.conv2d(x, f1, groups=e1_embedded.size(0), padding=(int((self.h1 - 1)//2), int((self.w1 - 1)//2))) # 用1*3的卷积核进行分组卷积，卷积（20*20）的矩阵，输出通道为20    (1,30000,20,20)
#         x1 = x1.reshape(e1_embedded.size(0), self.out_1, self.reshape_H, self.reshape_W) # （1500，20，20，20）
#         x1 = self.bn1_1(x1)
#         x1 = self.se1(x1)
#
#
#         x3 = F.conv2d(x, f3, groups=e1_embedded.size(0), padding=(int((self.h2 - 1)//2), int((self.w2 - 1)//2)))
#         x3 = x3.reshape(e1_embedded.size(0), self.out_2, self.reshape_H, self.reshape_W)
#         x3 = self.bn1_2(x3)
#         x3 = self.se3(x3)
#
#
#         x5 = F.conv2d(x, f5, groups=e1_embedded.size(0), padding=(int((self.h3 - 1)//2), int((self.w3 - 1)//2)))
#         x5 = x5.reshape(e1_embedded.size(0), self.out_3, self.reshape_H, self.reshape_W)
#         x5 = self.bn1_3(x5)
#         x5 = self.se5(x5)
#
#
#         x = x1 + x3 + x5 # (128,20,20,20)    （1500，20，20，20）
#         y1,y3,y5 = self.att(x)  # 每个都是（1500，20，1，1）
#         y1 = y1.expand_as(x1)
#         y3 = y3.expand_as(x3)
#         y5 = y1.expand_as(x5)
#         x1 = x1 * y1
#         x3 = x3 * y3
#         x5 = x5 * y5
#
#         x1 = x1.reshape(e1_embedded.size(0),1,8000)
#         x3 = x3.reshape(e1_embedded.size(0), 1, 8000)
#         x5 = x5.reshape(e1_embedded.size(0), 1, 8000)
#
#         x = torch.cat([x1, x3, x5], dim=1)  #(1500,60,20,20)
#         x = torch.relu(x)
#         x = self.feature_map_drop(x)
#         #x = x.reshape(e1_embedded.size(0),3,8000)
#         #conv = nn.Conv2d(60, 6, 1).to('cuda')
#         #x = conv(x)
#         x = x.view(x.shape[0], -1) #(1500,24000)相当于平铺特征图
#
#
#         #x = x.unsqueeze(2)
#         #x = self.net(x)
#         x = self.fc(x) #（1500，200）
#         #x = x.squeeze(2)
#         x = self.hidden_drop(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         weight = self.emb.weight #(1500,41817)
#         weight = weight.transpose(1, 0) #(41817,1500)
#         x = torch.mm(x, weight) #（1500，41817）
#         x += self.b.expand_as(x) #加偏置b
#         pred = x
#         return pred
import torch
import numpy as np
import os
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn import functional as F
from SE_module import SELayer
from ATT_module import ATTLayer
from CBAM import *
from PConv import *
from PagFM import *
from WTDownSample import *
from CAFM import *
from FCA import *
from PagFMExtented import *
from Fusion import *
from LargeKernelAtt import *
from Wavelets import *
from MambaFusion import *
class SelectE(torch.nn.Module):
    def __init__(self, logger, num_emb, embedding_dim=300, input_drop=0.4, hidden_drop=0.3, feature_map_drop = 0.3,
                k_w = 10, k_h = 20, output_channel = 20, filter1_size = (1,5), filter2_size = (3,3), filter3_size = (1,9)):
        super(SelectE, self).__init__()

        current_file_name = os.path.basename(__file__)
        logger.info( "[Model Name]: " + str(current_file_name))

        # 定义模型
        self.wt1 = DWT_2D(wave='haar').to('cuda')
        self.idwt1 = IDWT_2D(wave='haar').to('cuda')
        self.wt3 = DWT_2D(wave='db1').to('cuda')
        self.idwt3 = IDWT_2D(wave='db1').to('cuda')
        self.wt5 = DWT_2D(wave='sym2').to('cuda')
        self.idwt5 = IDWT_2D(wave='sym2').to('cuda')
        self.emb = torch.nn.Embedding(num_emb, embedding_dim)
        self.logger = logger
        self.se1 = SELayer(4, reduction = int(0.5*output_channel))
        self.se3 = SELayer(4, reduction = int(0.5*output_channel))
        self.se5 = SELayer(4, reduction = int(0.5*output_channel))
        self.att = ATTLayer(output_channel,reduction = int(0.5*output_channel))
        self.embedding_dim = embedding_dim
        self.perm = 1
        self.k_w = k_w
        self.k_h = k_h
        self.loss = torch.nn.CrossEntropyLoss()
        self.device = torch.device('cuda')
        self.down_sample = Down_wt(20, 20).to(self.device)
        self.fusion = Fusion(20, wave='haar').to(self.device)
        # 定义尺寸
        self.chequer_perm = self.get_chequer_perm()
        self.reshape_H = 20
        self.reshape_W = 20
        self.in_channel = 1 # 输入通道数
        self.out_1 = output_channel # 第一个卷积核的输出通道数
        self.out_2 = output_channel # 第二个卷积核的输出通道数
        self.out_3 = output_channel # 第三个卷积核的输出通道数
        self.emb = torch.nn.Embedding(num_emb, embedding_dim)
        # 卷积核
        self.filter1_size = filter1_size
        self.filter2_size = filter2_size
        self.filter3_size = filter3_size
        self.h1 = self.filter1_size[0]
        self.w1 = self.filter1_size[1]
        self.h2 = self.filter2_size[0]
        self.w2 = self.filter2_size[1]
        self.h3 = self.filter3_size[0]
        self.w3 = self.filter3_size[1]

        filter1_dim = self.in_channel * self.out_1 * self.h1 * self.w1  # 1*8*1*5=40
        self.filter1 = torch.nn.Embedding(num_emb, filter1_dim, padding_idx=0) # 22,40
        filter2_dim = self.in_channel * self.out_2 * self.h2 * self.w2  # 1*20*3*3=180
        self.filter3 = torch.nn.Embedding(num_emb, filter2_dim, padding_idx=0) # 22,180
        filter3_dim = self.in_channel * self.out_3 * self.h3 * self.w3 # 1*8*1*9=72
        self.filter5 = torch.nn.Embedding(num_emb, filter3_dim, padding_idx=0) # 22,72

        # 定义dropout和batchnorm
        self.input_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feature_map_drop)
        self.bn0 = torch.nn.BatchNorm2d(self.in_channel)
        self.bn1 = torch.nn.BatchNorm2d(self.out_1 + self.out_2 + self.out_3)
        self.bn1_1 = torch.nn.BatchNorm2d(4)
        self.bn1_2 = torch.nn.BatchNorm2d(4)
        self.bn1_3 = torch.nn.BatchNorm2d(4)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.Mamba = Mamba_Enhancement_Fusion_Module(20, 20).to('cuda')

        #fc_length = self.reshape_H * self.reshape_W * (self.out_1 + self.out_2 + self.out_3)
        fc_length = 1*20*20
        self.fc = torch.nn.Linear(fc_length, embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_emb)))

    def to_var(self, x, use_gpu=True):
        if use_gpu:
            return Variable(torch.from_numpy(x).long().cuda())

    def init(self):
        torch.nn.init.xavier_normal_(self.emb.weight.data)
        torch.nn.init.xavier_normal_(self.filter1.weight.data)
        torch.nn.init.xavier_normal_(self.filter3.weight.data)
        torch.nn.init.xavier_normal_(self.filter5.weight.data)

    def get_chequer_perm(self):
        ent_perm = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)]) # 返回一个随机排列
        rel_perm = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)])
        comb_idx = []
        for k in range(self.perm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.k_h):
                for j in range(self.k_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm

    def forward(self, e1, rel):
        e1 = self.to_var(e1)
        rel = self.to_var(rel)

        e1_embedded = self.emb(e1)
        rel_embedded = self.emb(rel)
        comb_emb = torch.cat([e1_embedded, rel_embedded], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.perm, 2*self.k_w, self.k_h))
        x = self.bn0(stack_inp)
        x = self.input_drop(x)
       # x = x.permute(1, 0, 2, 3)

        f1 = self.filter1(rel) # (1500,160)

        f1 = f1.reshape(e1_embedded.size(0) * self.in_channel * self.out_1, 1, self.h1, self.w1) # (48000,4,1,5)
        f3 = self.filter3(rel) # (1500,720)
        f3 = f3.reshape(e1_embedded.size(0) * self.in_channel * self.out_2, 1, self.h2, self.w2) # (120000,4,3,3)
        f5 = self.filter5(rel) # (1500,288)
        f5 = f5.reshape(e1_embedded.size(0) * self.in_channel * self.out_3, 1, self.h3, self.w3) # (48000,4,1,9)
        # (1,1500,20,20)  (48000,1,1,5)


        # x1 = F.conv2d(x, f1, groups=e1_embedded.size(0), padding=(int((self.h1 - 1)//2), int((self.w1 - 1)//2))) # (4,48000,20,20)
        # x1 = x1.reshape(e1_embedded.size(0), self.out_1, self.reshape_H, self.reshape_W) # (128,128,20,20)
        # x1_down = self.down_sample(x1)
        # x1 = self.fusion(x1, x1_down)


        x1 = self.wt1(x) #1500 80 10 10
        print(x1.shape)
        x1 = self.bn1_1(x1)
        # CBAM_1 = CBAM(20).to('cuda')
        # x1 = CBAM_1(x1)
        #x1 = self.se1(x1)
        #
        #att_block = CAFM(dim=20).to('cuda')
        #x1 = att_block(x1)
        #FCAnet = FCAttention(channel=20).to('cuda')
        #x1 = FCAnet(x1)
        #LKA = MLKA(21).to('cuda')
        #x1 = LKA(x1)
        x1 = self.se1(x1)

        x1 = self.idwt1(x1) #1500 20 20 20
        # print(x1.shape)
        # x3 = F.conv2d(x, f3, groups=e1_embedded.size(0), padding=(int((self.h2 - 1)//2), int((self.w2 - 1)//2))) # (1,2560,20,20)
        # x3 = x3.reshape(e1_embedded.size(0), self.out_2, self.reshape_H, self.reshape_W)# (128,20,20,20)
        # x3_down = self.down_sample(x3)
        # x3 = self.fusion(x3, x3_down)

        x3 = self.wt3(x)
        #
        x3 = self.bn1_2(x3)
        # CBAM_3 = CBAM(20).to('cuda')
        # x3 = CBAM_3(x3)
        #x3 = att_block(x3)
        #x3 = FCAnet(x3)
        #x3 = LKA(x3)
        x3 = self.se3(x3)

        x3 = self.idwt3(x3)

        # x5 = F.conv2d(x, f5, groups=e1_embedded.size(0), padding=(int((self.h3 - 1)//2), int((self.w3 - 1)//2)))# (1,1024,20,20)
        # x5 = x5.reshape(e1_embedded.size(0), self.out_3, self.reshape_H, self.reshape_W) # (128,8,20,20)
        # x5_down = self.down_sample(x5)
        # x5 = self.fusion(x5, x5_down)

        x5 = self.wt5(x)
        x5 = self.bn1_3(x5)
        # CBAM_5 = CBAM(20).to('cuda')
        # x5 = CBAM_5(x5)

        #x5 = LKA(x5)
        x5 = self.se5(x5)

        x5 = self.idwt5(x5)

        # x = torch.cat([x1, x3, x5], dim=1)
        # block = PConv(dim=60, n_div=4).to('cuda')
        # x = block(x)
        # conv_layer = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=1).to('cuda')
        # x = conv_layer(x)
        #x = x1 + x3 + x5 # (128,20,20,20)
        # Fusion = MAFM(inc=21).to('cuda')
        # x = Fusion(x1, x3, x5)


        #x = self.Mamba(x1, x3, x5)
        PagFM_block = PagFMExtended(1,1).to('cuda')
        x = PagFM_block(x1, x3, x5)
        #x = PagFM_block(x, x5)
        # y1,y3,y5 = self.att(x)
        # y1 = y1.expand_as(x1)
        # y3 = y3.expand_as(x3)
        # y5 = y1.expand_as(x5)
        # x1 = x1 * y1
        # x3 = x3 * y3
        # x5 = x5 * y5
        #
        # x = torch.cat([x1, x3, x5], dim=1)
        # x = torch.relu(x)
        # x = self.feature_map_drop(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        weight = self.emb.weight
        weight = weight.transpose(1, 0)
        x = torch.mm(x, weight)
        x += self.b.expand_as(x)
        pred = x
        return pred
