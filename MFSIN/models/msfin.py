import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone_res12 import ResNet
from models.msfn import MSFN

class MSFIN(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args
        self.n_way = args.way
        self.n_query = args.query
        self.n_support = args.shot
        self.encoder = ResNet()
        self.encoder_dim = 640
        self.kernel_size1 = (5, 5)
        self.kernel_size2 = (3, 3)
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)
        self.msfn1 = MSFN(320, args.num_token)
        self.msfn2 = MSFN(640, args.num_token)
        self.unfold1 = nn.Unfold(self.kernel_size1,dilation=1, stride=2,padding=1)
        self.unfold2 = nn.Unfold(self.kernel_size2,dilation=1, stride=2,padding=1)

        self.conv_1 = nn.Sequential(
            nn.Conv1d(320, 640, kernel_size=1, bias=False),
            nn.BatchNorm1d(640),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv1d(self.encoder_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.i= 0


    def forward(self, input):

        if self.mode == 'fc':
            return self.fc_forward(input)

        elif self.mode == 'encoder':
            x1,x2 = self.encoder(input)
            return x1,x2

        elif self.mode == 'local_feat1':
            b, c, h, w = input.shape

            x = self.unfold1(input)
            n = x.shape[-1]
            x = x.contiguous().view(b,c,self.kernel_size1[0],self.kernel_size1[1],n)
            x = x.permute(0, 4, 1, 2, 3)
            x = x.contiguous().view(b*n, c, self.kernel_size1[0], self.kernel_size1[1])
            return x

        elif self.mode == 'local_feat2':
            b, c, h, w = input.shape
            x = self.unfold2(input)
            n = x.shape[-1]
            x = x.contiguous().view(b,c,self.kernel_size2[0],self.kernel_size2[1],n)
            x = x.permute(0, 4, 1, 2, 3)
            x = x.contiguous().view(b*n, c, self.kernel_size2[0], self.kernel_size2[1])
            return x


        elif self.mode == 'msfn1':
            x = self.msfn1(input)
            x = self.conv_1(x)
            return x

        elif self.mode == 'msfn2':
            x = self.msfn2(input)
            return x

        elif self.mode == 'mfin':
            spt, qry = input
            return self.metric(spt,qry)


    def fc_forward(self, x):
        x = x.mean(dim=[-1,-2])
        return self.fc(x)


    def metric(self, token_support, token_query):

        token_spt = self.normalize_feature(token_support)
        token_qry = self.normalize_feature(token_query)
        qry = self.normalize_feature(token_query)

        corr4d = self.mfin(token_spt, token_qry)  #q s n1 n2
        corr4d_s = self.gaussian_normalize(corr4d, dim=2)
        corr4d_q = self.gaussian_normalize(corr4d, dim=3)

        corr4d_s = F.softmax(corr4d_s / self.args.temperature_attn, dim=2) # q s n1 n2
        corr4d_q = F.softmax(corr4d_q / self.args.temperature_attn, dim=3) #q s n2 n2

        attn_s = corr4d_s.sum(dim=[3])
        attn_q = corr4d_q.sum(dim=[2])

        spt_attended = attn_s.unsqueeze(2) * token_spt.unsqueeze(0)
        qry_attended = attn_q.unsqueeze(2) * token_qry.unsqueeze(1)

        spt_attended_pooled = spt_attended.mean(dim=[-1])
        qry_attended_pooled = qry_attended.mean(dim=[-1])


        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)


        qry_pooled = qry.mean(dim=[-1])


        if self.training:
            return similarity_matrix/ self.args.temperature, self.fc(qry_pooled)
        else:
            return similarity_matrix/ self.args.temperature



    def mfin(self, spt, qry):

        way = spt.shape[0]
        num_qry = qry.shape[0]

        # reduce channel size via 1x1 conv
        spt = self.conv(spt)
        qry = self.conv(qry)

        # normalize channels for later cosine similarity
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)
        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1)
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1)
        similarity_map_einsum = torch.einsum('qnci,qnck->qnik', spt, qry)

        return similarity_map_einsum

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x
