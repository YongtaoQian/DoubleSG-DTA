import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import math

# drug sequence && preteion sequence---->Coss Multi-head Attention

# class mutil_head_attention1(nn.Module):
#     def __init__(self,head = 8,conv=32):
#         super(mutil_head_attention1,self).__init__()
#         self.conv = conv
#         self.head = head
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.d_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
#         self.p_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
#         self.scale = torch.sqrt(torch.FloatTensor([self.conv * 3])).cpu()

#     def forward(self, drug, protein):
#         bsz, d_ef,d_il = drug.shape
#         bsz, p_ef, p_il = protein.shape
#         drug_att = self.relu(self.d_a(drug.permute(0, 2, 1))).view(bsz,self.head,d_il,d_ef)
#         protein_att = self.relu(self.p_a(protein.permute(0, 2, 1))).view(bsz,self.head,p_il,p_ef)
#         interaction_map = torch.mean(self.tanh(torch.matmul(drug_att, protein_att.permute(0, 1, 3, 2)) / self.scale),1)
#         Compound_atte = self.tanh(torch.sum(interaction_map, 2)).unsqueeze(1)
#         Protein_atte = self.tanh(torch.sum(interaction_map, 1)).unsqueeze(1)
#         drug = drug * Compound_atte
#         protein = protein * Protein_atte
#         return drug,protein


# drug graph && preteion sequence---->Coss Multi-head Attention
# class mutil_head_attention2(nn.Module):
#     def __init__(self,head = 8,conv=32):
#         super(mutil_head_attention2,self).__init__()
#         self.conv = conv
#         self.head = head
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.d_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
#         self.p_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
#         self.scale = torch.sqrt(torch.FloatTensor([self.conv * 3])).cpu()

#     def forward(self, drug, protein):
#         bsz, d_ef,d_il = drug.shape
#         bsz, p_ef, p_il = protein.shape
#         drug_att = self.relu(self.d_a(drug.permute(0, 2, 1))).view(bsz,self.head,d_il,d_ef)
#         protein_att = self.relu(self.p_a(protein.permute(0, 2, 1))).view(bsz,self.head,p_il,p_ef)
#         interaction_map = torch.mean(self.tanh(torch.matmul(drug_att, protein_att.permute(0, 1, 3, 2)) / self.scale),1)
#         Compound_atte = self.tanh(torch.sum(interaction_map, 2)).unsqueeze(1)
#         Protein_atte = self.tanh(torch.sum(interaction_map, 1)).unsqueeze(1)
#         drug = drug * Compound_atte
#         protein = protein * Protein_atte
#         return drug,protein

# SE模块
class SE_Block(nn.Module):                         # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        
#         nn6 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#         self.conv6 = GINConv(nn6)
#         self.bn6 = torch.nn.BatchNorm1d(dim)
        
#         nn7 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#         self.conv7 = GINConv(nn6)
#         self.bn7 = torch.nn.BatchNorm1d(dim)
        
        self.fc1_xd = Linear(dim, output_dim)
        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
#         self.fnn = nn.Linear(embed_dim, embed_dim)
        self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.SE = SE_Block(n_filters)
        self.fc_xt1 = nn.Linear(32 * 121, output_dim)

        # 1D convolution on smile sequence
        self.embedding_xt_smile = nn.Embedding(100, embed_dim)
#         self.fnn1 = nn.Linear(embed_dim, embed_dim)
        self.conv_xt2 = nn.Conv1d(in_channels=100, out_channels=n_filters, kernel_size=8)
        self.SE1 = SE_Block(n_filters)
        self.fc_xt2 = nn.Linear(32 * 121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(384, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
#         x = F.relu(self.conv6(x, edge_index))
#         x = self.bn6(x)
#         x = F.relu(self.conv7(x, edge_index))
#         x = self.bn7(x)
        
        
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = global_max_pool(x, batch)


        target = data.target
        embedded_xt = self.embedding_xt(target)
        # 跳连，将嵌入层加入到FNN中
#         embedded_xt_fnn = self.fnn(embedded_xt)
#         embedded_xt = embedded_xt + embedded_xt_fnn
        conv_xt = self.conv_xt1(embedded_xt)
        conv_xt = self.relu(conv_xt)
        SE_proteinConv = self.SE(conv_xt)
        conv_xt = conv_xt * SE_proteinConv
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc_xt1(xt)
        x = global_max_pool(xt, batch)
        
        drug_smiles = data.drug_smiles
        embedded_xt1 = self.embedding_xt_smile(drug_smiles)
        # 跳连，将嵌入层加入到FNN中
#         embedded_xt_fnn = self.fnn1(embedded_xt)
#         embedded_xt = embedded_xt + embedded_xt_fnn
        conv_xt2 = self.conv_xt2(embedded_xt1)
        conv_xt2 = self.relu(conv_xt2)
        SE_drug = self.SE1(conv_xt2)
        conv_xt2 = conv_xt2 * SE_drug
        # flatten
        xd = conv_xt2.view(-1, 32 * 121)
        xd = self.fc_xt2(xd)
        x = global_max_pool(xd, batch)
        
#         x,xt1 = mutil_head_attention2(x, xt)
#         xd,xt2 = mutil_head_attention2(xd, xt)
#         xt = torch.cat(xt1,xt2)
        
        # concat
        xc = torch.cat((x, xt, xd), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
