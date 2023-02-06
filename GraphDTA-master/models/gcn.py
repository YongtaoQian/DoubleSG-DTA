import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=25, output_dim=128,
                 dropout=0.2):
        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)  # x
        self.conv2 = GCNConv(num_features_xd * 2, num_features_xd * 1)  # x+x1
        self.conv3 = GCNConv(num_features_xd * 3, num_features_xd * 2)  # x+x1+x2
        self.conv4 = GCNConv(num_features_xd * 5, num_features_xd * 4)  # x+x1+x2+x3
        self.conv5 = GCNConv(num_features_xd * 9, num_features_xd * 8)  # x+x1+x2+x3+x4
        self.conv6 = GCNConv(num_features_xd * 17, num_features_xd * 16)  # x+x1+x2+x3+x4+x5
        self.conv7 = GCNConv(num_features_xd * 33, num_features_xd * 32)
        self.conv8 = GCNConv(num_features_xd * 65, num_features_xd * 32)
        self.conv9 = GCNConv(num_features_xd * 97, num_features_xd * 32)

        self.fc_g1 = torch.nn.Linear(num_features_xd * 32, num_features_xd * 128)
        self.fc_g2 = torch.nn.Linear(num_features_xd * 128, num_features_xd * 32)
        self.fc_g3 = torch.nn.Linear(num_features_xd * 32, num_features_xd * 4)
        self.fc_g4 = torch.nn.Linear(num_features_xd * 4, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.rnn=nn.LSTM(in_channels=1000, out_channels=n_filters,num_layers=3,bidirectional=True,dropout=dropout)

        #self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        # self.conv_xt_2 = nn.Conv1d(in_channels=n_filters * 8, out_channels=n_filters * 4, kernel_size=8)
        # self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 4, out_channels=n_filters * 2, kernel_size=8)
        # self.conv_xt_4 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 1, kernel_size=8)
        self.fc1_xt = nn.Linear(32 * 121, output_dim)


        # 1D convolution on smile sequence
        self.embedding_xt_smile = nn.Embedding(100, embed_dim)
        #         self.fnn1 = nn.Linear(embed_dim, embed_dim)
        self.conv_xt2 = nn.Conv1d(in_channels=100, out_channels=n_filters, kernel_size=8)
        self.fc_xt2 = nn.Linear(32 * 121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(384, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        target = data.target

        x1 = self.relu(self.conv1(x, edge_index))
        x2 = self.relu(self.conv2(torch.cat((x, x1), 1), edge_index))
        x3 = self.relu(self.conv3(torch.cat((x, x1, x2), 1), edge_index))
        x4 = self.relu(self.conv4(torch.cat((x, x1, x2, x3), 1), edge_index))
        x5 = self.relu(self.conv5(torch.cat((x, x1, x2, x3, x4), 1), edge_index))
        x6 = self.relu(self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1), edge_index))
        x7 = self.relu(self.conv7(torch.cat((x, x1, x2, x3, x4, x5, x6), 1), edge_index))
        x8 = self.relu(self.conv8(torch.cat((x, x1, x2, x3, x4, x5, x6, x7), 1), edge_index))
        x9 = self.relu(self.conv9(torch.cat((x, x1, x2, x3, x4, x5, x6, x7, x8), 1), edge_index))
        x = gmp(x9, batch)  # global max pooling

        # flatten
        x1 = self.dropout(self.relu(self.fc_g1(x)))
        x2 = self.dropout(self.relu(self.fc_g2(x1)))
        x3 = self.dropout(self.relu(self.fc_g3(x2)))
        x = self.fc_g4(x3)
        x = self.dropout(x)

        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        output,(hidden,cell)=self.rnn(embedded_xt)
        conv_xt1=torch.cat(hidden[-3],hidden[-2],hidden[-1],dim=1)
        #conv_xt1 = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt1.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        drug_smiles = data.drug_smiles
        embedded_xt1 = self.embedding_xt_smile(drug_smiles)
        # 跳连，将嵌入层加入到FNN中
        #         embedded_xt_fnn = self.fnn1(embedded_xt)
        #         embedded_xt = embedded_xt + embedded_xt_fnn
        conv_xt2 = self.conv_xt2(embedded_xt1)

        # flatten
        xd = conv_xt2.view(-1, 32 * 121)
        xd = self.fc_xt2(xd)





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
