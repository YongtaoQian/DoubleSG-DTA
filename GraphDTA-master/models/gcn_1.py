import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.2):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd*2, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.conv4 = GCNConv(num_features_xd * 4, num_features_xd * 8)
        self.conv5 = GCNConv(num_features_xd * 8, num_features_xd * 16)
        self.conv6 = GCNConv(num_features_xd * 16, num_features_xd * 32)
        self.fc_g1 = torch.nn.Linear(num_features_xd*32, num_features_xd*16)
        self.fc_g2 = torch.nn.Linear(num_features_xd * 16, num_features_xd * 4)
        self.fc_g3 = torch.nn.Linear(num_features_xd * 4, 1024)
        self.fc_g4 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        target = data.target

        x = self.conv1(x, edge_index)
        x1 = self.relu(x)

        x = self.conv2(torch.cat((x, x1), 1), edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)

        x = self.conv4(x, edge_index)
        x = self.relu(x)
        x = self.conv5(x, edge_index)
        x = self.relu(x)
        x = self.conv6(x, edge_index)
        x = self.relu(x)

        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.dropout(self.relu(self.fc_g1(x)))
        x = self.dropout(self.relu(self.fc_g2(x)))
        x = self.dropout(self.relu(self.fc_g3(x)))
        x = self.fc_g4(x)
        x = self.dropout(x)

        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out