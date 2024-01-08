# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class AdaptiveFusion(nn.Module):
    def __init__(self, input_dim=300):
        super(AdaptiveFusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.w1 = nn.Linear(input_dim, 1)
        self.w2 = nn.Linear(input_dim, 1)
    
    def forward(self, rowg_rep, colg_rep):
        alpha = self.sigmoid(self.w1(rowg_rep))
        beta = self.sigmoid(self.w2(colg_rep))
        alpha = alpha / (alpha + beta)
        beta = 1 - alpha
        return alpha * rowg_rep + beta * colg_rep

class DG_Interaction(nn.Module):
    def __init__(self, input_dim=300, output_dim=300, hidden_dim=300, activation=F.relu, final_layer=False):
        super().__init__()
        """
            DualG 子模块。
        """
        self.final_layer = final_layer

        self.row_aware = GraphConv(input_dim, hidden_dim, activation=activation)
        self.col_aware = GraphConv(input_dim, hidden_dim, activation=activation)

        self.row_graph_support = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        self.col_graph_support = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        self.graph_merge = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )


    def graph_interaction(self, rowg_rep, colg_rep):
        """
            进行图间数据信息交互。。。
        """
        rowg_rep = self.row_graph_support(rowg_rep)
        colg_rep = self.col_graph_support(colg_rep)

        return self.graph_merge(torch.cat([rowg_rep, colg_rep], dim=1))
    
    def forward(self, table_feat, row_graph, col_graph):

        rowg_rep = self.row_aware(row_graph, table_feat)
        
        colg_rep = self.col_aware(col_graph, table_feat)

        if self.final_layer:
            return rowg_rep, colg_rep
        
        # 图间信息交互。
        g_rep = self.graph_interaction(rowg_rep, colg_rep)

        return g_rep


class DualG(nn.Module):
    def __init__(self, input_dim=300, output_dim=300, hidden_dim=300):
        super().__init__()
        
        self.first_layer = DG_Interaction(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)

        self.final_layer = DG_Interaction(input_dim=300, output_dim=300, hidden_dim=300, final_layer=True)

        self.adafusion = AdaptiveFusion(output_dim)

    def forward(self, table_feat, row_graph, col_graph):

        table_feat = self.first_layer(table_feat, row_graph, col_graph)

        rowg_rep, colg_rep = self.final_layer(table_feat, row_graph, col_graph)

        g_rep = self.adafusion(rowg_rep, colg_rep)
    
        return g_rep
        

class DualGraph_RL(nn.Module):
    """
        Dual Graph Representation Learning.
    """
    def __init__(self, input_dim=300, output_dim=300, hidden_dim=300, num_heads=4, residual=True):
        super(DualGraph_RL, self).__init__()

        self.residual = residual

        layers = []

        for _ in range(num_heads):
            dagcn = DualG(input_dim, output_dim, hidden_dim)
            layers.append(dagcn)

        self.layers = nn.ModuleList(layers)


    def forward(self, table_feat, row_graph, col_graph):
        res_list = []

        if self.residual:
            res_list = [table_feat]

        for layer in self.layers:
            g_rep = layer(table_feat, row_graph, col_graph)
            res_list.append(g_rep)
        
        all_rep = torch.stack(res_list, dim=0)
        g_rep = torch.mean(all_rep, dim=0)
        g_rep = torch.squeeze(g_rep)
        return g_rep
