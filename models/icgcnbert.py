from transformers import BertPreTrainedModel,BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ICGCNBert(BertPreTrainedModel):
    def __init__(self,config):
        super(ICGCNBert, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.gcn1 = GraphConvolution(config.hidden_size, config.hidden_size)
        self.gcn2 = GraphConvolution(config.hidden_size, config.hidden_size)
        self.gcn3 = GraphConvolution(config.hidden_size, config.hidden_size)
        self.gcn4 = GraphConvolution(config.hidden_size, config.hidden_size)
        self.gcn5 = GraphConvolution(config.hidden_size, config.hidden_size)
        self.gcn6 = GraphConvolution(config.hidden_size, config.hidden_size)

        #self.gc3 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc4 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc5 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc6 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc7 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc8 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.fc = nn.Linear(config.hidden_size, config.polarities_size)
        # self.dfc = nn.Linear(4*config.hidden_size, config.polarities_size)

        self.text_embed_dropout = nn.Dropout(0.3)
        self.init_weights()

    def forward(self, inputs):
        text_indices, attention_mask, in_adj, cross_adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        # target_len = torch.sum(target_indices != 0, dim=-1)
        # text = self.embed(text_indices)
        outputs = self.bert(input_ids=text_indices, attention_mask=attention_mask)
        text = outputs[0]
        text_out = self.text_embed_dropout(text)
        # text_out, (_, _) = self.text_lstm(text, text_len)
        # x = text_out
        # for i in range(0,2*self.config.gcn_layers,2):
        #     x = F.relu(self.gcn_list[i](x, in_adj))
        #     x = F.relu(self.gcn_list[i+1](x, cross_adj))
        x = F.relu(self.gcn1(text_out, in_adj))
        x = F.relu(self.gcn2(x, cross_adj))
        x = F.relu(self.gcn3(x, in_adj))
        x = F.relu(self.gcn4(x, cross_adj))
        x = F.relu(self.gcn5(x, in_adj))
        x = F.relu(self.gcn6(x, cross_adj))
        #x = F.relu(self.gcn7(x, in_adj))
        #x = F.relu(self.gcn8(x, cross_adj))
        #x = F.relu(self.cross_gcn3(x, cross_adj))
        #x_c = F.relu(self.cross_gcn4(x_c, in_adj))
        #x = F.relu(self.cross_gcn5(x, in_adj))
        #x = F.relu(self.cross_gcn6(x, cross_adj))

        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim

        #alpha_mat = torch.matmul(x_c, text_out.transpose(1, 2))
        #alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        #x_c = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim

        output = self.fc(x)
        return output
