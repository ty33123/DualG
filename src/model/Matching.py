import torch
import math
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertModel

from src.model.DualG import DualGraph_RL


class Attention(nn.Module):
    def __init__(self, q_in_dim=300, kv_in_dim=300, hid_dim=300):
        super().__init__()
        self.q_linear = nn.Linear(q_in_dim, hid_dim)
        self.k_linear = nn.Linear(kv_in_dim, hid_dim)
        self.softmax = nn.Softmax(1)

    def forward(self, q, k):
        n_q = self.q_linear(q)
        n_k = self.k_linear(k)
        qk = torch.mm(n_q, n_k.T)
        s = self.softmax(qk * 1. / math.sqrt(300))
        atten = torch.mm(s, n_k)
        return atten


class MatchingModel(nn.Module):
    def __init__(self, bert_dir='bert-base-uncased', do_lower_case=True, bert_size=768, gnn_output_size=300):
        super().__init__()
        # bert and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_dir, do_lower_case=do_lower_case)
        self.bert = BertModel.from_pretrained(bert_dir)

        self.project_table = nn.Sequential(
            nn.Linear(gnn_output_size, 300),
            nn.LayerNorm(300),
            nn.LeakyReLU(0.2)
        )

        # Dual Graph Representation Learning.
        self.dgrl = DualGraph_RL()

        self.qg_atten = Attention()

        self.dimr_bert = nn.Sequential(
            nn.Linear(bert_size, 300)
        )

        self.ss_project = nn.Sequential(
            nn.Linear(300, 300),
            nn.LeakyReLU(0.2)
        )

        self.regression = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(600, 12),
            nn.LeakyReLU(0.1),
            nn.Linear(12, 1),
        )

    def forward(self, table, query, row_graph, col_graph, t_feat, q_feat):
        """
            table retrieval
            表格与查询的相关度计算。
            t_feat->node_features
            q_feat->query_feature
            t_feat and q_feat 由fasttext编码得到
        """
        hqc = self.dimr_bert(self.query_context_matching(table, query))

        hqg = self.query_graph_matching(row_graph, col_graph, t_feat, q_feat)

        hqc = self.ss_project(hqc.unsqueeze(0))

        hqg = self.ss_project(hqg)

        score = self.regression(torch.cat([hqc, hqg], dim=-1))

        return score

    def query_graph_matching(self, row_graph, col_graph, t_feat, query_emb):
        """
            query-table matching module
            查询与表格的匹配。
        """
        # creps 为所有节点的表示，通过DualG得到。
        table_rep = self.dgrl(t_feat, row_graph, col_graph)

        table_rep = self.project_table(table_rep)

        query_emb = self.project_table(query_emb.unsqueeze(0))

        hqg = self.qg_atten(query_emb, table_rep)

        return hqg

    def query_context_matching(self, table, query):
        """
            查询与标题 等上下文信息进行匹配。
        """
        tokens = ["[CLS]"]
        tokens += self.tokenizer.tokenize(" ".join(query))[:64]
        tokens += ["[SEP]"]

        token_types = [0 for _ in range(len(tokens))]

        tokens += self.tokenizer.tokenize(table["caption"])[:20]
        tokens += ["[SEP]"]

        if 'subcaption' in table:
            tokens += self.tokenizer.tokenize(table["subcaption"])[:20]
            # tokens += ["[SEP]"]

        if 'pgTitle' in table:
            tokens += self.tokenizer.tokenize(table["pgTitle"])[:10]
            # tokens += ["[SEP]"]

        if 'secondTitle' in table:
            tokens += self.tokenizer.tokenize(table["secondTitle"])[:10]
            # tokens += ["[SEP]"]

        token_types += [1 for _ in range(len(tokens) - len(token_types))]

        # truncate and pad
        tokens = tokens[:128]
        token_types = token_types[:128]

        assert len(tokens) == len(token_types)

        token_indices = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([token_indices]).to("cuda")
        token_type_tensor = torch.tensor([token_types]).to("cuda")

        outputs = self.bert(tokens_tensor, token_type_ids=token_type_tensor)

        return outputs[1][0]  # pooled output of the [CLS] token
