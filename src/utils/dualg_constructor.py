import dgl
import numpy as np
import networkx as nx
import fasttext
import unicodedata
import re

SPACE_NORMALIZER = re.compile(r"\s+")


class DualGraph:
    def __init__(self, w2v_path=None, dataset="wikitables"):
        """
            初始化,w2v_path指定fasttext的预训练word2vec向量。
        """
        self.w2v = fasttext.load_model(w2v_path)
        self.dataset = dataset

    def construct_graph(self, table):
        """
            通过表格中的table_array字段中的数据进行构造图。
            node_features 是一行编码后的数据表示。
            graph 是编号好了的图1.2.3.4.。
        """
        table_data = table['table_array']
         # m行n列
        m = len(table_data)
        n = len(table_data[0])
        pt = ""
        st = ""
        caption = ""
        if self.dataset == "wikitables":
            pt = table['pgTitle']
            st = table['secondTitle']
            caption = table['caption']
            table_data.append([pt, st, caption])
        else:
            caption = table['caption']
            st = table['subcaption']
            table_data.append([caption, st])

        # table_data 转换为 node_data 表格最后的三个节点分别为 pt、st、capt
       
        node_features = self._node_embs(table_data)

        row_graph = self._build_graph(table_data, m, n, g_type="row")
        col_graph = self._build_graph(table_data, m, n, g_type="col")
        

        row_graph = dgl.from_networkx(row_graph)
        col_graph = dgl.from_networkx(col_graph)

        if row_graph.num_nodes() != node_features.shape[0]:
            print(f"error: graph.num_nodes() != node_features.shape[0], "
                  f"{row_graph.num_nodes()} != {node_features.shape[0]}")

        return row_graph, col_graph, node_features, table_data

    def _node_embs(self, tarr):
        """
            节点编码
        """
        # 数据清理
        tarr = [[self._normalize_text(c) for c in row] for row in tarr]

        features_textual = []
        for i, row in enumerate(tarr):
            for j, c in enumerate(row):
                features_textual.append(self._fasttext_sentence_emb(c))

        features = np.array(features_textual)
        return features

    def _fasttext_sentence_emb(self, text):
        if isinstance(text, list):
            text = " ".join(text)
        res = self.w2v[text]
        return res

    @staticmethod
    def _normalize_text(x):
        """
            规格化文本,因为其中有很多unicode字符以及特殊符号。
        """
        x = unicodedata.normalize('NFD', x).encode('ascii', 'ignore').decode("utf-8").strip()
        x = x.replace("'", " ")
        x = x.replace('"', " ")
        x = x.replace('|', " ")
        x = x.replace('#', " ")
        x = x.replace('/', " ")
        x = x.replace('\\', " ")
        x = x.replace('(', " ").replace(')', " ")
        x = x.replace('[', " ").replace(']', " ")
        x = SPACE_NORMALIZER.sub(" ", x)
        return x

    def _build_graph(self, tarr, m, n, g_type="row"):
        graph = nx.DiGraph()
        if self.dataset == "wikitables":
            graph.add_nodes_from(np.arange(m * n + 3))
        else:
            graph.add_nodes_from(np.arange(m * n + 2))
        edges = set()

        # merge cells
        cell_node_id = [[i * n + j for j in range(n)] for i in range(m)]
        if self.dataset == "wikitables":
            cell_node_id.append([m * n, m * n + 1, m * n + 2])
        else:
            cell_node_id.append([m * n, m * n + 1])
        
        for j in range(n):
            last_text = self._normalize_text(tarr[0][j])
            for i in range(1, m):
                curr_text = self._normalize_text(tarr[i][j])
                if last_text == curr_text:
                    cell_node_id[i][j] = cell_node_id[i-1][j]
                last_text = curr_text
        
        cell_node_id = [idx for row in cell_node_id for idx in row]
        # edges between cell nodes
        for i in range(m * n):
            for j in self._get_neighbors(i, m, n, g_type=g_type):
                idxi = cell_node_id[i]
                idxj = cell_node_id[j]
                edges.add((idxi, idxj))
                edges.add((idxj, idxi))
            # add self loop
            edges.add((i, i))
        if self.dataset == "wikitables":
            edges.add((m * n, m * n + 1))
            edges.add((m * n, m * n + 2))
            edges.add((m * n + 1, m * n + 2))
        else:
            edges.add((m * n, m * n + 1))
        graph.add_edges_from(edges)
        return graph

    def _get_neighbors(self, ind, m, n, g_type="row"):
        i = ind // n
        j = ind % n
        all = m * n
        res = []
        if self.dataset == "wikitables":
            res = [all, all+1, all+2]
        else:
            res = [all, all+1]

        # row type neighbor
        if g_type == "row":
            res.append(i)

        # col type neighbor
        if g_type == "col":
            res.append(j)

        return res
