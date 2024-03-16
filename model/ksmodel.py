import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, AutoModel, RobertaModel
import torch
from torch import nn
import torch.nn.functional as F
# from test_ks import cpuInfo
import logging

logger = logging.getLogger('general')


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # Xavier均匀分布初始化
        # xavier初始化方法中服从均匀分布U(−a,a) ，分布的参数a = gain * sqrt(6/fan_in+fan_out)，这里有一个gain，增益的大小是依据激活函数类型来设定

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # print('h', h[0], torch.isnan(h[0]).any())
        Wh = torch.matmul(h, self.W)  # h.shape: (bs, N, in_features), Wh.shape: (bs, N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(
            torch.matmul(a_input, self.a).squeeze(3))  # torch.matmul矩阵乘法，输入可以是高维
        # eij = a([Whi||Whj]),j属于Ni

        zero_vec = -9e15 * torch.ones_like(e)  # 范围维度和e一样的全1的矩阵
        attention = torch.where(adj > 0, e, zero_vec)
        # print('att1', attention[0], torch.isnan(attention[0]).any()) # 已经有nan
        attention = F.softmax(attention, dim=1)
        # print('att2', attention[0], torch.isnan(attention[0]).any())
        attention = F.dropout(attention, self.dropout, training=self.training)
        # print('att3', attention[0], torch.isnan(attention[0]).any())
        h_prime = torch.matmul(attention, Wh)
        # print(h_prime[0])
        if self.concat:
            return F.elu(h_prime, inplace=True)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        bs, N = Wh.size()[0], Wh.size()[1]  # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)  # [bs, node_len * node_len, dim], [bs, [a1, a1, a1, ...,] dim]
        Wh_repeated_alternating = Wh.repeat(1, N, 1)   # [bs, node_len * node_len, dim], [bs, [a1, a2, a1, a2, ...], dim]
        # repeat_interleave()：在原有的tensor上，按每一个tensor复制。
        # repeat()：根据原有的tensor复制n个，然后拼接在一起。
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)

        # dim=0表示按行拼接，1表示按列拼接
        # all_combinations_matrix.shape == (N * N, 2 * out_features)
        return all_combinations_matrix.view(-1, N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, out_dim, dropout=0.5, alpha=0.2, nheads=6):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            # Module.add_module(name: str, module: Module)。功能为，为Module添加一个子module，对应名字为name。

        self.out_att = GraphAttentionLayer(nhid * nheads, out_dim, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj), inplace=True)
        return F.log_softmax(x, dim=1)


class KsModel(nn.Module):
    def __init__(self, args):
        super(KsModel, self).__init__()
        if args.version == 'roberta':
            self.bert = RobertaModel.from_pretrained(args.bert_path)
        else:
            self.bert = BertModel.from_pretrained(args.bert_path)
        self.bert.resize_token_embeddings(args.ks_vocab_len)
        self.graph_encoder = nn.ModuleList([
            GAT(768, args.hidden_dim, args.hidden_dim * args.nheads, nheads=args.nheads, ),
            # GAT(args.hidden_dim * args.nheads, args.hidden_dim, args.hidden_dim * args.nheads, nheads=args.nheads)
        ])
        # self.hidden_size = 768
        self.hidden_size = args.hidden_dim * args.nheads
        self.know_layer = nn.Linear(self.hidden_size, 1)
        self.word_node_layer = nn.Linear(self.hidden_size, 2)
        self.project = nn.Sequential(
            # nn.LeakyReLU(0.2),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Linear(768, self.hidden_size)
        )
        # self.project = nn.Sequential(nn.Linear(768, self.hidden_size),
        #                         #    nn.LayerNorm(self.hidden_size),
        #                            nn.ReLU())
        # self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=1e-6)

        # self.project2 = nn.Sequential(
        #     # nn.LeakyReLU(0.2),
        #     nn.Dropout(),
        #     # nn.ReLU(),
        #     nn.Linear(self.hidden_size, self.hidden_size)
        # )
        # self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=1e-6)

        self.device = args.device
        self.word = args.add_word_loss

        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False

    def _get_embedding(self, sematic_unit):
        with torch.no_grad():
            input_ids = sematic_unit[0].to(self.device)
            attention_mask = sematic_unit[1].to(self.device)
            output = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
            embedding = (output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(
                    -1)
        return embedding

    def _get_embedding_with_grad(self, sematic_unit):
        input_ids = sematic_unit[0].to(self.device)
        attention_mask = sematic_unit[1].to(self.device)
        output = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        embedding = (output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(
                -1)
        return embedding
    
    # def forward(self, input_ids, attention_mask, token_type_ids, graph, ent_node_embeds, know_ids, node_mask):
    #     bs, doc_num, seq_len = input_ids.size()
    #     bs, node_len, _ = graph.size()
    #     is_know_node = know_ids > 0

    #     # init node embedding
    #     ent_node_embed = []     # [bs, doc, sematic_unit, dim]
    #     ent_node_num = []
    #     for i in range(bs):
    #         ent = None
    #         num = []

    #         for j in range(len(ent_node_embeds[i])):
    #             e1 = torch.tensor([])
    #             if ent_node_embeds[i][j][0].shape[0] > 1:
    #                 e1 = self._get_embedding([ent_node_embeds[i][j][0][:-1, :512], ent_node_embeds[i][j][1][:-1, :512]])
    #                 if ent is None:
    #                     ent = e1
    #                 else:
    #                     ent = torch.cat([ent, e1], dim=0)

    #             e2 = self._get_embedding_with_grad([ent_node_embeds[i][j][0][-1, :512].unsqueeze(0), ent_node_embeds[i][j][1][-1, :512].unsqueeze(0)])
    #             if ent is None:
    #                 ent = e2
    #             else:
    #                 ent = torch.cat([ent, e2], dim=0)
    #             num.append(e1.shape[0] + e2.shape[0])
    #         ent_node_num.append(num)
    #         ent_node_embed.append(ent)
    #     ent_node_embed = pad_sequence(ent_node_embed, batch_first=True)
    #     # print(input_ids[0])
    #     input_ids = input_ids.view(-1, seq_len)
    #     attention_mask = attention_mask.view(-1, seq_len)
    #     token_type_ids = token_type_ids.view(-1, seq_len)

    #     # sentence embedding
    #     outputs = self.bert(input_ids, attention_mask, token_type_ids, return_dict=True)
    #     outputs = outputs.last_hidden_state
    #     outputs = outputs.view(bs, doc_num, seq_len, -1)

    #     # get graph representations
    #     graph_node = []
    #     for i in range(bs):
    #         sent_node_embed = [outputs[i, j, is_know_node[i][j]] for j in range(doc_num)]
    #         g_node = [torch.cat([ent_node_embed[i, sum(ent_node_num[i][:j]): sum(ent_node_num[i][:j + 1]), :], sent_node_embed[j]]) for j in range(len(ent_node_num[i]))]
    #         for j in range(len(ent_node_num[i]), len(sent_node_embed)):
    #             g_node.append(sent_node_embed[j])
    #         g_node = torch.cat(g_node, dim=0)
    #         graph_node.append(g_node)

    #     graph_node = pad_sequence(graph_node, batch_first=True)
    #     assert graph_node.size()[1] == node_len

    #     # GAT
    #     hidden_states = self.graph_encoder[0](graph_node, graph)  # [bs, node_len, hidden_size]
    #     # hidden_states = self.graph_encoder[1](hidden_states, graph)  # [bs, node_len, hidden_size]

    #     # Residual
    #     graph_node_project = self.project(graph_node)
    #     hidden_states = graph_node_project + hidden_states

    #     # sentence logistics
    #     # hidden_states = graph_node
    #     # hidden_states = self.relu(hidden_states)
    #     know_logitis = self.know_layer(hidden_states).squeeze(-1)
    #     mask_node_vec = float("-inf") * torch.ones_like(know_logitis)
    #     know_logitis = torch.where(node_mask < 1, mask_node_vec, know_logitis)

    #     if not self.word:
    #         return know_logitis, None

    #     # word logistics
    #     word_logitis = self.word_node_layer(hidden_states).squeeze(-1)
    #     mask_node_vec = float("-inf") * torch.ones_like(word_logitis)
    #     word_mask = node_mask.unsqueeze(-1).repeat(1, 1, 2)
    #     word_logitis = torch.where(word_mask < 1, word_logitis, mask_node_vec)

    #     return know_logitis, word_logitis
    # @profile(precision=4, stream=open('model.log', 'w+'))
    def forward(self, input_ids, attention_mask, token_type_ids, graph, ent_node_embeds, know_ids, node_mask):
        bs, doc_num, seq_len = input_ids.size()
        bs, node_len, _ = graph.size()
        is_know_node = know_ids > 0

        # get sentence node and entity node
        sent_num, ent_num = [], []
        for i in range(bs):
            # sent_num.append(sum(node_mask[i]) - 1)
            # ent_num.append(list(node_mask[i]).index(1) + 1)
            stmp, etmp = -1, 0
            s, e = [], []
            for j in range(len(node_mask[i])):
                if node_mask[i][j] == 1:
                    if etmp != 0 or e == []:
                        e.append(etmp + 1)
                        etmp = 0
                    stmp += 1
                else:
                    if stmp != -1:
                        s.append(stmp)
                        stmp = -1
                    etmp += 1

            if stmp != -1:
                s.append(stmp)
            if etmp != 0:
                e.append(etmp)
            if e == []:
                e.append(1)
            s = s + [0] * (len(e) - len(s))
            e = e + [0] * (len(s) - len(e))
            sent_num.append(s)
            ent_num.append(e)
        # init node embedding
        ent_node_embed = []     # [bs, doc, sematic_unit, dim]
        for i in range(bs):
            ent = None
            for j in range(len(ent_node_embeds[i])):
                e1 = torch.tensor([])
                if ent_node_embeds[i][j][0].shape[0] > 1:
                    e1 = self._get_embedding([ent_node_embeds[i][j][0][:-1, :512], ent_node_embeds[i][j][1][:-1, :512]])
                    if ent is None:
                        ent = e1
                    else:
                        ent = torch.cat([ent, e1], dim=0)
                e2 = self._get_embedding_with_grad([ent_node_embeds[i][j][0][-1, :512].unsqueeze(0), ent_node_embeds[i][j][1][-1, :512].unsqueeze(0)])
                if ent is None:
                    ent = e2
                else:
                    ent = torch.cat([ent, e2], dim=0)
            ent_node_embed.append(ent)
            # if ent.shape[0] != sum(ent_num[i]):
            #     ent_num[i][-1] += ent.shape[0] - sum(ent_num[i])
        ent_node_embed = pad_sequence(ent_node_embed, batch_first=True)

        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        token_type_ids = token_type_ids.view(-1, seq_len)

        # sentence embedding
        # outputs = self.bert(input_ids, attention_mask, token_type_ids, return_dict=True)
        outputs = self.bert(input_ids, attention_mask)
        outputs = outputs.last_hidden_state
        outputs = outputs.view(bs, doc_num, seq_len, -1)
        # get graph representations
        graph_node = []
        for i in range(bs):
            sent_node_embed = [outputs[i, j, is_know_node[i][j]] for j in range(doc_num)]
            sent_node_embed = torch.cat(sent_node_embed, dim=0)
            g_node = []
            for j in range(len(ent_num[i])):
                g_node.append(ent_node_embed[i, sum(ent_num[i][:j]): sum(ent_num[i][:j + 1]), :])
                g_node.append(sent_node_embed[sum(sent_num[i][:j]): sum(sent_num[i][: j + 1]), :])
            g_node = torch.cat(g_node, dim=0)
            graph_node.append(g_node)

        graph_node = pad_sequence(graph_node, batch_first=True)
        # sent = 0
        # print(know_ids.shape)
        # for i in range(len(know_ids[0])):
        #     for j in range(len(know_ids[0][0])):
        #         if know_ids[0][i][j] == 1:
        #             sent += 1
        # print(sent)
        # print(graph_node.shape, ent_node_embed.shape, outputs.shape, node_len, is_know_node.shape, node_mask.shape, node_mask[0].sum())
        # print(node_len, graph_node.shape, sent_num, ent_num, node_mask)
        assert graph_node.size()[1] == node_len

        # GAT
        hidden_states = self.graph_encoder[0](graph_node, graph)  # [bs, node_len, hidden_size]
        # # Residual
        # graph_node_project = self.project(graph_node)
        # hidden_states = graph_node_project + hidden_states
        # hidden_states = self.layer_norm1(hidden_states)

        # # GAT
        # hidden_states = self.graph_encoder[1](hidden_states, graph)  # [bs, node_len, hidden_size]
        # # Residual
        graph_node_project = self.project(graph_node)
        hidden_states = graph_node_project + hidden_states
        # hidden_states = self.layer_norm2(hidden_states)

        # sentence logistics
        # hidden_states = graph_node
        # hidden_states = self.relu(hidden_states)
        know_logitis = self.know_layer(hidden_states).squeeze(-1)
        mask_node_vec = float("-inf") * torch.ones_like(know_logitis)
        know_logitis = torch.where(node_mask < 1, mask_node_vec, know_logitis)

        if not self.word:
            return know_logitis, None

        # word logistics
        word_logitis = self.word_node_layer(hidden_states).squeeze(-1)
        mask_node_vec = float("-inf") * torch.ones_like(word_logitis)
        word_mask = node_mask.unsqueeze(-1).repeat(1, 1, 2)
        word_logitis = torch.where(word_mask < 1, word_logitis, mask_node_vec)

        return know_logitis, word_logitis
