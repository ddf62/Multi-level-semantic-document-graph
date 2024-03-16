import argparse
import copy
import json
import os.path
from copy import deepcopy

import numpy as np
import random
import sys
sys.path.append('.')
from model.tokenizer_llama import Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import spacy
import neuralcoref
from transformers import BertTokenizer, BertModel, BertTokenizerFast, BertConfig, GPT2TokenizerFast, LlamaTokenizerFast, RobertaModel, RobertaTokenizerFast, AutoTokenizer
import logging
from multiprocessing import Manager

logger = logging.getLogger('general')


class dataprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        neuralcoref.add_to_pipe(self.nlp)

    @staticmethod
    def remove_node(lists, graph, sematic_unit, dic, new_dic=None):
        if new_dic:
            conj = reversed(lists)
            new_dic.reverse()
        else:
            conj = list(set(lists))
            conj.sort(reverse=True)
        for idx, i in enumerate(conj):
            for j in range(i, len(graph) - 1):
                graph[j] = graph[j + 1]
                for k in range(len(graph)):
                    graph[k][j] = graph[k][j + 1]
                for kk in range(len(sematic_unit[j])):
                    dic[sematic_unit[j][kk].i] -= 1
            for kk in range(len(sematic_unit[-1])):
                dic[sematic_unit[-1][kk].i] -= 1
            for kk in range(len(sematic_unit[i])):
                dic[sematic_unit[i][kk].i] = None if not new_dic else -1 * sematic_unit[new_dic[idx]][0].i
            graph = [ii[:-1] for ii in graph]
            graph = graph[:-1]
            sematic_unit.pop(i)
        return graph, sematic_unit, dic

    def process_data(self, text_list, topic):
        text = ' '.join(text_list)
        docs = self.nlp(text)
        span, change = [], []
        for e in docs:
            if e._.in_coref and (not span or span[0]._.coref_clusters[0].main.text == e._.coref_clusters[0].main.text):
                span.append(e)
            elif span:
                start_index = span[0].idx
                end_index = span[-1].idx + len(span[-1].text)
                idx = 0
                while sum([len(t) for t in text_list[:idx + 1]]) + idx < start_index + 1:
                    idx += 1
                k = len(span) - 1
                while k >= 0 and end_index > sum([len(t) for t in text_list[:idx + 1]]) + idx:
                    k -= 1
                    end_index = span[k].idx + len(span[k].text)
                start_indexs = start_index - sum([len(t) for t in text_list[:idx]]) - idx
                end_indexs = end_index - sum([len(t) for t in text_list[:idx]]) - idx
                assert text_list[idx][start_indexs: end_indexs] == text[start_index: end_index]
                change.append([idx, start_indexs, end_indexs, span[0]._.coref_clusters[0].main.text])
                span = span[k + 1:]

        change.reverse()
        for i in change:
            idx, s, e, t = i
            text_list[idx] = list(text_list[idx])
            text_list[idx][s:e] = list(t)
            text_list[idx] = ''.join(text_list[idx])
        text = ' '.join(text_list)
        doc = self.nlp(text)
        tag, sematic_unit = None, []
        ind = []
        dic, sent_dic = {}, {}
        sent_num, text_sub = 0, 0
        for e in doc:
            if e.pos_ in ['SPACE', 'PUNCT']:
                if e.idx + len(e.text) - text_sub == len(text_list[sent_num]):
                    text_sub += len(text_list[sent_num]) + 1
                    sent_num += 1
                dic[e.i] = None
                continue
            if tag is not None and tag != e.tag_:
                sematic_unit.append(ind)
                ind = []
            dic[e.i] = len(sematic_unit)
            sent_dic[e.i] = sent_num
            tag = e.tag_
            ind.append(e)
            if e.idx + len(e.text) - text_sub == len(text_list[sent_num]):
                text_sub += len(text_list[sent_num]) + 1
                sent_num += 1

        assert sent_num == len(text_list)
        sent_num = len(text_list)
        if ind:
            sematic_unit.append(ind)
        # origin graph
        graph = [[0] * len(sematic_unit) for _ in range(len(sematic_unit))]
        for phrase in sematic_unit:
            for e in phrase:
                if dic[e.head.i]:
                    graph[dic[e.head.i]][dic[e.i]] = 1

        # 将介词合并
        prop = []
        for idx, phrase in enumerate(sematic_unit):
            for e in phrase:
                if e.dep_ in ['prep', 'aux']:
                    ru, chu = None, None
                    for j in range(len(sematic_unit)):
                        if graph[idx][j] == 1:
                            ru = j
                        if graph[j][idx] == 1:
                            chu = j
                    if ru and chu:
                        graph[chu][ru] = 1
                    prop.append(idx)
        graph, sematic_unit, dic = self.remove_node(prop, graph, sematic_unit, dic)

        # 处理连词
        conj = []
        for idx, phrase in enumerate(sematic_unit):
            for e in phrase:
                if e.pos_ == 'CCONJ':
                    conj.append(idx)
                elif e.dep_ == 'conj':
                    for i in range(len(sematic_unit)):
                        if dic[e.head.i]:
                            if graph[i][idx] == 1:
                                graph[i][dic[e.head.i]] = 1
                            if graph[i][dic[e.head.i]] == 1:
                                graph[i][idx] = 1
        graph, sematic_unit, dic = self.remove_node(conj, graph, sematic_unit, dic)

        # 合并共指关系
        node_text = []
        coref, newidx = [], []
        for idx, phrase in enumerate(sematic_unit):
            tex = ' '.join([e.text.lower() for e in phrase])
            if tex in node_text:
                i = node_text.index(tex)
                for j in range(len(graph)):
                    graph[i][j] = graph[i][j] | graph[idx][j]
                    graph[j][i] = graph[j][i] | graph[j][idx]
                coref.append(idx)
                newidx.append(i)
            node_text.append(tex)
        graph, sematic_unit, dic = self.remove_node(coref, graph, sematic_unit, dic, newidx)

        # 添加全局超级节点
        sematic_unit.append([topic])
        dic[len(dic)] = len(dic)
        graph.append([1] * len(graph))
        for i in range(len(graph)):
            graph[i] += [1]
        for i in range(len(graph)):
            for j in range(len(graph)):
                if graph[i][j] == 1:
                    graph[j][i] = 1
                if i == j:
                    graph[i][j] = 1

        # 添加句子节点
        graph_np = np.zeros((len(graph) + sent_num, len(graph) + sent_num), dtype=int)
        graph_np[:len(graph), :len(graph)] = graph
        for k, s in sent_dic.items():
            if dic[k] is not None:
                if dic[k] >= 0:
                    graph_np[len(graph) + s][dic[k]] = 1
                    graph_np[dic[k]][len(graph) + s] = 1
                else:
                    graph_np[len(graph) + s][dic[abs(dic[k])]] = 1
                    graph_np[dic[abs(dic[k])]][len(graph) + s] = 1
        # 前后句相连
        for i in range(1, sent_num):
            graph_np[len(graph) + i - 1][len(graph) + i] = 1
            graph_np[len(graph) + i][len(graph) + i - 1] = 1
        # super node
        graph_np[len(graph) - 1, :] = 1
        graph_np[:, len(graph) - 1] = 1
        graph = graph_np.tolist()

        # 修改sematic_unit存储格式
        sematic_u = []
        for idx, i in enumerate(sematic_unit):
            d = []
            for k in i:
                if isinstance(k, str):
                    d.append((k, -1))
                else:
                    d.append((k.text, k.i))
            sematic_u.append(d)

        # 将图存为稀疏矩阵
        # g = []
        # for i in range(len(graph)):
        #     for j in range(len(graph[0])):
        #         if graph[i][j] == 1:
        #             g.append((i, j))

        return {
            'graph': graph,
            'sematic_unit': sematic_u,
            'dict': dic,
            'ent_num': len(graph) - sent_num,
            'sent_num': sent_num,
            'text': text
        }


def grather_graph(graph_list):
    ent_num = [g['ent_num'] for g in graph_list]
    sent_num = [g['sent_num'] for g in graph_list]
    dic = [g['dict'] for g in graph_list]
    sematic_u = [g['sematic_unit'] for g in graph_list]
    text = [g['text'] for g in graph_list]
    gather_graph = np.zeros((sum(ent_num) + sum(sent_num), sum(ent_num) + sum(sent_num)))
    idx = 0
    node_mask = []
    for i, g in enumerate(graph_list):
        graph = g['graph']
        y = idx + sent_num[i] + ent_num[i]
        gather_graph[idx: y, idx: y] = graph
        idx = y
        node_mask += [0] * (ent_num[i] - 1) + [1] * (sent_num[i] + 1)
    # [no_passage_used]
    node_mask = node_mask[:-1]
    # 将图存为稀疏矩阵
    graph = []
    for i in range(len(gather_graph)):
        for j in range(len(gather_graph[0])):
            if gather_graph[i][j] == 1:
                graph.append((i, j))

    return {
               'graph': graph,
               'sematic_unit': sematic_u,
               'dict': dic,
               'ent_num': ent_num,
               'sent_num': sent_num,
               'text': text
           }, node_mask


class GraphDataset(Dataset):
    def __init__(self, tokenizer, split, args):
        """
        self._data: {
            'context': context,
            'user': user,
            'response': [],
            'knowledge': [],
            'gt': [],
            'graph': {
                'graph',
                'sematic_unit',
                'ent_num': semantic_unit + topic,
                'sent_num',
                'text',
                'dict'
            },
            'topics': [],
            'node_mask': []
        }
        """
        # load tokenizer
        self.ks_tokenizer = tokenizer

        self.usr_token = "[usr]"
        self.wizard_token = "[wizard]"
        self.know_token = "[know]"
        self.super_token = '[super]'

        self.ks_pad_id = self.ks_tokenizer.pad_token_id
        self.ks_usr_token_id = self.ks_tokenizer.convert_tokens_to_ids('[usr]')
        self.ks_wizard_token_id = self.ks_tokenizer.convert_tokens_to_ids('[wizard]')
        self.ks_know_token_id = self.ks_tokenizer.convert_tokens_to_ids('[know]')
        self.ks_super_token_id = self.ks_tokenizer.convert_tokens_to_ids('[super]')


        self.split = split
        # load data
        if not os.path.exists("{}/{}_p_{}/{}.json".format(args.data_dir, args.dataset, args.version, split)):
            # load model
            self.bert = BertModel.from_pretrained(args.bert_path)
            self.bert.to(args.device)
            self.device = args.device
            # process data
            data_path = "{}/{}/{}.json".format(args.data_dir, args.dataset, split)
            if args.version == 'topic':
                data_path = "{}/{}/{}_topic.json".format(args.data_dir, args.dataset, split)
            logger.info(f"loading ks data from {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)

            self.nlp = dataprocessor()
            if args.dataset == 'wow':
                _data = self._wow_preprocess(self._data)
            else:
                if args.version == 'v2':
                    _data = self._holle_preprocess(self._data)

            # save data
            if not os.path.exists("{}/{}_p_{}/".format(args.data_dir, args.dataset, args.version)):
                os.mkdir("{}/{}_p_{}/".format(args.data_dir, args.dataset, args.version))
            data_path = "{}/{}_p_{}/{}_0822.json".format(args.data_dir, args.dataset, args.version, split)
            with open(data_path, 'w') as f:
                json.dump(_data, f, ensure_ascii=True)

        self.device = args.device
        self.max_ent_node = args.max_ent_node
        self.is_ks = args.is_ks
        self.is_gpt = args.is_gpt
        self.dataset = args.dataset
        self.version = args.version
        if self.is_ks:
            data_path = "{}/{}_p_{}/{}_ks".format(args.data_dir, args.dataset, args.version, split)
            logger.info(f"loading ks data from {data_path}")
            files = os.listdir(data_path)
            self._n_data = len(files)
            self.data_path = data_path
            self.random_word_mask = args.random_word_mask
            logger.info(split + f" ks dataset size: {self._n_data}")
        

    def _get_embedding(self, sematic_unit):
        unit = []
        max_len = 0
        te = []
        for se in sematic_unit:
            unit_text = ' '.join([u for u in se])
            te.append(unit_text)
            inputs = self.ks_tokenizer.encode_plus(unit_text, add_special_tokens=False)
            max_len = max(max_len, len(inputs['input_ids']))
            unit.append(inputs)

        input_ids = [u['input_ids'] + [self.ks_pad_id] * (max_len - len(u['input_ids'])) for u in
                     unit]
        attention_mask = [u['attention_mask'] + [0] * (max_len - len(u['attention_mask'])) for u in
                          unit]
        for i in range(len(attention_mask)):
            attention_mask[i][0] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return [input_ids, attention_mask], te

    def _wow_preprocess(self, _data):
        data = []
        notopic = 0
        nochecked = 0
        toolong = 0
        nochecked_s = 0
        passgaes = {}
        # collect all passages
        for i in _data:
            dialog = i['dialog']
            for j in dialog:
                if 'retrieved_passages' in j.keys():
                    for f in j['retrieved_passages']:
                        k, v = list(f.items())[0]
                        if k not in passgaes:
                            passgaes[k] = v

        for i in tqdm(_data):
            dialog = i['dialog']
            if 'Wizard' in dialog[0]['speaker']:
                context = [i['persona']]
                user = ['[user]']
            else:
                context, user = [], []
            for d in dialog:
                data_ = {
                    'context': context.copy(),
                    'user': user.copy(),
                    'response': [],
                    'knowledge': [],
                    'gt': None,
                    'graph': {},
                    'topics': [],
                    'node_mask': []
                }
                utterance = d['text']
                context.append(utterance)
                g = []
                if 'Wizard' in d['speaker']:
                    data_['response'] = utterance
                    user.append('[wizard]')
                    checked_sent = list(d['checked_sentence'].values())

                    t = list(d['checked_passage'].values())
                    if t:
                        t = t[0]
                        if t not in d['retrieved_passages'] and t != 'no_passages_used' and t in passgaes:
                            d['retrieved_passages'].append({t: passgaes[t]})

                    for f in d['retrieved_passages']:
                        k, v = list(f.items())[0]
                        topic, know = k, [i for i in v if i != '']
                        data_['knowledge'].append(know)
                        data_['topics'].append(topic)
                        text = ' '.join([sent for sent in v])

                        if len(text.split(' ')) > 400:
                            data_['knowledge'].pop(-1)
                            data_['topics'].pop(-1)
                            toolong += 1
                            continue
                        # g.append(self.nlp.process_data(v, topic))
                        try:
                            g.append(self.nlp.process_data(v, topic))
                        except:
                            data_['knowledge'].pop(-1)
                            data_['topics'].pop(-1)
                            logger.error(f"{v} ... process failed")
                            continue

                    # add no passage used node
                    data_['knowledge'].append(["[no_passages_used]"])
                    data_['topics'].append("[no_passages_used]")
                    g.append({
                        'graph': [[1]],
                        'sematic_unit': [],
                        'dict': {0: 0},
                        'ent_num': 0,
                        'sent_num': 1,
                        'text': ""
                    })
                    data_['graph'], data_['node_mask'] = grather_graph(g)

                    # get ground_truth index
                    if checked_sent == ["no_passages_used"]:
                        # no passage is the ground truth
                        data_['gt'] = len(data_['node_mask']) - 1
                    else:
                        t = list(d['checked_passage'].values())
                        if not t:
                            # checked passage is None
                            nochecked += 1
                            continue
                        t = t[0]
                        if t not in data_['topics']:
                            # checked passage in not retrieved
                            notopic += 1
                            continue
                        if len(checked_sent) == 0:
                            # topic is the ground truth
                            t = list(d['checked_passage'].values())[0]
                            t_index = data_['topics'].index(t)
                            data_['gt'] = sum(data_['graph']['ent_num'][:t_index + 1]) + \
                                          sum(data_['graph']['sent_num'][:t_index]) - 1
                        else:
                            # sentence is the ground truth
                            try:
                                t = list(d['checked_passage'].values())[0]
                                t_index = data_['topics'].index(t)
                                key = checked_sent[0]
                                s_index = data_['knowledge'][t_index].index(key)
                                data_['gt'] = sum(data_['graph']['ent_num'][:t_index + 1]) + \
                                              sum(data_['graph']['sent_num'][:t_index]) + s_index
                            except:
                                nochecked_s += 1
                                continue
                    data.append(data_)
                else:
                    user.append('[user]')
        logger.info(f"Checked topic is not be retrieved: {notopic}!")
        logger.info(f"Checked sentence is not be retrieved: {nochecked_s}!")
        logger.info(f"No passage is checked: {nochecked}!")
        logger.info(f"Document is too long: {toolong}!")
        return data

    def _holle_preprocess(self, _data):
        data = []
        notopic = 0
        nochecked = 0
        toolong = 0
        nochecked_s = 0

        for i in tqdm(_data):
            dialog = i
            for d in dialog:
                context, user = d['text'][:-1], []
                for idx in range(len(context)):
                    if idx % 2 == 0:
                        user.append('[user]')
                    else:
                        user.append('[wizard]')
                data_ = {
                    'context': context.copy(),
                    'user': user.copy(),
                    'response': [d['text'][-1]],
                    'knowledge': [],
                    'gt': d['gt'],
                    'graph': {},
                    'topics': [],
                    'node_mask': []
                }
                if self.split == 'test':
                    data_["multi_eval_labels"] = d["multi_eval_labels"]
                    data_["multi_checked_sentences"] = d["multi_checked_sentences"]

                g = []
                checked_sent = d['checked_sentence']

                topic, know = d['title'], [i for i in d['knowledge'] if i != '' and i != "{}"]
                data_['knowledge'].append(know)
                data_['topics'].append(topic)
                
                k = []
                for i in d['knowledge']:
                    k += i
                k.append("[no_passages_used]")
                if isinstance(d['gt'], list):
                    if '' in k[:d['gt'][0]] or '{}' in k[:d['gt'][0]]:
                        tmp = k[:d['gt'][0]].count('') + k[:d['gt'][0]].count('{}')
                        for tmp_ids in range(len(d['gt'])):
                            d['gt'][tmp_ids] -= tmp
                else:
                    if '' in k[:d['gt']] or '{}' in k[:d['gt']]:
                        tmp = k[:d['gt']].count('') + k[:d['gt']].count('{}')
                        d['gt'] -= tmp
                data_['gt'] = d['gt']

                try:
                    g.append(self.nlp.process_data(know, topic))
                except:
                    logger.error(f"{know[:100]} ... process failed")
                    continue
                
                # add no passage used node
                data_['knowledge'].append("[no_passages_used]")
                data_['topics'].append("[no_passages_used]")
                g.append({
                    'graph': [[1]],
                    'sematic_unit': [],
                    'dict': {0: 0},
                    'ent_num': 0,
                    'sent_num': 1,
                    'text': ""
                })
                data_['graph'], data_['node_mask'] = grather_graph(g)

                # get ground_truth index
                if checked_sent == ["no_passages_used"]:
                    # no passage is the ground truth
                    data_['gt'] = len(data_['node_mask']) - 1
                data.append(data_)
        logger.info(f"Checked topic is not be retrieved: {notopic}!")
        logger.info(f"Checked sentence is not be retrieved: {nochecked_s}!")
        logger.info(f"No passage is checked: {nochecked}!")
        logger.info(f"Document is too long: {toolong}!")
        return data

    def __len__(self):
        return self._n_data

    def gpt_encoding(self, text):
        # if isinstance(self.gpt_tokenizer, Tokenizer):
        #     inputs = self.gpt_tokenizer.encode(text, bos=False, eos=False)
        #     return inputs 
        inputs = self.gpt_tokenizer(text,
                                    add_special_tokens=False,
                                    return_attention_mask=False,
                                    return_token_type_ids=False,
                                    max_length=1024)
        return inputs['input_ids']

    def ks_encoding(self, text, text_pair):
        return self.ks_tokenizer(text,
                                 text_pair=text_pair,
                                 add_special_tokens=True,
                                 return_attention_mask=True,
                                 return_token_type_ids=True,
                                 return_offsets_mapping=True,
                                 truncation='longest_first',
                                 max_length=512)

    def convert_ks_data(self):
        # load data
        data_path = "{}/{}_p_{}/{}.json".format(args.data_dir, args.dataset, self.version, self.split)
        with open(data_path, 'r', encoding='utf-8') as f:
            self._data = json.load(f)
        self._n_data = len(self._data)
        if not os.path.exists(f'./data/{self.dataset}_p_{self.version}/{self.split}_ks_masktp'):
            os.mkdir(f'./data/{self.dataset}_p_{args.version}/{self.split}_ks_masktp')
        count = -1
        for i in tqdm(range(self._n_data)):
            d = self.get_ks_item(i)
            if d != None:
                count += 1
            else:
                continue
            with open(f'./data/{self.dataset}_p_{args.version}/{self.split}_ks_masktp/{count}.json', 'w') as f:
                json.dump(d, f, ensure_ascii=True)

    def open_ks_file(self, item):
        data_path = f'{self.data_path}/{item}.json'
        with open(data_path, 'r', encoding='utf-8') as f:
            _data = json.load(f)
        for idx, i in enumerate(_data['ent_node_embed']):
            _data['ent_node_embed'][idx][0] = torch.tensor(i[0], dtype=torch.long)
            _data['ent_node_embed'][idx][1] = torch.tensor(i[1], dtype=torch.long)
        # random mask word node
        mask = random.sample(range(len(_data['node_mask'])), int(len(_data['node_mask']) * self.random_word_mask))
        for i in mask:
            if _data['node_mask'][i] == 0 and _data['word_node_gt'][i] == -100:
                _data['word_node_gt'][i] = 0
        _data['index'] = item
        return deepcopy(_data)

    def get_ks_item(self, data_i):
        data_item = self._data[data_i]
        knowledges = copy.deepcopy(data_item['knowledge'])
        history = copy.deepcopy(data_item['context'])   # [::-1]  # ["bla bla...", "bla bla.."] 倒置
        response = copy.deepcopy(data_item['response'])  # text
        user = copy.deepcopy(data_item['user'])  # [::-1]
        g = copy.deepcopy(data_item['graph'])
        gt = copy.deepcopy(data_item['gt'])
        node_mask = copy.deepcopy(data_item['node_mask'])
        graph, ent_num, sent_num, sematic_unit = g['graph'], g['ent_num'], g['sent_num'], g['sematic_unit']
        gt_len = None
        if self.dataset == 'Holl-E':
            # 对 holle，每7个句子分为一个文档
            if isinstance(gt, list):
                gt_len = len(gt)
                gt = gt[0]
            if self.version != 'topic':
                if gt != len(node_mask) - 1:
                    gt += ent_num[0]
                knowledge = []
                for i in range(len(knowledges) - 1):
                    tmp = []
                    for j in range(0, len(knowledges[i]), 5):
                        tmp.append(knowledges[i][j: j + 5])
                    knowledge.extend(tmp)
                knowledges = knowledge + [[knowledges[-1]]]
            else:
                knowledge = []
                for i in range(len(knowledges)):
                    tmp = []
                    for j in range(0, len(knowledges[i]), 10):
                        tmp.append(knowledges[i][j:j + 10])
                    knowledge.extend(tmp)
                knowledges = knowledge
                
        assert node_mask[gt] == 1
        # if the data is new version, this part should be annotated.
        for idx, doc in enumerate(sematic_unit):
            for idn, j in enumerate(doc):
                j = [k[0] for k in j]
                doc[idn] = j
            sematic_unit[idx] = doc

        # init node embedding
        ent_node_embed, tt = [], []
        # sematic_unit[-1] is []
        # sematic_unit[i][-1] is topic unit
        for i in range(len(sematic_unit) - 1):
            e, _ = self._get_embedding(sematic_unit[i])     # [sematic_unit_num]
            ent_node_embed.append(e)

        history, user = history[-4:], user[-4:]

        context_input = ''
        for i in range(len(history[-4:])):
            if user[i] == '[user]':
                role = self.usr_token
            else:
                role = self.wizard_token
            context_input += "{} {} ".format(role, history[i])
        if len(context_input) > 0:
            context_input = context_input[:-1]

        # get the contextualized text
        ids = []
        attention_mask = []
        token_type_ids = []
        know_ids = []
        for idn, knowledge in enumerate(knowledges):
            know_input = ''
            for i in knowledge:
                know_input += "{} {} ".format(self.know_token, i)
            if len(know_input) > 0:
                know_input = know_input[:-1]
            doc_input = self.ks_encoding(context_input, know_input)

            if doc_input['input_ids'].count(self.ks_know_token_id) != len(knowledge):
                know_input = ''
                for i in knowledges[:-1]:
                    know_input += "{} {} ".format(self.know_token, i[:int(len(i) * 0.7)])
                know_input += "{} {}".format(self.know_token, knowledges[-1])
                context_input_ = context_input[int(len(context_input) * 0.6):]
                doc_input = self.ks_encoding(context_input_, know_input)
                if doc_input['input_ids'].count(self.ks_know_token_id) != len(knowledge):
                    logger.error(f'Document is too long to truncate. The length of context: {len(context_input)}. '
                                 f'The length of knowledge: {len(know_input)}.')
                    return None
            know_id = [0] * len(doc_input['input_ids'])
            for idx, i in enumerate(doc_input['input_ids']):
                if i == self.ks_know_token_id:
                    know_id[idx] = 1
            know_ids.append(know_id)

            ids.append(doc_input['input_ids'])
            attention_mask.append(doc_input['attention_mask'])
            token_type_ids.append(doc_input['token_type_ids'])

        maxlen = max([len(i) for i in ids])
        ids = [s + [self.ks_pad_id] * (maxlen - len(s)) for s in ids]
        attention_mask = [s + [0] * (maxlen - len(s)) for s in attention_mask]
        token_type_ids = [s + [1] * (maxlen - len(s)) for s in token_type_ids]
        know_ids = [s + [0] * (maxlen - len(s)) for s in know_ids]

        graph_np = np.eye(sum(ent_num) + sum(sent_num))
        for i in graph:
            graph_np[i[0], i[1]] = 1

        num_sent = 0
        for i in range(len(know_ids)):
            for j in range(len(know_ids[0])):
                if know_ids[i][j] == 1:
                    num_sent += 1
        assert len(graph_np) == len(node_mask) - sum(node_mask) + num_sent + len(sent_num) - 1
            
        if sum(ent_num) + sum(sent_num) > self.max_ent_node:
            sumn = 0
            delete = []
            for i in range(len(ent_num) - 1):
                if ent_num[i] < 10:
                    sumn += ent_num[i]
                    sumn += sent_num[i]
                    continue
                s, e = sumn + int(ent_num[i] * 0.7), sumn + ent_num[i] - 1
                # keep topic node
                ent_node_embed[i][0] = torch.cat(
                    [ent_node_embed[i][0][:s - sumn], ent_node_embed[i][0][-1].unsqueeze(0)])
                ent_node_embed[i][1] = torch.cat(
                    [ent_node_embed[i][1][:s - sumn], ent_node_embed[i][1][-1].unsqueeze(0)])
                delete = [(s, e)] + delete
                # correct ground truth
                if sumn <= gt < sumn + sent_num[i] + ent_num[i]:
                    for j in delete:
                        gt = gt - (j[1] - j[0])

                sumn += ent_num[i]
                sumn += sent_num[i]
                ent_num[i] = int(ent_num[i] * 0.7) + 1

            for i in delete:
                graph_np = np.delete(graph_np, slice(i[0], i[1]), 0)
                graph_np = np.delete(graph_np, slice(i[0], i[1]), 1)
                node_mask = node_mask[: i[0]] + node_mask[i[1]:]
            # ground truth is the [no passage used]
            if gt > len(node_mask):
                gt = len(node_mask) - 1
        # graph_np = graph_np.tolist()

        # assert len(graph_np) == len(node_mask) - sum(node_mask) + num_sent + 1
        assert node_mask[gt] == 1 and gt >= 0

        word_node_gt = [0] * len(node_mask)

        for i in range(len(node_mask)):
            if node_mask[i] == 0 and (graph_np[i][gt] == 1 or graph_np[gt][i] == 1): #or (node_mask[i] == 0 and i in mask):
                word_node_gt[i] = 1
                continue
            if node_mask[i] == 1:
                word_node_gt[i] = -100

        idx, sumn = 0, 0
        while sumn < gt:
            idx += 1
            sumn = sum(ent_num[:idx]) + sum(sent_num[:idx])
        word_node_gt[: sum(ent_num[:idx - 1]) + sum(sent_num[:idx - 1])] = [-100] * (
                    sum(ent_num[:idx - 1]) + sum(sent_num[:idx - 1]))
        word_node_gt[sum(ent_num[:idx]) + sum(sent_num[:idx]):] = [-100] * (
            len(word_node_gt[sum(ent_num[:idx]) + sum(sent_num[:idx]):]))

        # convert ent_node_embed from tensor to list
        for idx, i in enumerate(ent_node_embed):
            ent_node_embed[idx][0] = i[0].tolist()
            ent_node_embed[idx][1] = i[1].tolist()
        if 'test' in self.split:
            if gt_len is not None:
                gt = [gt + i for i in range(gt_len)]
            else:
                if self.dataset != 'wow':
                    gt = [gt]
        else:
            if gt_len is not None:
                gt = [gt + i for i in range(gt_len)]
                gt = sum(gt) // len(gt)
        return {
            'input_ids': ids,  # [doc_num, seq_len]
            'attention_mask': attention_mask,  # [doc_num, seq_len]
            'token_type_ids': token_type_ids,  # [doc_num, seq_len]
            'graph': graph_np.tolist(),  # [node_len, node_len]
            'ent_node_embed': ent_node_embed,  # [doc_num - 1, tensor(ent_node_num, ent_node_text)]
            'know_ids': know_ids,  # [doc, seq_len]
            'gt': gt,
            'node_mask': node_mask,  # [node_len]
            'word_node_gt': word_node_gt,  # [node_len]
            'knowledge': data_item['knowledge'],
            'history': data_item['context'],
            'response': data_item['response'],
            'user': data_item['user'],
            'graph_origin': data_item['graph'],
            'node_mask_origin': data_item['node_mask'],
            'origin_gt': data_item['gt']
        }

    def __getitem__(self, data_i):
        ks_input, gpt_input = None, None
        if self.is_ks:
            # ks_input = self.get_ks_item(data_i)
            ks_input = self.open_ks_file(data_i)
        
        return ks_input, gpt_input


class Collate:
    def __init__(self, ks_tokenizer, args):
        self.ks_pad_id = ks_tokenizer.pad_token_id
        self.device = args.device
        self.is_ks = args.is_ks
        self.is_gpt = args.is_gpt

    # @profile(precision=4, stream=open('collate.log', 'w+'))
    def get_ks_batch(self, batch):
        input_ids = [sample['input_ids'] for sample in batch]
        attention_mask = [sample["attention_mask"] for sample in batch]
        token_type_ids = [sample['token_type_ids'] for sample in batch]
        graph = [sample['graph'] for sample in batch]
        ent_node_embed = [sample['ent_node_embed'] for sample in batch]
        know_ids = [sample['know_ids'] for sample in batch]
        node_mask = [sample['node_mask'] for sample in batch]
        gt = [sample['gt'] for sample in batch]
        word_node_gt = [sample['word_node_gt'] for sample in batch]
        bs = len(input_ids)
        
        maxlen = max([len(i[0]) for i in input_ids])
        maxdoc = max([len(i) for i in input_ids])
        if isinstance(gt[0], list):
            maxgt = max([len(i) for i in gt])
            gt = [s + [s[0]] * (maxgt - len(s)) for s in gt]

        maxlen_graph = max([len(i) for i in graph])
        for i in range(bs):
            input_ids[i] = [s + [self.ks_pad_id] * (maxlen - len(s)) for s in input_ids[i]]
            attention_mask[i] = [s + [0] * (maxlen - len(s)) for s in attention_mask[i]]
            token_type_ids[i] = [s + [1] * (maxlen - len(s)) for s in token_type_ids[i]]
            know_ids[i] = [s + [0] * (maxlen - len(s)) for s in know_ids[i]]
            input_ids[i] += [[self.ks_pad_id] * maxlen for _ in range(maxdoc - len(input_ids[i]))]
            attention_mask[i] += [[0] * maxlen for _ in range(maxdoc - len(attention_mask[i]))]
            token_type_ids[i] += [[0] * maxlen for _ in range(maxdoc - len(token_type_ids[i]))]
            know_ids[i] += [[0] * maxlen for _ in range(maxdoc - len(know_ids[i]))]

        graph_np = np.zeros((bs, maxlen_graph, maxlen_graph))

        for i in range(len(graph)):
            graph_np[i, :len(graph[i]), :len(graph[i])] = graph[i]

        node_mask = [s + [0] * (maxlen_graph - len(s)) for s in node_mask]
        word_node_gt = [s + [-100] * (maxlen_graph - len(s)) for s in word_node_gt]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        know_ids = torch.tensor(know_ids, dtype=torch.long)
        graph = torch.tensor(graph_np, dtype=torch.long)
        node_mask = torch.tensor(node_mask, dtype=torch.long)
        word_node_gt = torch.tensor(word_node_gt, dtype=torch.long)
        gt = torch.tensor(gt, dtype=torch.long)
        return input_ids, attention_mask, token_type_ids, graph, ent_node_embed, know_ids, node_mask, gt, word_node_gt

    def __call__(self, batch):
        ks_batch, gpt_batch = None, None
        ks_input, gpt_input = [i[0] for i in batch], [i[1] for i in batch]
        if self.is_ks:
            ks_batch = self.get_ks_batch(ks_input)
        return ks_batch, gpt_batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train for GraphKS')
    parser.add_argument('--dataset', type=str, default='wow')
    parser.add_argument('--version', type=str, default='v2')
    parser.add_argument('--bert_path', type=str, default='./data/model/bert-base-uncased')
    parser.add_argument('--gpt_path', type=str, default='./data/model/gpt2')
    parser.add_argument('--is_ks', type=bool, default=False)
    parser.add_argument('--is_gpt', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_ent_node', type=int, default=500)
    args = parser.parse_args()
    usr_token = "[usr]"
    wizard_token = "[wizard]"
    know_token = "[know]"
    super_token = '[super]'
    no_passage_used_token = '[no_passages_used]'

    # ks_tokenizer
    ks_tokenizer = BertTokenizerFast.from_pretrained(args.bert_path, do_lower_case=True)
    ks_tokenizer.add_tokens([usr_token], special_tokens=True)
    ks_tokenizer.add_tokens([wizard_token], special_tokens=True)
    ks_tokenizer.add_tokens([know_token], special_tokens=True)
    ks_tokenizer.add_tokens([super_token], special_tokens=True)
    ks_tokenizer.add_tokens([no_passage_used_token], special_tokens=True)

    # gpt_tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt_path)
    gpt_tokenizer = AutoTokenizer.from_pretrained(args.gpt_path)

    # gpt_tokenizer.add_tokens([usr_token], special_tokens=True)
    # gpt_tokenizer.add_tokens([wizard_token], special_tokens=True)
    # gpt_tokenizer.add_tokens([know_token], special_tokens=True)
    # gpt_tokenizer.add_tokens([super_token], special_tokens=True)
    # gpt_tokenizer.add_tokens([no_passage_used_token], special_tokens=True)
    test_seen_dataset = GraphDataset(ks_tokenizer, gpt_tokenizer, 'test_random_split', args)
    test_unseen_dataset = GraphDataset(ks_tokenizer, gpt_tokenizer, 'test_topic_split', args)
    # test_dataset = GraphDataset(ks_tokenizer, gpt_tokenizer, 'test', args)
    train_dataset = GraphDataset(ks_tokenizer, gpt_tokenizer, 'train', args)
    test_unseen_dataset.convert_ks_data()
    test_seen_dataset.convert_ks_data()
    train_dataset.convert_ks_data()
    # test_seen_dataset.convert_ks_data()
    
