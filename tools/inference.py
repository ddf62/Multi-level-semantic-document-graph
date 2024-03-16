import copy
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from data_processer.graph_dataset import dataprocessor
from tools.utils import ks_to_cuda


def inference(model, dataset, collate, path, args):
    fp = open(path)
    data = json.load(fp)
    model.eval()
    preds, grounds = [[], [], [], [], []], [[], [], [], [], []]
    dataset.nlp = dataprocessor()
    path = './infer'
    if not os.path.exists(path):
        os.mkdir(path)
    num = [0, 0, 0, 0, 0]
    thread = 100
    random.seed(55)
    random.shuffle(data)
    for i in range(len(data)):
        d = data[i]
        graph = dataset._preprocess([d])
        dataset._data = copy.deepcopy(graph)
        for k in range(len(graph)):
            output = copy.deepcopy(graph[k])
            output['graph'].pop('dict')
            know = []
            w, p = 0, 0
            passage = []
            word = []
            wd_gt = []
            wd_pred = []
            for h in range(len(output['knowledge'])):
                passage.append(output['topics'][h])
                passage += output['knowledge'][h]
            print(len(passage))
            if len(passage) <= 40:
                index = 0
                if num[0] < thread:
                    num[0] += 1
                    index = 0
                else:
                    continue
            elif len(passage) <= 45:
                index = 1
                # if num[1] < thread:
                #     num[1] += 1
                #     index = 1
                # else:
                #     continue
            elif len(passage) <= 50:
                index = 2
                # if num[2] < thread:
                #     num[2] += 1
                #     index = 2
                # else:
                #     continue
            elif len(passage) <= 55:
                index = 3
                # if num[3] < thread:
                #     num[3] += 1
                #     index = 3
                # else:
                #     continue
            else:
                index = 4
                # if num[4] < thread:
                #     num[4] += 1
                #     index = 4
                # else:
                #     continue
            for h in output['graph']['sematic_unit']:
                for jj in h:
                    t = ' '.join([tk[0] for tk in jj])
                    word.append(t)
            pre = -1
            for h in output['node_mask']:
                if h == 0:
                    if pre == 1:
                        w += 1
                    know.append(word[w])
                    w += 1
                else:
                    know.append(passage[p])
                    p += 1
                pre = h
            output['knowledge'] = know

            edge = []
            output['edge'] = edge
            output.pop('graph')

            with torch.no_grad():
                input = dataset.get_ks_item(k)
                for idx, h in enumerate(input['ent_node_embed']):
                    input['ent_node_embed'][idx][0] = torch.tensor(h[0], dtype=torch.long)
                    input['ent_node_embed'][idx][1] = torch.tensor(h[1], dtype=torch.long)

                input = collate.get_ks_batch([input])
                input_ids, attention_mask, token_type_ids, graphs, ent_node_embed, know_ids, node_mask, gt, word_node_gt = ks_to_cuda(
                    input, args.device)
                tr_logits, word_logits = model(input_ids, attention_mask, token_type_ids, graphs, ent_node_embed,
                                               know_ids, node_mask)
                tr_logits = tr_logits.detach().cpu().numpy()
                node_mask = node_mask.detach().cpu().numpy()[0]
                graphs = graphs.detach().cpu().numpy()[0]
                word_node_gt = word_node_gt.reshape(-1)
                word_logits = word_logits.reshape(-1, 2)
                word_node_gt = word_node_gt.detach().cpu().numpy()
                word_logits = word_logits.detach().cpu().numpy()
                oh = 0
                kk = []
                for h in range(len(word_node_gt)):
                    while node_mask[h] != output['node_mask'][oh]:
                        oh += 1
                    kk.append(know[oh])
                    oh += 1
                    if word_node_gt[h] == 1 or word_node_gt[h] == 0:
                        wd_gt.append([kk[-1], int(word_node_gt[h])])
                        if word_logits[h][0] > word_logits[h][1]:
                            wd_pred.append([0, float(word_logits[h][0]), float(word_logits[h][1])])
                        else:
                            wd_pred.append([1, float(word_logits[h][0]), float(word_logits[h][1])])

                for h in range(len(graphs)):
                    for hh in range(len(graphs[0])):
                        if graphs[h][hh] == 1:
                            edge.append([kk[h], kk[hh]])
                output['edge'] = edge
                output['knowledge'] = kk
                node_mask = node_mask.tolist()
                output['node_mask'] = node_mask
                output['gt'] = kk[gt[0]]
                output['wd'] = wd_gt
                output['wd_pred'] = wd_pred
                target = gt.detach().cpu().numpy()[0]
                pred = np.argmax(tr_logits, axis=1)[0]
                preds[index].append(pred)
                grounds[index].append(target)
                # print(output)
                # passe = 0
                # for h in node_mask[0][:pred]:
                #     if h == 1:
                #         passe += 1
                # print(tr_logits[0][pred - 5: pred + 5], tr_logits[0][pred], node_mask[0][pred], output['node_mask'][pred])
            output['pred'] = kk[pred]
            # print(passage[passe], output['gt'])
            with open(path + '/' + str(i) + '_' + str(k) + '.json', 'w') as f:
                json.dump(output, f, ensure_ascii=True, indent=4)
            if not any([h != thread for h in num]):
                break
        print(num)
        if not any([h != thread for h in num]):
            break
    g = []
    for i in grounds:
        g += i
    p = []
    for i in preds:
        p += i
    print('acc-40: ',  accuracy_score(grounds[0], preds[0]))
    print('acc-45: ', accuracy_score(grounds[1], preds[1]))
    print('acc-50: ', accuracy_score(grounds[2], preds[2]))
    print('acc-55: ', accuracy_score(grounds[3], preds[3]))
    print('acc>55: ', accuracy_score(grounds[4], preds[4]))
    print('acc-all:', accuracy_score(g, p))
    print('seed: ', 55)


def inference_chatgpt(model, dataset, collate, path, args):
    fp = open(path)
    data = json.load(fp)
    model.eval()
    preds, grounds = [], []
    dataset.nlp = dataprocessor()
    path = './infer-holle'
    if not os.path.exists(path):
        os.mkdir(path)
    num = [0, 0, 0, 0, 0]
    thread = 100
    random.seed(55)
    # random.shuffle(data)
    dataset._data = copy.deepcopy(data)
    dataset._n_data = len(data)
    for i in range(100):
        d = data[i]
        output = copy.deepcopy(d)
        passage = []
        for h in range(len(output['knowledge'])):
            passage.append(output['topics'][h])
            passage += output['knowledge'][h]
        word = []
        wd_gt = []
        wd_pred = []
        edge = []
        know = []
        w, p = 0, 0
        for h in output['graph']['sematic_unit']:
            for jj in h:
                t = ' '.join([tk[0] for tk in jj])
                word.append(t)
        pre = -1
        for h in output['node_mask']:
            if h == 0:
                if pre == 1:
                    w += 1
                know.append(word[w])
                w += 1
            else:
                know.append(passage[p])
                p += 1
            pre = h
        output['knowledge'] = know

        edge = []
        output['edge'] = edge
        output.pop('graph')
        with torch.no_grad():
            input = dataset.get_ks_item(i)
            for idx, h in enumerate(input['ent_node_embed']):
                input['ent_node_embed'][idx][0] = torch.tensor(h[0], dtype=torch.long)
                input['ent_node_embed'][idx][1] = torch.tensor(h[1], dtype=torch.long)

            input = collate.get_ks_batch([input])
            input_ids, attention_mask, token_type_ids, graphs, ent_node_embed, know_ids, node_mask, gt, word_node_gt = ks_to_cuda(
                input, args.device)
            tr_logits, word_logits = model(input_ids, attention_mask, token_type_ids, graphs, ent_node_embed,
                                            know_ids, node_mask)
            tr_logits = tr_logits.detach().cpu().numpy()
            node_mask = node_mask.detach().cpu().numpy()[0]
            graphs = graphs.detach().cpu().numpy()[0]
            word_node_gt = word_node_gt.reshape(-1)
            word_logits = word_logits.reshape(-1, 2)
            word_node_gt = word_node_gt.detach().cpu().numpy()
            word_logits = word_logits.detach().cpu().numpy()
            oh = 0
            kk = []
            for h in range(len(word_node_gt)):
                while node_mask[h] != output['node_mask'][oh]:
                    oh += 1
                kk.append(know[oh])
                oh += 1
                if word_node_gt[h] == 1 or word_node_gt[h] == 0:
                    wd_gt.append([kk[-1], int(word_node_gt[h])])
                    if word_logits[h][0] > word_logits[h][1]:
                        wd_pred.append([0, float(word_logits[h][0]), float(word_logits[h][1])])
                    else:
                        wd_pred.append([1, float(word_logits[h][0]), float(word_logits[h][1])])

            for h in range(len(graphs)):
                for hh in range(len(graphs[0])):
                    if graphs[h][hh] == 1:
                        edge.append([kk[h], kk[hh]])
            output['edge'] = edge
            output['knowledge'] = kk
            node_mask = node_mask.tolist()
            output['node_mask'] = node_mask
            output['gt'] = kk[gt[0]]
            output['wd'] = wd_gt
            output['wd_pred'] = wd_pred
            target = gt.detach().cpu().numpy()[0]
            pred = np.argmax(tr_logits, axis=1)[0]
            preds.append(pred)
            grounds.append(target)
            # print(output)
            # passe = 0
            # for h in node_mask[0][:pred]:
            #     if h == 1:
            #         passe += 1
            # print(tr_logits[0][pred - 5: pred + 5], tr_logits[0][pred], node_mask[0][pred], output['node_mask'][pred])
        output['pred'] = kk[pred]
        # print(passage[passe], output['gt'])
        with open(path + '/' + str(i) + '.json', 'w') as f:
            json.dump(output, f, ensure_ascii=True, indent=4)
        if not any([h != thread for h in num]):
            break
    print(num)
    # print('acc-40: ',  accuracy_score(grounds[0], preds[0]))
    # print('acc-45: ', accuracy_score(grounds[1], preds[1]))
    # print('acc-50: ', accuracy_score(grounds[2], preds[2]))
    # print('acc-55: ', accuracy_score(grounds[3], preds[3]))
    # print('acc>55: ', accuracy_score(grounds[4], preds[4]))
    print('acc-all:', accuracy_score(grounds, preds))
    # print('seed: ', 55)