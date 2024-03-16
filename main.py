import gc
import json
import logging
import os.path
import random

import numpy as np
import torch
import argparse

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizerFast, GPT2Tokenizer, GPT2TokenizerFast, get_linear_schedule_with_warmup, \
    LlamaTokenizerFast, RobertaTokenizerFast, AutoTokenizer
from model.tokenizer_llama import Tokenizer
from data_processer.graph_dataset import GraphDataset, Collate
from model.ksmodel import KsModel
from model.genmodel import GPTModel, LlamaModel
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import time
from sklearn.metrics import accuracy_score

from tools.inference import inference, inference_chatgpt
from tools.utils import get_unigram_F1, str2bool, remove_punctuation_and_words, compute_bleu
from tools.utils import ks_to_cuda, gpt_to_cuda, charge_multi_label
from tools.utils import lora_state_dict, mark_only_lora_as_trainable
from rouge import Rouge
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    GenerationConfig
)


def get_logger():
    loggers = logging.getLogger('general')
    logging.basicConfig(level=logging.DEBUG)
    return loggers


logger = get_logger()


def set_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(args.seed)


def get_dataloader(dataset, collate, shuffle, args):
    if shuffle:
        return DataLoader(dataset, collate_fn=collate, batch_size=args.train_batch_size, shuffle=shuffle, num_workers=args.num_workers,
                          pin_memory=True)
    else:
        return DataLoader(dataset, collate_fn=collate, batch_size=args.valid_batch_size, shuffle=shuffle, num_workers=args.num_workers,
                          pin_memory=True)

def valid_ks(train_dataloader, args):
    model.eval()
    nb_step, nb_loss = 0, 0
    loss_ft = nn.CrossEntropyLoss()
    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            # try:
            ks_batch, _ = batch
            input_ids, attention_mask, token_type_ids, graph, ent_node_embed, know_ids, node_mask, gt, word_node_gt = ks_to_cuda(
                ks_batch, args.device)
            tr_logits, word_logits = model(input_ids, attention_mask, token_type_ids, graph, ent_node_embed,
                                            know_ids, node_mask)
            
            sent_loss = loss_ft(tr_logits, gt)
            loss = sent_loss
            if args.add_word_loss:
                word_node_gt = word_node_gt.reshape(-1)
                word_logits = word_logits.reshape(-1, 2)
                word_loss = loss_ft(word_logits, word_node_gt)
                loss += args.word_loss_weight * word_loss

            tr_logits = tr_logits.detach().cpu().numpy()
            target = gt.detach().cpu().numpy()
            pred = np.argmax(tr_logits, axis=1)

            if len(target.shape) > 1:
                target = charge_multi_label(target, pred)
            preds = np.append(preds, pred)
            targets = np.append(targets, target)

            nb_step += 1
    epoch_loss = nb_loss / (nb_step + 1)
    targets, preds = targets.astype(int), preds.astype(int)
    acc = accuracy_score(targets, preds)
    logger.info("******************************************")
    logger.info(
        {"precision": acc, "loss": epoch_loss})
    del targets, preds
    gc.collect()
    torch.cuda.empty_cache()
    return acc


def train_ks(train_dataloader, stime, args):
    if not os.path.exists(f'./tb-logs/{stime}'):
        os.mkdir(f'./tb-logs/{stime}')
    writer = SummaryWriter(f'./tb-logs/{stime}')
    if not os.path.exists(args.output_path + '/' + stime):
        os.mkdir(args.output_path + '/' + stime)
    optimizer = torch.optim.AdamW([{'params': model.bert.parameters(), 'lr': args.learning_rate_for_bert,
                                    "weight_decay": args.weight_decay},
                                   {'params': model.graph_encoder.parameters(),
                                    'lr': args.learning_rate_for_others,
                                    "weight_decay": 0},
                                    {'params': model.project.parameters(),
                                    'lr': args.learning_rate_for_others,
                                    "weight_decay": 0},
                                   {'params': model.know_layer.parameters(),
                                    'lr': args.learning_rate_for_others,
                                    "weight_decay": 0},
                                    {'params': model.word_node_layer.parameters(),
                                    'lr': args.learning_rate_for_others,
                                    "weight_decay": 0}
                                   ],
                                  eps=1e-6,
                                  )
    total_steps = len(train_dataloader) * args.epoch // args.gradient_accumulation
    # linear warm up
    # sched = get_linear_schedule_with_warmup(optimizer,
    #                                         num_warmup_steps=0.2 * total_steps,
    #                                         num_training_steps=total_steps)
    # cos warm up
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        len(train_dataloader) * args.norm // args.gradient_accumulation,
        args.T_max
    )
    model.train()
    loss_ft = nn.CrossEntropyLoss(reduction='mean')
    global_step = 0

    for i in range(args.epoch):
        logger.info(f"### Epoch: {i + 1}")
        lr1 = optimizer.param_groups[0]['lr']
        lr2 = optimizer.param_groups[-1]['lr']
        logger.info(f'### LR_bert = {lr1}')
        logger.info(f'### LR_Linear = {lr2}')
        nb_step, nb_loss = 0, 0
        with tqdm(total=len(train_dataloader), desc="Train") as pbar:
            for idx, batch in enumerate(train_dataloader):
                ks_batch, _ = batch
                input_ids, attention_mask, token_type_ids, graph, ent_node_embed, know_ids, node_mask, gt, word_node_gt = ks_to_cuda(ks_batch, args.device)
                try:
                    tr_logits, word_logits = model(input_ids, attention_mask, token_type_ids, graph, ent_node_embed, know_ids, node_mask)
                    sent_loss = loss_ft(tr_logits, gt)
                    word_loss = torch.tensor(0)
                    loss = sent_loss
                    if args.add_word_loss:
                        word_node_gt = word_node_gt.reshape(-1)
                        word_logits = word_logits.reshape(-1, 2)
                        word_loss = loss_ft(word_logits, word_node_gt)
                        if not torch.isnan(word_loss):
                            loss += args.word_loss_weight * word_loss
                    p1, p2 = sent_loss.item(), word_loss.item()
                    tr_loss = loss.item()

                    # optimizer.zero_grad()
                    loss = loss / args.gradient_accumulation
                    loss.backward()
                    if idx % args.gradient_accumulation == 1:
                        torch.nn.utils.clip_grad_norm_(
                            parameters=model.parameters(), max_norm=args.max_grad_norm,
                            norm_type=1
                        )
                        optimizer.step()
                        sched.step()
                        optimizer.zero_grad()

                        # add tensorboard
                        writer.add_scalar("ks_sent_loss", p1, global_step)
                        writer.add_scalar("ks_word_loss", p2, global_step)
                        writer.add_scalar("ks_lr-rate-for-bert", optimizer.param_groups[0]['lr'], global_step)
                        writer.add_scalar("ks_lr-rate-for-others", optimizer.param_groups[1]['lr'], global_step)

                        nb_step += 1
                        nb_loss += tr_loss
                        global_step += 1

                    # update tqdm
                    pbar.set_postfix({'s_loss': '{0:1.5f}'.format(p1), 'w_loss': '{0:1.5f}'.format(p2)})
                    pbar.update(1)

                except RuntimeError as exception:
                    if 'out of memory' in str(exception):
                        sched.step()
                        del input_ids, attention_mask, token_type_ids, graph, ent_node_embed, know_ids, node_mask, gt, word_node_gt
                        optimizer.zero_grad()
                        gc.collect()
                        torch.cuda.empty_cache()
                        logger.error(f'{idx} out of memory!')
                        pbar.update(1)
                    else:
                        raise exception
                except AssertionError:
                    sched.step()
                    del input_ids, attention_mask, token_type_ids, graph, ent_node_embed, know_ids, node_mask, gt, word_node_gt
                    gc.collect()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    logger.error(f'{idx} is too long!')
                    pbar.update(1)

        epoch_loss = nb_loss / nb_step
        logger.info(f"Training loss epoch: {epoch_loss}")
        logger.info('Testing...')

        torch.cuda.empty_cache()
        acc = valid_ks(train_dataloader, args)
        torch.save({
            'epoch': i + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'sched_state_dict': sched.state_dict(),
            'loss': loss,
        }, args.output_path + f'/{stime}/epoch_{i + 1}_acc_{acc}.pth.tar')

        model.train()
        gc.collect()
        torch.cuda.empty_cache()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train for GraphKS')
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='wow')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_ent_node', type=int, default=600)
    parser.add_argument('--output_path', type=str, default='./outputs/wow')
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--valid_batch_size', type=int, default=8)
    parser.add_argument('--add_word_loss', type=str2bool, default=True)
    parser.add_argument('--random_word_mask', type=float, default=0.3)
    parser.add_argument('--word_loss_weight', type=float, default=1)
    parser.add_argument('--train_ks', type=str2bool, default=True)
    parser.add_argument('--test_ks', type=str2bool, default=False)
    parser.add_argument('--train_gpt', type=str2bool, default=True)
    parser.add_argument('--test_gpt', type=str2bool, default=False)
    parser.add_argument('--is_ks', type=str2bool, default=True)
    parser.add_argument('--is_gpt', type=str2bool, default=False)
    parser.add_argument('--is_llama', type=str2bool, default=False)
    parser.add_argument('--infer', type=str2bool, default=False)
    parser.add_argument('--log_file', type=str2bool, default=True)
    parser.add_argument('--bert_path', type=str, default='./data/model/bert-base-uncased')
    parser.add_argument('--gpt_path', type=str, default='./data/model/gpt2')
    parser.add_argument('--check_path', type=str, default='./data/model/gpt2')
    parser.add_argument('--max_seq_len', type=int, default=4096)
    parser.add_argument('--max_batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--nheads', type=int, default=3)
    parser.add_argument('--learning_rate_for_bert', type=float, default=5e-5)
    parser.add_argument('--learning_rate_for_others', type=float, default=2e-3)
    parser.add_argument('--max_grad_norm', type=float, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--norm', type=int, default=1)
    parser.add_argument('--T_max', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--visible_cuda', type=str, default='1')
    parser.add_argument('--version', type=str, default='topic')
    parser.add_argument('--gradient_accumulation', type=int, default=2)
    parser.add_argument('--alpha_t', type=float, default=0.8)
    parser.add_argument('--copy', type=str2bool, default=True)

    args = parser.parse_args()

    if args.log_file:
        log_path = './logs'
        start_time = time.localtime(time.time())
        start_time_str = f'{start_time[1]}-{start_time[2]}-{start_time[3]}:{start_time[4]}'
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logfile = f'{log_path}/log_' + start_time_str + '.log'

        log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                       datefmt='%m/%d/%Y %H:%M:%S')
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    
    # logger.info('word node 添加随机mask- 0.05')
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('%s: %s' % (k, vars(args)[k]))
    logger.info('--------args----------\n')

    start_time = time.localtime(time.time())
    stime = f'{start_time[1]}-{start_time[2]}-{start_time[3]}:{start_time[4]}'

    set_seed(args.seed)

    usr_token = "[usr]"
    wizard_token = "[wizard]"
    know_token = "[know]"
    super_token = '[super]'
    no_passage_used_token = '[no_passages_used]'

    # ks_tokenizer
    if args.version == 'roberta':
        ks_tokenizer = RobertaTokenizerFast.from_pretrained(args.bert_path, do_lower_case=True)
    else:
        ks_tokenizer = BertTokenizerFast.from_pretrained(args.bert_path, do_lower_case=True)
    ks_tokenizer.add_tokens([usr_token], special_tokens=True)
    ks_tokenizer.add_tokens([wizard_token], special_tokens=True)
    ks_tokenizer.add_tokens([know_token], special_tokens=True)
    ks_tokenizer.add_tokens([super_token], special_tokens=True)
    ks_tokenizer.add_tokens([no_passage_used_token], special_tokens=True)

    collate = Collate(ks_tokenizer, args)

    args.ks_vocab_len = len(ks_tokenizer)
    # args.gpt_vocab_len = len(gpt_tokenizer)
    # args.llama2_vocab_len = len(gpt_tokenizer)

    if args.is_ks:
        model = KsModel(args)
        model.to(args.device)
        logger.info(f'{model}')

    if args.train_ks or args.train_gpt:
        train_dataset = GraphDataset(ks_tokenizer, 'train', args)
        train_dataloader = get_dataloader(train_dataset, collate, shuffle=True, args=args)
        if args.train_ks:
            train_ks(train_dataloader, stime, args)

    if args.test_ks or args.infer:
        if args.dataset == 'wow':
            test_seen_dataset = GraphDataset(ks_tokenizer, 'test_random_split', args)
            test_unseen_dataset = GraphDataset(ks_tokenizer, 'test_topic_split', args)

            test_seen_dataloader = get_dataloader(test_seen_dataset, collate, shuffle=False, args=args)
            test_unseen_dataloader = get_dataloader(test_unseen_dataset, collate, shuffle=False, args=args)
        
            if args.test_file and not args.infer:
                file = args.test_file
                checkpoint = torch.load(args.output_path + f'/{file}')
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f'LOAD CHECKPOINT FROM {args.output_path}/{file}')
                logger.info('### test seen dataset:')
                valid_ks(test_seen_dataloader, args)
                logger.info('### test unseen dataset:')
                valid_ks(test_unseen_dataloader, args))
            else:
                filenames = os.listdir(args.output_path + f'/{stime}/')
                for file in filenames:
                    if file.endswith('.tar'):
                        checkpoint = torch.load(args.output_path + f'/{stime}/{file}')
                        model.load_state_dict(checkpoint['model_state_dict'])
                        logger.info(f'load checkpoint from {args.output_path}/{stime}/{file}')
                        logger.info('### test seen dataset:')
                        valid_ks(test_seen_dataloader, args)
                        logger.info('### test unseen dataset:')
                        valid_ks(test_unseen_dataloader, args)
                        logger.info('\n')
        else:
            test_dataset = GraphDataset(ks_tokenizer, 'test', args)
            test_dataloader = get_dataloader(test_dataset, collate, shuffle=False, args=args)
            train_dataset = GraphDataset(ks_tokenizer, 'train', args)
            train_dataloader = get_dataloader(train_dataset, collate, shuffle=False, args=args)
            if args.test_file and not args.infer:
                file = args.test_file
                checkpoint = torch.load(args.output_path + f'/{file}')
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f'LOAD CHECKPOINT FROM {args.output_path}/{file}')
                logger.info('### test dataset:')
                valid_ks(test_dataloader, args, 'test')
                logger.info('\n')
            else:
                filenames = os.listdir(args.output_path + f'/{stime}/')
                for file in filenames:
                    if file.endswith('.tar'):
                        checkpoint = torch.load(args.output_path + f'/{stime}/{file}')
                        model.load_state_dict(checkpoint['model_state_dict'])
                        logger.info(f'load checkpoint from {args.output_path}/{stime}/{file}')
                        logger.info('### test dataset:')
                        valid_ks(test_dataloader, args)
                        logger.info('\n')

