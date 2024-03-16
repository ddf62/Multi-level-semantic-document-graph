import json
import logging
import re
import sys
from collections import OrderedDict
from operator import itemgetter

from tqdm import tqdm
import spacy

sent_tok = spacy.load('en_core_web_sm')

logger = logging.getLogger('general')

_PLOT = 0
_REVIEW = 1
_COMMENTS = 2
_FACT_TABLE = 3
LABEL_ID2STR = {
    _PLOT: 'plot',
    _REVIEW: 'review',
    _COMMENTS: 'comments',
    _FACT_TABLE: 'fact_table'
}
_PUNCS_RE = re.compile(r'[^\w\s]')


def _remove_duplicate(a_list):
    return list(OrderedDict.fromkeys(a_list))


def _f1_score(true_set, pred_set, eps=sys.float_info.epsilon):
    precision = len(true_set.intersection(pred_set)) / (float(len(pred_set)) + eps)
    recall = len(true_set.intersection(pred_set)) / (float(len(true_set)) + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    return f1_score


def _check_continuity(bool_list):
    """Check if all matches are adjoint"""
    matched_indices = [idx for idx, is_match in enumerate(bool_list) if is_match]
    return all(a + 1 == b for a, b in zip(matched_indices[:-1], matched_indices[1:])), matched_indices


def validate_sentences(sentences):
    return [sent for sent in sentences if len(sent) > 0]


def validate_spacy_sentences(spacy_sentences):
    def _validate_sent(sent):
        if len(_PUNCS_RE.sub('', sent.text).strip()) > 1:
            return True
        else:
            False

    return [sent.text for sent in spacy_sentences.sents if _validate_sent(sent)]


def get_best_match_idx(gt_span, label_candidates, response):
    gt_span_words = set(gt_span.split())
    response_words = set(response.split())
    label_words_candidates = [
        set(x.split()) for x in label_candidates
    ]

    f1_scores = []
    for label_words_candidate in label_words_candidates:
        f1_scores.append(_f1_score(gt_span_words, label_words_candidate))

    if sum(f1_scores) == 0.0:
        f1_scores = []
        for label_words_candidate in label_words_candidates:
            f1_scores.append(_f1_score(response_words, label_words_candidate))

    max_idx = f1_scores.index(max(f1_scores))

    return max_idx


def get_gt_knowledge(raw_episode, knowledge_candidates, example_idx):
    label = raw_episode['labels'][example_idx + 1]
    label_str = LABEL_ID2STR.get(label, 'none')
    raw_gt_span = raw_episode['spans'][example_idx + 1]
    gt_span = _PUNCS_RE.sub('', raw_gt_span)
    raw_response = raw_episode['chat'][example_idx + 1]
    response = _PUNCS_RE.sub('', raw_response)

    # Find GT knowledge sentence
    if label_str == 'none':
        gt_knowledge = 'no_passages_used'
        gt_knowledge_idx = -1
    else:
        raw_label_candidates = knowledge_candidates[label_str]
        if label_str not in ['plot', 'review']:
            raw_label_candidates = _remove_duplicate(raw_label_candidates)
        label_candidates = [_PUNCS_RE.sub('', x) for x in raw_label_candidates]
        is_gt_in_cand = [gt_span in x for x in label_candidates]
        is_cand_in_gt = [x in gt_span for x in label_candidates]

        num_gt_in_cand = sum(is_gt_in_cand)
        num_cand_in_gt = sum(is_cand_in_gt)

        # Find matched candidate index
        if num_gt_in_cand == 1:  # Exact match
            gt_knowledge_idx = is_gt_in_cand.index(True)
        elif num_gt_in_cand > 1 or label in [_COMMENTS, _FACT_TABLE] or num_cand_in_gt == 0:
            # Find best match
            gt_knowledge_idx = get_best_match_idx(gt_span, label_candidates, response)
        elif num_cand_in_gt == 1:  # Inverse exact match
            gt_knowledge_idx = is_cand_in_gt.index(True)
        else:  # Span can exist over multiple sentences
            is_continue, matched_indices = _check_continuity(is_cand_in_gt)
            matched_words = ' '.join([label_candidates[idx] for idx in matched_indices])

            if is_continue and len(gt_span) > len(matched_words):
                add_front = gt_span.split()[-1] == matched_words.split()[-1]
                add_rear = gt_span.split()[0] == matched_words.split()[0]
                index_to_add_front = [] if matched_indices[0] == 0 else [matched_indices[0] - 1]
                if matched_indices[-1] + 1 == len(label_candidates):
                    index_to_add_rear = []
                else:
                    index_to_add_rear = [matched_indices[-1] + 1]

                if add_front:
                    matched_indices = index_to_add_front + matched_indices
                elif add_rear:
                    matched_indices = matched_indices + index_to_add_rear
                else:  # Add front & rear
                    matched_indices = index_to_add_front + matched_indices + \
                                      index_to_add_rear
                gt_knowledge_idx = matched_indices
            elif is_continue:
                gt_knowledge_idx = matched_indices
            else:
                gt_knowledge_idx = get_best_match_idx(
                    gt_span, label_candidates, response)

        # Get GT knowledge
        if isinstance(gt_knowledge_idx, int):
            gt_knowledge = raw_label_candidates[gt_knowledge_idx]
            gt_knowledge_idx = [gt_knowledge_idx]
        elif isinstance(gt_knowledge_idx, list):
            gt_knowledge = ' '.join(itemgetter(*gt_knowledge_idx)(raw_label_candidates))
        else:
            raise ValueError()

        # Remove GT from candidates
        # for idx in sorted(gt_knowledge_idx, reverse=True):
        #     del raw_label_candidates[idx]
        knowledge_candidates[label_str] = raw_label_candidates

    return gt_knowledge, knowledge_candidates, gt_knowledge_idx


def extract_fact_table(fact_table):
    if len(fact_table.keys()) == 2:
        return []

    awards = validate_sentences(fact_table['awards'])
    taglines = validate_sentences(fact_table['taglines'])
    similar_movies = validate_sentences(fact_table['similar_movies'])
    box_office = fact_table['box_office']
    if isinstance(box_office, str):
        box_office = [box_office if len(box_office) > 0 else []]
    else:
        box_office = []

    return awards + taglines + similar_movies + box_office


def get_knowledge_candidates(raw_episode, example_idx):
    # label = raw_episode['labels'][example_idx + 1]
    doc = raw_episode['documents']

    plot = validate_spacy_sentences(sent_tok(doc['plot']))
    review = validate_spacy_sentences(sent_tok(doc['review']))
    comments = doc['comments']
    fact_table = extract_fact_table(doc['fact_table'])
    knowledge_candidates = {
        'plot': plot,
        'review': review,
        'comments': comments,
        'fact_table': fact_table
    }

    return knowledge_candidates


def get_knowledge_sentences(raw_episode, episode_idx, example_idx, mode):
    # Handle special case
    if episode_idx == 5958 and mode == 'train':
        if example_idx in [0, 2]:
            return ['no_passages_used', 'Transformers: Aget of Extinction', '1']
        elif example_idx == 4 or example_idx == 8:  # review
            return ['1', 'Transformers: Age of Extinction']
        elif example_idx == 6:
            return ['Transformers: Age of Extinction', '1']

    label = raw_episode['labels'][example_idx + 1]
    label_str = LABEL_ID2STR.get(label, 'none')
    # Make GT and candidates
    knowledge_candidates = get_knowledge_candidates(raw_episode, example_idx)
    gt_knowledge, knowledge_candidates, gt_knowledge_idx = get_gt_knowledge(
        raw_episode, knowledge_candidates, example_idx
    )
    for key, value in knowledge_candidates.items():
        knowledge_candidates[key] = _remove_duplicate(value)

    # Concat GT and candidates
    # all_knowledge_sentences = [gt_knowledge]
    all_knowledge_sentences = []
    topic = []
    true_gt_idx, flag = 0, 0
    for key, candidates in knowledge_candidates.items():
        if key != label_str and not flag:
            true_gt_idx += len(candidates)
        if key == label_str:
            true_gt_idx = [true_gt_idx + i for i in gt_knowledge_idx]
            flag = 1
        all_knowledge_sentences.append(candidates)
        topic.append(key)

    return all_knowledge_sentences, topic, gt_knowledge, true_gt_idx


def to_wow_format(raw_episodes, mode):
    logger.info("Convert holle dataset to wow format")
    episodes = []
    for episode_idx, raw_episode in enumerate(tqdm(raw_episodes, ncols=70)):
        episode = []
        for example_idx in range(0, len(raw_episode['chat']), 2):
            if example_idx + 1 < len(raw_episode['chat']):
                chosen_topic = raw_episode['movie_name']
                response = raw_episode['chat'][example_idx + 1]
                try:
                    knowledge_sentences, topic, checked_sentence, gt_know_idx = get_knowledge_sentences(
                        raw_episode,
                        episode_idx,
                        example_idx,
                        mode
                    )
                except ValueError:
                    continue
                # checked_sentence = knowledge_sentences[0]
                title = 'no_passages_used' if checked_sentence == 'no_passages_used' else chosen_topic
                # formatted_knowledge = '\n'.join([
                #     chosen_topic + ' __knowledge__ ' + k
                #     if k != 'no_passages_used'
                #     else 'no_passages_used __knowledge__ no_passages_used=-
                #     for k in knowledge_sentences
                # ])
                formatted_knowledge = knowledge_sentences

                example = {
                    'text': raw_episode['chat'][:example_idx + 2],
                    'chosen_topic': chosen_topic,
                    'title': title,
                    'episode_num': episode_idx,
                    'example_num': example_idx // 2,
                    'checked_sentence': checked_sentence,
                    'knowledge': formatted_knowledge,
                    'topic': topic,
                    'gt': gt_know_idx
                }
                if mode == 'train':
                    example['labels'] = [response]
                else:
                    example['eval_labels'] = [response]
            episode.append(example)
        episodes.append(episode)
    return episodes


def to_wow_format_multi(raw_episodes, multi_responses, mode):
    print("Convert holle test dataset to wow format")
    episodes = []
    for episode_idx, raw_episode in enumerate(tqdm(raw_episodes, ncols=70)):
        episode = []
        multi_cnt = 0
        for example_idx in range(0, len(raw_episode['chat']), 2):
            if example_idx + 1 < len(raw_episode['chat']):
                chosen_topic = raw_episode['movie_name']
                response = raw_episode['chat'][example_idx + 1]
                knowledge_sentences, topic, checked_sentence, gt_know_idx = get_knowledge_sentences(
                    raw_episode,
                    episode_idx,
                    example_idx,
                    'test'
                )
                # checked_sentence = knowledge_sentences[0]
                title = 'no_passages_used' if checked_sentence == 'no_passages_used' else chosen_topic
                # formatted_knowledge = '\n'.join([
                #     chosen_topic + ' __knowledge__ ' + k
                #     if k != 'no_passages_used'
                #     else 'no_passages_used __knowledge__ no_passages_used'
                #     for k in knowledge_sentences
                # ])
                formatted_knowledge = knowledge_sentences
                example = {
                    'text': raw_episode['chat'][:example_idx + 2],
                    'chosen_topic': chosen_topic,
                    'title': title,
                    'episode_num': episode_idx,
                    'example_num': example_idx // 2,
                    'checked_sentence': checked_sentence,
                    'knowledge': formatted_knowledge,
                    'topic': topic,
                    'gt': gt_know_idx,
                    'eval_labels': [response],
                    'multi_eval_labels': [response],
                    'multi_checked_sentences': [checked_sentence]
                }

                # add multiple responses
                if multi_cnt < len(raw_episode['chat']) // 2:
                    if f'ts_{episode_idx}_{multi_cnt}' in multi_responses.keys():
                        multi_response_id = f'ts_{episode_idx}_{multi_cnt}'
                        for multi_idx in range(len(multi_responses[multi_response_id]['responses'])):
                            raw_multi_response = multi_responses[multi_response_id]['responses'][multi_idx]
                            raw_multi_span = multi_responses[multi_response_id]['spans'][multi_idx]
                            if raw_multi_response != response:
                                multi_response = _PUNCS_RE.sub('', str(raw_multi_response))
                                multi_span = _PUNCS_RE.sub('', str(raw_multi_span))
                                multi_knowledge_sentences = [_PUNCS_RE.sub('', str(x)) for x in knowledge_sentences]
                                multi_knowledge_idx = get_best_match_idx(multi_span, multi_knowledge_sentences, multi_response)
                                example['multi_eval_labels'].append(raw_multi_response)
                                example['multi_checked_sentences'].append(knowledge_sentences[multi_knowledge_idx])
                        multi_cnt += 1
            episode.append(example)
        episodes.append(episode)
    return episodes


if __name__ == '__main__':
    raw_fname = '/home/zhr/mychat/data/Holl-E/raw/test_data.json'
    with open(raw_fname, 'r') as fp:
        episodes = json.load(fp)
    multi_fname = '/home/zhr/mychat/data/Holl-E/raw/multi_reference_test.json'
    with open(multi_fname, 'r') as fp:
        multi_responses = json.load(fp)
    episodes = to_wow_format_multi(episodes, multi_responses, 'test')
    with open('/home/zhr/mychat/data/Holl-E/test_topic.json', 'w') as fp:
        json.dump(episodes, fp, ensure_ascii=True, indent=4)

    raw_fname = '/home/zhr/mychat/data/Holl-E/raw/train_data.json'
    with open(raw_fname, 'r') as fp:
        episodes = json.load(fp)
    episodes = to_wow_format(episodes, 'test')
    with open('/home/zhr/mychat/data/Holl-E/train_topic.json', 'w') as fp:
        json.dump(episodes, fp, ensure_ascii=True, indent=4)
    # print(episodes)
