import torch
import matplotlib.pyplot as plt
import re
import torch
import torch.nn.functional as F
from torch import linalg


def fix_indices(index_list, string_list):
    new_index_list = []
    curr_start = index_list[0][0]
    curr_end = index_list[0][1]

    for i in range(1, len(index_list)):
        if string_list[i][0] != ' ' and curr_end == index_list[i][0]:
            curr_end = index_list[i][1]
        else:
            new_index_list.append([curr_start, curr_end])
            curr_start = index_list[i][0]
            curr_end = index_list[i][1]

    new_index_list.append([curr_start, curr_end])
    return new_index_list


def get_offset_mapping(token_list):
    current = len(token_list[2])-1
    res = [(0, current)]

    for token in token_list[3:]:
        if token == '</s>':
            res.append((current,current))
        elif token.startswith('▁'):
            res.append((current, current+len(token)))
            current += len(token)
        elif token != '<pad>':
            res.append((current, current+len(token)))
            current += len(token)
        else: break
    return res


def capture_NE_indices_model(sentence_list, NER_model):
    ner_ress = NER_model.predict(sentence_list)
    res_dict = {}
    for n, ner_res in enumerate(ner_ress):
        if len(ner_res) == 0: continue
        temp = [[x['start'], x['end']] for x in ner_res]
        res_dict[n] = fix_indices(temp, sentence_list[n])
    return res_dict


def entropy_algorithm(scores, encoded_output):
    res = []
    for n in range(len(encoded_output)):
        try:
            output_cut = torch.argwhere(encoded_output[n] == 2)[1].item()
        except Exception:
            output_cut = len(encoded_output[n])
        cut_scores = scores[n, 1:output_cut, :]
        probs = F.softmax(cut_scores, dim=1)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        res.append(entropy.cpu())
    return res


def el2n_algorithm(scores, encoded_output, answer):
    res = []
    for n in range(len(encoded_output)):
        if len(answer[n]) > scores[n].shape[0]:
            answer_cut = scores[n].shape[0]
        else:
            answer_cut = len(answer[n])
        cut_scores = scores[n, 1:answer_cut, :]
        el = answer[n][1:answer_cut] - cut_scores
        el2n = linalg.vector_norm(el, dim=1)
        res.append(el2n.cpu())
    return res

