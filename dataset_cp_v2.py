"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""
from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
import tools.compute_softscore
from dataset import is_howmany

COUNTING_ONLY = False


def _create_entry(img, question, answer):
    if answer is not None:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'coco_split': question["coco_split"],
        'question': question['question'],
        'answer': answer}
    return entry


def _load_dataset(dataroot, name, coco_train_img_id2val, coco_val_img_id2val,
                  label2ans):
    """Load entries

    coco_train_img_id2val/coco_val_img_id2val:
        dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'cp_v2_questions/vqacp_v2_%s_questions.json' % name)
    questions = sorted(json.load(open(question_path)),
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', 'cp_v2_%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        coco_split = question["coco_split"]
        index = coco_train_img_id2val[img_id]\
            if coco_split == "train2014" else coco_val_img_id2val[img_id]
        if not COUNTING_ONLY \
           or is_howmany(question['question'], answer, label2ans):
            entries.append(_create_entry(index, question, answer))
    return entries


class Image_Feature_Loader():
    def __init__(self, coco_split, relation_type, dataroot='data',
                 adaptive=True):
        super(Image_Feature_Loader, self).__init__()
        assert coco_split in ['train', 'val']
        self.adaptive = adaptive
        self.relation_type = relation_type
        prefix = '36'

        self.img_id2idx = pickle.load(
            open(os.path.join(dataroot, 'imgids/%s%s_imgid2idx.pkl' %
                              (coco_split, '' if self.adaptive else prefix)),
                 'rb'))
        h5_dataroot = dataroot+"/Bottom-up-features-adaptive" \
            if self.adaptive else dataroot+"/Bottom-up-features-fixed"
        h5_path = os.path.join(h5_dataroot,
                               '%s%s.hdf5' % (coco_split,
                                              '' if self.adaptive else prefix))

        print('loading features from h5 file %s' % h5_path)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))
            self.bb = np.array(hf.get('image_bb'))
            if "semantic_adj_matrix" in hf.keys() \
               and self.relation_type == "semantic":
                self.semantic_adj_matrix = np.array(
                                            hf.get('semantic_adj_matrix'))
                print("Loaded semantic adj matrix from file...",
                      self.semantic_adj_matrix.shape)
            else:
                self.semantic_adj_matrix = None
                print("Setting semantic adj matrix to None...")
            if "image_adj_matrix" in hf.keys() \
               and self.relation_type == "spatial":
                self.spatial_adj_matrix = np.array(hf.get('image_adj_matrix'))
                print("Loaded spatial adj matrix from file...",
                      self.spatial_adj_matrix.shape)
            else:
                self.spatial_adj_matrix = None
                print("Setting spatial adj matrix to None...")
            self.pos_boxes = None
            if self.adaptive:
                self.pos_boxes = np.array(hf.get('pos_boxes'))
        self.tensorize()

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)
        self.bb = torch.from_numpy(self.bb)
        if self.semantic_adj_matrix is not None:
            self.semantic_adj_matrix = torch.from_numpy(
                                        self.semantic_adj_matrix).double()
        if self.spatial_adj_matrix is not None:
            self.spatial_adj_matrix = torch.from_numpy(
                                        self.spatial_adj_matrix).double()
        if self.pos_boxes is not None:
            self.pos_boxes = torch.from_numpy(self.pos_boxes)


class VQA_cp_Dataset(Dataset):
    def __init__(self, name, dictionary, coco_train_features,
                 coco_val_features, dataroot='data', adaptive=False,
                 pos_emb_dim=64):
        super(VQA_cp_Dataset, self).__init__()
        assert name in ['train', 'test']

        ans2label_path = os.path.join(dataroot, 'cache',
                                      'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache',
                                      'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary
        self.adaptive = adaptive
        self.relation_type = coco_train_features.relation_type
        self.coco_train_features = coco_train_features
        self.coco_val_features = coco_val_features
        self.entries = _load_dataset(dataroot, name,
                                     self.coco_train_features.img_id2idx,
                                     self.coco_val_features.img_id2idx,
                                     self.label2ans)
        self.tokenize()
        self.tensorize()
        self.emb_dim = pos_emb_dim
        self.v_dim = self.coco_train_features.features.size(1 if self.adaptive
                                                            else 2)
        self.s_dim = self.coco_train_features.spatials.size(1 if self.adaptive
                                                            else 2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad to the back of the sentence
                padding = [self.dictionary.padding_idx] * \
                          (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            if answer is not None:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        raw_question = entry["question"]
        image_id = entry["image_id"]
        coco_split = entry["coco_split"]

        question = entry['q_token']
        question_id = entry['question_id']
        if "train" in coco_split:
            coco_features = self.coco_train_features
        elif "val" in coco_split:
            coco_features = self.coco_val_features
        else:
            print("Unknown coco split: %s" % coco_split)

        if coco_features.spatial_adj_matrix is not None:
            spatial_adj_matrix = coco_features.spatial_adj_matrix[
                                    entry["image"]]
        else:
            spatial_adj_matrix = torch.zeros(1).double()
        if coco_features.semantic_adj_matrix is not None:
            semantic_adj_matrix = coco_features.semantic_adj_matrix[
                                    entry["image"]]
        else:
            semantic_adj_matrix = torch.zeros(1).double()

        if not self.adaptive:
            # fixed number of bounding boxes
            features = coco_features.features[entry['image']]
            spatials = coco_features.spatials[entry['image']]
            bb = coco_features.bb[entry["image"]]
        else:
            features = coco_features.features[
                coco_features.pos_boxes[
                    entry['image']][0]:coco_features.pos_boxes[
                                                    entry['image']][1], :]
            spatials = coco_features.spatials[
                coco_features.pos_boxes[
                    entry['image']][0]:coco_features.pos_boxes[
                                                    entry['image']][1], :]
            bb = coco_features.bb[
                coco_features.pos_boxes[
                    entry['image']][0]:coco_features.pos_boxes[
                                                    entry['image']][1], :]

        answer = entry['answer']
        if answer is not None:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return features, spatials, question, target, question_id,\
                image_id, bb, spatial_adj_matrix, semantic_adj_matrix

        else:
            return features, spatials, question, question_id, question_id,\
                image_id, bb, spatial_adj_matrix, semantic_adj_matrix

    def __len__(self):
        return len(self.entries)
