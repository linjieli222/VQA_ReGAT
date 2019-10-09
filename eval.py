"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import json

from dataset import Dictionary, VQAFeatureDataset
from dataset_cp_v2 import VQA_cp_Dataset, Image_Feature_Loader
from model.regat import build_regat
from train import compute_score_with_logits
from model.position_emb import prepare_graph_variables
from config.parser import Struct
import utils


@torch.no_grad()
def evaluate(model, dataloader, model_hps, args, device):
    model.eval()
    label2ans = dataloader.dataset.label2ans
    num_answers = len(label2ans)
    relation_type = dataloader.dataset.relation_type
    N = len(dataloader.dataset)
    results = []
    score = 0
    pbar = tqdm(total=len(dataloader))

    if args.save_logits:
        idx = 0
        pred_logits = np.zeros((N, num_answers))
        gt_logits = np.zeros((N, num_answers))

    for i, (v, norm_bb, q, target, qid, _, bb,
            spa_adj_matrix, sem_adj_matrix) in enumerate(dataloader):
        batch_size = v.size(0)
        num_objects = v.size(1)
        v = Variable(v).to(device)
        norm_bb = Variable(norm_bb).to(device)
        q = Variable(q).to(device)
        pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables(
            relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
            model_hps.nongt_dim, model_hps.imp_pos_emb_dim,
            model_hps.spa_label_num, model_hps.sem_label_num, device)
        pred, att = model(v, norm_bb, q, pos_emb, sem_adj_matrix,
                          spa_adj_matrix, None)
        # Check if target is a placeholder or actual targets
        if target.size(-1) == num_answers:
            target = Variable(target).to(device)
            batch_score = compute_score_with_logits(
                pred, target, device).sum()
            score += batch_score
            if args.save_logits:
                gt_logits[idx:batch_size+idx, :] = target.cpu().numpy()

        if args.save_logits:
            pred_logits[idx:batch_size+idx, :] = pred.cpu().numpy()
            idx += batch_size

        if args.save_answers:
            qid = qid.cpu()
            pred = pred.cpu()
            current_results = make_json(pred, qid, dataloader)
            results.extend(current_results)

        pbar.update(1)

    score = score / N
    results_folder = f"{args.output_folder}/results"
    if args.save_logits:
        utils.create_dir(results_folder)
        save_to = f"{results_folder}/logits_{args.dataset}" +\
            f"_{args.split}.npy"
        np.save(save_to, pred_logits)

        utils.create_dir("./gt_logits")
        save_to = f"./gt_logits/{args.dataset}_{args.split}_gt.npy"
        if not os.path.exists(save_to):
            np.save(save_to, gt_logits)
    if args.save_answers:
        utils.create_dir(results_folder)
        save_to = f"{results_folder}/{args.dataset}_" +\
            f"{args.split}.json"
        json.dump(results, open(save_to, "w"))
    return score


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results


def parse_args():
    parser = argparse.ArgumentParser()

    '''
    For eval logistics
    '''
    parser.add_argument('--save_logits', action='store_true',
                        help='save logits')
    parser.add_argument('--save_answers', action='store_true',
                        help='save poredicted answers')

    '''
    For loading expert pre-trained weights
    '''
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--output_folder', type=str, default="",
                        help="checkpoint folder")

    '''
    For dataset
    '''
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='vqa',
                        choices=["vqa", "vqa_cp"])
    parser.add_argument('--split', type=str, default="val",
                        choices=["train", "val", "test", "test2015"],
                        help="test for vqa_cp, test2015 for vqa")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available," +
                         "this code currently only support GPU.")

    n_device = torch.cuda.device_count()
    print("Found %d GPU cards for eval" % (n_device))
    device = torch.device("cuda")

    dictionary = Dictionary.load_from_file(
                 os.path.join(args.data_folder, 'glove/dictionary.pkl'))

    hps_file = f'{args.output_folder}/hps.json'
    model_hps = Struct(json.load(open(hps_file)))
    batch_size = model_hps.batch_size*n_device

    print("Evaluating on %s dataset with model trained on %s dataset" %
          (args.dataset, model_hps.dataset))
    if args.dataset == "vqa_cp":
        coco_train_features = Image_Feature_Loader(
                            'train', model_hps.relation_type,
                            adaptive=model_hps.adaptive,
                            dataroot=model_hps.data_folder)
        coco_val_features = Image_Feature_Loader(
                            'val', model_hps.relation_type,
                            adaptive=model_hps.adaptive,
                            dataroot=model_hps.data_folder)
        eval_dset = VQA_cp_Dataset(
                    args.split, dictionary, coco_train_features,
                    coco_val_features, adaptive=model_hps.adaptive,
                    pos_emb_dim=model_hps.imp_pos_emb_dim,
                    dataroot=model_hps.data_folder)
    else:
        eval_dset = VQAFeatureDataset(
                args.split, dictionary, model_hps.relation_type,
                adaptive=model_hps.adaptive,
                pos_emb_dim=model_hps.imp_pos_emb_dim,
                dataroot=model_hps.data_folder)

    model = build_regat(eval_dset, model_hps).to(device)

    model = nn.DataParallel(model).to(device)

    if args.checkpoint > 0:
        checkpoint_path = os.path.join(
                            args.output_folder,
                            f"model_{args.checkpoint}.pth")
    else:
        checkpoint_path = os.path.join(args.output_folder,
                                       f"model.pth")
    print("Loading weights from %s" % (checkpoint_path))
    if not os.path.exists(checkpoint_path):
        raise ValueError("No such checkpoint exists!")
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint.get('model_state', checkpoint)
    matched_state_dict = {}
    unexpected_keys = set()
    missing_keys = set()
    for name, param in model.named_parameters():
        missing_keys.add(name)
    for key, data in state_dict.items():
        if key in missing_keys:
            matched_state_dict[key] = data
            missing_keys.remove(key)
        else:
            unexpected_keys.add(key)
    print("\tUnexpected_keys:", list(unexpected_keys))
    print("\tMissing_keys:", list(missing_keys))
    model.load_state_dict(matched_state_dict, strict=False)

    eval_loader = DataLoader(
        eval_dset, batch_size, shuffle=False,
        num_workers=4, collate_fn=utils.trim_collate)

    eval_score = evaluate(
        model, eval_loader, model_hps, args, device)

    print('\teval score: %.2f' % (100 * eval_score))
