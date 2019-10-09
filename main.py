'''
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
'''

import os
from os.path import join, exists
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
import random
import json

from dataset import Dictionary, VQAFeatureDataset, VisualGenomeFeatureDataset
from dataset import tfidf_from_questions
from dataset_cp_v2 import VQA_cp_Dataset, Image_Feature_Loader
from model.regat import build_regat
from config.parser import parse_with_config
from train import train
import utils
from utils import trim_collate


def parse_args():
    parser = argparse.ArgumentParser()
    '''
    For training logistics
    '''
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay_start', type=int, default=15)
    parser.add_argument('--lr_decay_rate', type=float, default=0.25)
    parser.add_argument('--lr_decay_step', type=int, default=2)
    parser.add_argument('--lr_decay_based_on_val', action='store_true',
                        help='Learning rate decay when val score descreases')
    parser.add_argument('--grad_accu_steps', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=0.25)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output', type=str, default='saved_models/')
    parser.add_argument('--save_optim', action='store_true',
                        help='save optimizer')
    parser.add_argument('--log_interval', type=int, default=-1,
                        help='Print log for certain steps')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')

    '''
    loading trained models
    '''
    parser.add_argument('--checkpoint', type=str, default="")

    '''
    For dataset
    '''
    parser.add_argument('--dataset', type=str, default='vqa',
                        choices=["vqa", "vqa_cp"])
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--use_both', action='store_true',
                        help='use both train/val datasets to train?')
    parser.add_argument('--use_vg', action='store_true',
                        help='use visual genome dataset to train?')
    parser.add_argument('--adaptive', action='store_true',
                        help='adaptive or fixed number of regions')
    '''
    Model
    '''
    parser.add_argument('--relation_type', type=str, default='implicit',
                        choices=["spatial", "semantic", "implicit"])
    parser.add_argument('--fusion', type=str, default='mutan',
                        choices=["ban", "butd", "mutan"])
    parser.add_argument('--tfidf', action='store_true',
                        help='tfidf word embedding?')
    parser.add_argument('--op', type=str, default='c',
                        help="op used in tfidf word embedding")
    parser.add_argument('--num_hid', type=int, default=1024)
    '''
    Fusion Hyperparamters
    '''
    parser.add_argument('--ban_gamma', type=int, default=1, help='glimpse')
    parser.add_argument('--mutan_gamma', type=int, default=2, help='glimpse')
    '''
    Hyper-params for relations
    '''
    # hyper-parameters for implicit relation
    parser.add_argument('--imp_pos_emb_dim', type=int, default=64,
                        help='geometric embedding feature dim')

    # hyper-parameters for explicit relation
    parser.add_argument('--spa_label_num', type=int, default=11,
                        help='number of edge labels in spatial relation graph')
    parser.add_argument('--sem_label_num', type=int, default=15,
                        help='number of edge labels in \
                              semantic relation graph')

    # shared hyper-parameters
    parser.add_argument('--dir_num', type=int, default=2,
                        help='number of directions in relation graph')
    parser.add_argument('--relation_dim', type=int, default=1024,
                        help='relation feature dim')
    parser.add_argument('--nongt_dim', type=int, default=20,
                        help='number of objects consider relations per image')
    parser.add_argument('--num_heads', type=int, default=16,
                        help='number of attention heads \
                              for multi-head attention')
    parser.add_argument('--num_steps', type=int, default=1,
                        help='number of graph propagation steps')
    parser.add_argument('--residual_connection', action='store_true',
                        help='Enable residual connection in relation encoder')
    parser.add_argument('--label_bias', action='store_true',
                        help='Enable bias term for relation labels \
                              in relation encoder')

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)
    return args


if __name__ == '__main__':
    args = parse_args()
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available," +
                         "this code currently only support GPU.")
    n_device = torch.cuda.device_count()
    print("Found %d GPU cards for training" % (n_device))
    device = torch.device("cuda")
    batch_size = args.batch_size*n_device

    torch.backends.cudnn.benchmark = True

    if args.seed != -1:
        print("Predefined randam seed %d" % args.seed)
    else:
        # fix seed
        args.seed = random.randint(1, 10000)
        print("Choose random seed %d" % args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if "ban" == args.fusion:
        fusion_methods = args.fusion+"_"+str(args.ban_gamma)
    else:
        fusion_methods = args.fusion

    dictionary = Dictionary.load_from_file(
                    join(args.data_folder, 'glove/dictionary.pkl'))
    if args.dataset == "vqa_cp":
        coco_train_features = Image_Feature_Loader(
                            'train', args.relation_type,
                            adaptive=args.adaptive, dataroot=args.data_folder)
        coco_val_features = Image_Feature_Loader(
                            'val', args.relation_type,
                            adaptive=args.adaptive, dataroot=args.data_folder)
        val_dset = VQA_cp_Dataset(
                    'test', dictionary, coco_train_features, coco_val_features,
                    adaptive=args.adaptive, pos_emb_dim=args.imp_pos_emb_dim,
                    dataroot=args.data_folder)
        train_dset = VQA_cp_Dataset(
                    'train', dictionary, coco_train_features,
                    coco_val_features, adaptive=args.adaptive,
                    pos_emb_dim=args.imp_pos_emb_dim,
                    dataroot=args.data_folder)
    else:
        val_dset = VQAFeatureDataset(
                'val', dictionary, args.relation_type, adaptive=args.adaptive,
                pos_emb_dim=args.imp_pos_emb_dim, dataroot=args.data_folder)
        train_dset = VQAFeatureDataset(
                'train', dictionary, args.relation_type,
                adaptive=args.adaptive, pos_emb_dim=args.imp_pos_emb_dim,
                dataroot=args.data_folder)

    model = build_regat(val_dset, args).to(device)

    tfidf = None
    weights = None
    if args.tfidf:
        tfidf, weights = tfidf_from_questions(['train', 'val', 'test2015'],
                                              dictionary)
    model.w_emb.init_embedding(join(args.data_folder,
                                    'glove/glove6b_init_300d.npy'),
                               tfidf, weights)

    model = nn.DataParallel(model).to(device)

    if args.checkpoint != "":
        print("Loading weights from %s" % (args.checkpoint))
        if not os.path.exists(args.checkpoint):
            raise ValueError("No such checkpoint exists!")
        checkpoint = torch.load(args.checkpoint)
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
        print("Unexpected_keys:", list(unexpected_keys))
        print("Missing_keys:", list(missing_keys))
        model.load_state_dict(matched_state_dict, strict=False)

    # use train & val splits to optimize, only available for vqa, not vqa_cp
    if args.use_both and args.dataset == "vqa":
        length = len(val_dset)
        trainval_concat_dset = ConcatDataset([train_dset, val_dset])
        if args.use_vg or args.use_visdial:
            trainval_concat_dsets_split = random_split(
                                trainval_concat_dset,
                                [int(0.2*length),
                                 len(trainval_concat_dset)-int(0.2*length)])
        else:
            trainval_concat_dsets_split = random_split(
                                trainval_concat_dset,
                                [int(0.1*length),
                                 len(trainval_concat_dset)-int(0.1*length)])
        concat_list = [trainval_concat_dsets_split[1]]

        # use a portion of Visual Genome dataset
        if args.use_vg:
            vg_train_dset = VisualGenomeFeatureDataset(
                        'train', train_dset.features, train_dset.normalized_bb,
                        train_dset.bb, train_dset.spatial_adj_matrix,
                        train_dset.semantic_adj_matrix, dictionary,
                        adaptive=train_dset.adaptive,
                        pos_boxes=train_dset.pos_boxes,
                        dataroot=args.data_folder)
            vg_val_dset = VisualGenomeFeatureDataset(
                            'val', val_dset.features, val_dset.normalized_bb,
                            val_dset.bb, val_dset.spatial_adj_matrix,
                            val_dset.semantic_adj_matrix, dictionary,
                            adaptive=val_dset.adaptive,
                            pos_boxes=val_dset.pos_boxes,
                            dataroot=args.data_folder)
            concat_list.append(vg_train_dset)
            concat_list.append(vg_val_dset)
        final_train_dset = ConcatDataset(concat_list)
        final_eval_dset = trainval_concat_dsets_split[0]
        train_loader = DataLoader(final_train_dset, batch_size, shuffle=True,
                                  num_workers=4, collate_fn=trim_collate)
        eval_loader = DataLoader(final_eval_dset, batch_size,
                                 shuffle=False, num_workers=4,
                                 collate_fn=trim_collate)

    else:
        train_loader = DataLoader(train_dset, batch_size, shuffle=True,
                                  num_workers=4, collate_fn=trim_collate)
        eval_loader = DataLoader(val_dset, batch_size, shuffle=False,
                                 num_workers=4, collate_fn=trim_collate)

    output_meta_folder = join(args.output, "regat_%s" % args.relation_type)
    utils.create_dir(output_meta_folder)
    args.output = output_meta_folder+"/%s_%s_%s_%d" % (
                fusion_methods, args.relation_type,
                args.dataset, args.seed)
    if exists(args.output) and os.listdir(args.output):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output))
    utils.create_dir(args.output)
    with open(join(args.output, 'hps.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)
    logger = utils.Logger(join(args.output, 'log.txt'))

    train(model, train_loader, eval_loader, args, device)
