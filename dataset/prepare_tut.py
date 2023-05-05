#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : prepare_tut.py
# Author            : Zhepei Wang <zhepeiw2@illinois.edu>
# Date              : 24.04.2023
# Last Modified Date: 24.04.2023
# Last Modified By  : Zhepei Wang <zhepeiw2@illinois.edu>

import os
from tqdm import tqdm
import pandas as pd
import pdb


def prepare_tut2017_dev_eval_csv(
    root_dir,
    output_dir,
):
    '''
        preapare train, valid and test csv for tut2017 dataset

        valid and train csv are from the dev set
    '''
    train_path = os.path.join(root_dir, 'TUT-acoustic-scenes-2017-development', 'meta.txt')
    with open(train_path, 'r') as f_train:
        train_info = f_train.readlines()
    test_path = os.path.join(root_dir, 'TUT-acoustic-scenes-2017-evaluation', 'meta.txt')
    with open(test_path, 'r') as f_test:
        test_info = f_test.readlines()

    for split, meta_df in zip(['train', 'valid', 'test'], [train_info, train_info, test_info]):
        data = []
        for row in tqdm(meta_df):
            line = row.split('\t')
            split_dir = 'TUT-acoustic-scenes-2017-development' if split != 'test' \
                    else 'TUT-acoustic-scenes-2017-evaluation'
            fname = os.path.join(root_dir, split_dir, line[0])
            if os.path.exists(fname):
                # stripping out special characters in the labels
                class_name = line[1].replace('_', ' ').replace('/', ' ')
                data.append({
                    'wav_path': fname,
                    'class_name': class_name,
                })
        df = pd.DataFrame(data)
        df['ID'] = range(len(df))
        cols = ['ID', 'wav_path', 'class_name']
        df = df[cols]
        task_path = os.path.join(output_dir, '{}_raw.csv'.format(split))
        df.to_csv(task_path, index=False)


def prepare_tut2017_full_csv(
    root_dir,
    output_dir,
    output_split='test',
):
    train_path = os.path.join(root_dir, 'TUT-acoustic-scenes-2017-development', 'meta.txt')
    with open(train_path, 'r') as f_train:
        train_info = f_train.readlines()
    test_path = os.path.join(root_dir, 'TUT-acoustic-scenes-2017-evaluation', 'meta.txt')
    with open(test_path, 'r') as f_test:
        test_info = f_test.readlines()

    data = []
    for split, meta_df in zip(['train', 'test'], [train_info, test_info]):
        for row in tqdm(meta_df):
            line = row.split('\t')
            split_dir = 'TUT-acoustic-scenes-2017-development' if split != 'test' \
                    else 'TUT-acoustic-scenes-2017-evaluation'
            fname = os.path.join(root_dir, split_dir, line[0])
            if os.path.exists(fname):
                # stripping out special characters in the labels
                class_name = line[1].replace('_', ' ').replace('/', ' ')
                data.append({
                    'wav_path': fname,
                    'class_name': class_name,
                })
    df = pd.DataFrame(data)
    df['ID'] = range(len(df))
    cols = ['ID', 'wav_path', 'class_name']
    df = df[cols]
    task_path = os.path.join(output_dir, '{}_raw.csv'.format(output_split))
    df.to_csv(task_path, index=False)


def prepare_tut2017_label_encoder(
    root_dir,
    output_dir,
):
    train_path = os.path.join(root_dir, 'TUT-acoustic-scenes-2017-development', 'meta.txt')
    with open(train_path, 'r') as f_train:
        train_info = f_train.readlines()
    class_names = sorted(list(set([line.split('\t')[1] for line in train_info])))
    class_names = [e.replace('_', ' ').replace('/', ' ') for e in class_names]
    from speechbrain.dataio.encoder import CategoricalEncoder
    label_encoder = CategoricalEncoder()
    label_encoder.load_or_create(
        os.path.join(output_dir, 'label_encoder_tut2017_ordered.txt'),
        from_iterables=[class_names],
    )


if __name__ == '__main__':
    prepare_tut2017_dev_eval_csv(
        '/mnt/data2/Sound Sets/TUT-acoustic-scenes-2017',
        './tmp'
    )
    #  prepare_tut2017_label_encoder(
    #      '/mnt/data2/Sound Sets/TUT-acoustic-scenes-2017',
    #      './'
    #  )
