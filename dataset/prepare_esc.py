#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : prepare_esc.py
# Author            : Zhepei Wang <zhepeiw2@illinois.edu>
# Date              : 02.12.2022
# Last Modified Date: 02.12.2022
# Last Modified By  : Zhepei Wang <zhepeiw2@illinois.edu>

import os
import pandas as pd
from tqdm import tqdm
import pdb


def prepare_esc_csv(
    root_dir,
    output_dir,
    train_folds,
    valid_folds,
    test_folds,
):
    meta_df = pd.read_csv(os.path.join(root_dir, 'meta', 'esc50.csv'))
    train_data, valid_data, test_data = [], [], []
    for idx, row in tqdm(meta_df.iterrows()):
        fold = row['fold']
        fname = os.path.join(root_dir, 'audio', row['filename'])
        class_name = row['category']
        if os.path.exists(fname):
            if fold in train_folds:
                train_data.append((fname, class_name))
            if fold in valid_folds:
                valid_data.append((fname, class_name))
            if fold in test_folds:
                test_data.append((fname, class_name))
    print('Found {} training, {} validation, {} test files'.format(len(train_data), len(valid_data), len(test_data)))
    for split, data in zip(['train', 'valid', 'test'], [train_data, valid_data, test_data]):
        if len(data) > 0:
            tmp_data = [{
                'ID': i,
                'wav_path': e[0],
                'class_name': e[1].replace('_', ' '),
            } for i, e in enumerate(data)]
            df = pd.DataFrame(tmp_data)
            csv_path = os.path.join(output_dir, '{}_raw.csv'.format(split))
            df.to_csv(csv_path, index=False)

def prepare_esc_label_encoder(
    root_dir,
    output_dir,
):
    train_csv = os.path.join(root_dir, 'meta', 'esc50.csv')
    train_meta = pd.read_csv(train_csv)
    class_list = set(zip(train_meta['target'], train_meta['category']))
    class_list = sorted(list(class_list), key=lambda i: i[0])
    class_names = [e[1].replace('_', ' ') for e in class_list]
    from speechbrain.dataio.encoder import CategoricalEncoder
    label_encoder = CategoricalEncoder()
    label_encoder.load_or_create(
        os.path.join(output_dir, 'label_encoder_esc.txt'),
        from_iterables=[class_names],
    )


if __name__ == '__main__':
    root_dir = '/mnt/data2/Sound Sets/ESC-50'
    output_dir = './'
    prepare_esc_label_encoder(root_dir, output_dir)
    #  prepare_esc_csv(root_dir, output_dir, [], [], [1, 2, 3, 4, 5])
