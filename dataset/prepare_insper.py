#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : prepare_insper.py
# Author            : Zhepei Wang <zhepeiw2@illinois.edu>
# Date              : 28.11.2022
# Last Modified Date: 28.11.2022
# Last Modified By  : Zhepei Wang <zhepeiw2@illinois.edu>

import os
from tqdm import tqdm
import speechbrain as sb
import torch
import torchaudio
import pandas as pd
import re
import random
import numpy as np
from copy import deepcopy
import pdb


def prepare_audiocaps_csv(
    source_dir,
    source_csv,
    output_dir,
    output_csv=False,
):
    train_df = pd.read_csv(source_csv, sep='\t', header=None)
    temp_data = []
    cnt = 0
    for idx, row in tqdm(train_df.iterrows()):
        if row[0][0] == '/':
            fname = row[0][1:]
        else:
            fname = row[0]
        wav_path = os.path.join(source_dir, fname)
        if not os.path.exists(wav_path):
            #  print('file {} not found'.format(fname))
            continue
        caption = row[1].replace(',', '').strip()
        temp_data.append({
            'ID': cnt,
            'wav_path': wav_path,
            'caption': caption,
        })
        cnt += 1
    if output_csv:
        df = pd.DataFrame(temp_data)
        df.to_csv(os.path.join(output_dir, 'train_audiocaps_raw.csv'), index=False)
    print('Found {} audio files'.format(cnt))
    return temp_data


def prepare_macs_csv(
    source_dir,
    source_csv,
    output_dir,
    output_csv=False,
):
    train_df = pd.read_csv(source_csv, sep='\t', header=None)
    temp_data = []
    cnt = 0
    for idx, row in tqdm(train_df.iterrows()):
        if row[0][0] == '/':
            fname = row[0][1:]
        else:
            fname = row[0]
        wav_path = os.path.join(source_dir, fname)
        if not os.path.exists(wav_path):
            #  print('file {} not found'.format(fname))
            continue
        caption = row[1].strip()
        caption = caption.replace('\n', ' ').replace(',', '').strip()
        temp_data.append({
            'ID': cnt,
            'wav_path': wav_path,
            'caption': caption,
        })
        cnt += 1
    if output_csv:
        df = pd.DataFrame(temp_data)
        df.to_csv(os.path.join(output_dir, 'train_macs_raw.csv'), index=False)
    print('Found {} audio files'.format(cnt))
    return temp_data


def prepare_clotho_csv(
    source_dir,
    source_csv,
    output_dir,
    output_csv=False,
):
    train_df = pd.read_csv(source_csv, sep='\t', header=None)
    temp_data = []
    cnt = 0
    for idx, row in tqdm(train_df.iterrows()):
        if row[0][0] == '/':
            fname = row[0][1:]
        else:
            fname = row[0]
        wav_path = os.path.join(source_dir, fname)
        if not os.path.exists(wav_path):
            #  print('file {} not found'.format(fname))
            continue
        caption = row[1].strip()
        caption = caption.replace('\n', ' ').replace(',', '').strip()
        temp_data.append({
            'ID': cnt,
            'wav_path': wav_path,
            'caption': caption,
        })
        cnt += 1
    if output_csv:
        df = pd.DataFrame(temp_data)
        df.to_csv(os.path.join(output_dir, 'train_clotho_raw.csv'), index=False)
    print('Found {} audio files'.format(cnt))
    return temp_data


def prepare_fsd50k_csv(
    source_dir,
    source_csv,
    output_dir,
    output_csv=False,
):
    train_df = pd.read_csv(source_csv, sep='\t', header=None)
    temp_data = []
    cnt = 0
    for idx, row in tqdm(train_df.iterrows()):
        if row[0][0] == '/':
            fname = row[0][1:]
        else:
            fname = row[0]
        wav_path = os.path.join(source_dir, fname)
        if not os.path.exists(wav_path):
            #  print('file {} not found'.format(fname))
            continue
        caption = row[1].strip()
        tags = re.split(r'<a.*</a>', caption)
        tags = [e.strip().replace('\n', ' ').replace('\r', '').replace(',', '').replace('$', '').replace('*', '') for e in tags]
        caption = ' '.join(tags).strip()
        temp_data.append({
            'ID': cnt,
            'wav_path': wav_path,
            'caption': caption,
        })
        cnt += 1
    if output_csv:
        df = pd.DataFrame(temp_data)
        df.to_csv(os.path.join(output_dir, 'train_fsd50k_raw.csv'), index=False)
    print('Found {} audio files'.format(cnt))
    return temp_data


def prepare_insper_multi_csv(
    source_dir,
    output_dir,
    csv_paths=[
        'insper_audiocaps_index.tsv',
        'insper_clotho_index.tsv',
        'insper_macs_index.tsv',
        'insper_fsd50k_index.tsv',
    ],
    prepare_fns=[
        prepare_audiocaps_csv,
        prepare_clotho_csv,
        prepare_macs_csv,
        prepare_fsd50k_csv
    ],
    valid_pct=0,
    subset_pct=1,
    overfit_size=None,
    shuffle_text=False,
):
    all_data = []
    assert len(csv_paths) == len(prepare_fns)
    csv_paths = [os.path.join(source_dir, e) for e in csv_paths]
    for csv_path, prepare_fn in zip(csv_paths, prepare_fns):
        temp_data = prepare_fn(source_dir, csv_path, output_dir, output_csv=False)
        all_data += temp_data
    # train/valid split
    random.seed(123)
    random.shuffle(all_data)
    if overfit_size is None:
        train_idx = int(len(all_data) * valid_pct)
        train_data = all_data[train_idx:]
        train_data = train_data[:int(len(train_data)*subset_pct)]
        for idx, dic in enumerate(train_data):
            dic['ID'] = idx
        train_df = pd.DataFrame(train_data)
        if shuffle_text:
            train_df['caption'] = np.random.permutation(train_df['caption'].values)
        train_df.to_csv(os.path.join(output_dir, 'train_raw.csv'), index=False)
        if valid_pct > 0:
            valid_data = all_data[:train_idx]
            for idx, dic in enumerate(valid_data):
                dic['ID'] = idx
            valid_df = pd.DataFrame(valid_data)
            if shuffle_text:
                valid_df['caption'] = np.random.permutation(valid_df['caption'].values)
            valid_df.to_csv(os.path.join(output_dir, 'valid_raw.csv'), index=False)
    else:
        overfit_data = deepcopy(all_data[:overfit_size])
        #  repeat_times = len(all_data) // overfit_size
        repeat_times = 1000
        train_data = []
        for _ in range(repeat_times):
            train_data += deepcopy(overfit_data)
        for idx, dic in enumerate(train_data):
            dic['ID'] = idx
        train_df = pd.DataFrame(train_data)
        if shuffle_text:
            train_df['caption'] = np.random.permutation(train_df['caption'].values)
        train_df.to_csv(os.path.join(output_dir, 'train_raw.csv'), index=False)
        valid_data = deepcopy(all_data[:overfit_size])
        for idx, dic in enumerate(valid_data):
            dic['ID'] = idx
        valid_df = pd.DataFrame(valid_data)
        if shuffle_text:
            valid_df['caption'] = np.random.permutation(valid_df['caption'].values)
        valid_df.to_csv(os.path.join(output_dir, 'valid_raw.csv'), index=False)


def dataio_audiocaps_prep(hparams, csv_path, is_train=True):
    "Creates the datasets and their data processing pipelines."

    config_sample_rate = hparams["sample_rate"]
    # TODO  use SB implementation but need to make sure it give the same results as PyTorch
    # resampler = sb.processing.speech_augmentation.Resample(orig_freq=latest_file_sr, new_freq=config_sample_rate)
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    def random_segment(sig, target_len):
        if len(sig) > target_len:
            rstart = torch.randint(0, len(sig) - target_len + 1, (1,)).item()
            return sig[rstart:rstart+target_len]
        else:
            new_sig = torch.zeros(target_len, dtype=sig.dtype)
            rstart = torch.randint(0, target_len - len(sig) + 1, (1,)).item()
            new_sig[rstart:rstart + len(sig)] = sig
            return new_sig

    def center_segment(sig, target_len):
        if len(sig) > target_len:
            rstart = len(sig) // 2 - target_len // 2
            return sig[rstart:rstart+target_len]
        else:
            new_sig = torch.zeros(target_len, dtype=sig.dtype)
            rstart = target_len // 2 - len(sig) // 2
            new_sig[rstart:rstart + len(sig)] = sig
            return new_sig

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav_path")
    #  @sb.utils.data_pipeline.provides("sig1", "sig2")
    @sb.utils.data_pipeline.provides("sig1")
    def audio_pipeline(wav_path):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        sig, read_sr = torchaudio.load(wav_path)

        # If multi-channels, downmix it to a mono channel
        sig = torch.squeeze(sig)
        if len(sig.shape) > 1:
            sig = torch.mean(sig, dim=0)

        # Convert sample rate to required config_sample_rate
        if read_sr != config_sample_rate:
            # Re-initialize sampler if source file sample rate changed compared to last file
            if read_sr != hparams["resampler"].orig_freq:
                hparams["resampler"] = torchaudio.transforms.Resample(
                    orig_freq=read_sr, new_freq=config_sample_rate
                )
            # Resample audio
            sig = hparams["resampler"].forward(sig)

        #  # scaling
        #  max_amp = torch.abs(sig).max().item()
        #  #  assert max_amp > 0
        #  if max_amp == 0:
        #      scaling = 1
        #  else:
        #      scaling = 1 / max_amp * 0.9
        #  sig = scaling * sig

        target_len = int(hparams["train_duration"] * config_sample_rate)
        if is_train:
            sig1 = random_segment(sig, target_len)
            #  sig2 = random_segment(sig, target_len)
        else:
            sig1 = center_segment(sig, target_len)
            #  sig2 = center_segment(sig, target_len)
        #  yield sig1
        #  yield sig2
        return sig1

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("caption")
    @sb.utils.data_pipeline.provides("caption", "input_ids", "token_type_ids", "attention_mask")
    def label_pipeline(caption):
        yield caption
        caption_encoded = hparams['txt_tokenizer'](
            caption,
            max_length=hparams['text_max_length'],
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        yield caption_encoded['input_ids'][0]
        yield caption_encoded['token_type_ids'][0]
        yield caption_encoded['attention_mask'][0]

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    ds = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=csv_path,
        dynamic_items=[audio_pipeline, label_pipeline],
        #  dynamic_items=[audio_pipeline],
        #  output_keys=["id", "sig1", "sig2", "caption", "input_ids", "token_type_ids", "attention_mask"],
        output_keys=["id", "sig1", "caption", "input_ids", "token_type_ids", "attention_mask"],
    )

    return ds


if __name__ == '__main__':
    #  prepare_audiocaps_csv(
    #      '/mnt/data3/insper',
    #      '/mnt/data3/insper/insper_audiocaps_index.tsv',
    #      './'
    #  )
    #  prepare_macs_csv(
    #      '/mnt/data3/insper',
    #      '/mnt/data3/insper/insper_macs_index.tsv',
    #      './'
    #  )
    prepare_insper_multi_csv(
        '/mnt/data3/insper',
        './'
    )
