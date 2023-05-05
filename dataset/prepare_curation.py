import os
import random
import re
from tqdm import tqdm
import speechbrain as sb
import torch
import torchaudio
from copy import deepcopy
import pandas as pd
import numpy as np
import pdb


def prepare_curation_csv(
    audio_dir,
    source_csv,
    output_dir,
    output_csv=False,
):
    train_df = pd.read_csv(source_csv)
    num_caps = len(train_df.columns) - 1
    temp_data = []
    cnt = 0
    for idx, row in tqdm(train_df.iterrows()):
        wav_path = os.path.join(audio_dir, row['wav_path'])
        if not os.path.exists(wav_path):
            continue
        dic = {
            'ID': cnt,
            'wav_path': wav_path,
        }
        dic.update({'caption_{}'.format(idx+1): row['caption_{}'.format(idx+1)] for idx in range(num_caps)})
        dic['num_caps'] = num_caps
        temp_data.append(dic)
        cnt += 1
    if output_csv:
        df = pd.DataFrame(temp_data)
        df.to_csv(os.path.join(output_dir, 'train_curation_raw.csv'), index=False)
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
        tags = [e.strip().replace('\n', ' ').replace('\r', '').replace(',', '').replace('$', '').replace('*', '').replace('_', ' ') for e in tags]
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
    print('FSD: Found {} audio files'.format(cnt))
    return temp_data


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
        caption = row[1].replace(',', '').replace('_', ' ').strip()
        temp_data.append({
            'ID': cnt,
            'wav_path': wav_path,
            'caption': caption,
        })
        cnt += 1
    if output_csv:
        df = pd.DataFrame(temp_data)
        df.to_csv(os.path.join(output_dir, 'train_audiocaps_raw.csv'), index=False)
    print('AudioCaps: Found {} audio files'.format(cnt))
    return temp_data


def prepare_macs_csv(
    source_dir,
    source_csv,
    output_dir,
    output_csv=False,
):
    def clean_macs_caption(cap):
        cap = cap.replace(',', '').replace('_', ' ').strip()
        # get rid of labels appending to the end
        #  cap = ' '.join(cap.split(' ')[:-1]).strip()
        #  words = cap.split(' ')
        #  if '_' in words[-1]:
        #      words = words[:-1]
        #  cap = ' '.join(words).strip()
        return cap
    train_df = pd.read_csv(source_csv, sep='\t', header=None)
    temp_data = []
    aud_cnt = 0
    cap_cnt = 0
    for idx, row in tqdm(train_df.iterrows()):
        if row[0][0] == '/':
            fname = row[0][1:]
        else:
            fname = row[0]
        wav_path = os.path.join(source_dir, fname)
        if not os.path.exists(wav_path):
            #  print('file {} not found'.format(fname))
            continue
        captions = row[1].strip().split('\n')
        # filter out the labels, keep only captions
        #  captions = list(set([clean_macs_caption(cap) for cap in captions if len(cap.split(' ')) > 1]))
        captions = list(set([clean_macs_caption(cap) for cap in captions]))
        curr_dics = [{'ID': cap_cnt+idx, 'wav_path': wav_path, 'caption': cap} for idx, cap in enumerate(sorted(captions))]
        cap_cnt += len(curr_dics)
        aud_cnt += 1
        temp_data.extend(curr_dics)
    if output_csv:
        df = pd.DataFrame(temp_data)
        df.to_csv(os.path.join(output_dir, 'train_macs_raw.csv'), index=False)
    print('MACS: Found {} audio files with {} annotations'.format(aud_cnt, cap_cnt))
    return temp_data


def prepare_clotho_csv(
    source_dir,
    source_csv,
    output_dir,
    output_csv=False,
):
    def clean_clotho_caption(cap):
        return cap.replace(',', '').replace('_', ' ').strip()
    train_df = pd.read_csv(source_csv, sep='\t', header=None)
    temp_data = []
    aud_cnt = 0
    cap_cnt = 0
    for idx, row in tqdm(train_df.iterrows()):
        if row[0][0] == '/':
            fname = row[0][1:]
        else:
            fname = row[0]
        wav_path = os.path.join(source_dir, fname)
        if not os.path.exists(wav_path):
            #  print('file {} not found'.format(fname))
            continue
        captions = row[1].strip().split('\n')
        captions = list(set([clean_clotho_caption(cap) for cap in captions]))
        curr_dics = [{'ID': cap_cnt+idx, 'wav_path': wav_path, 'caption': cap} for idx, cap in enumerate(sorted(captions))]
        cap_cnt += len(curr_dics)
        aud_cnt += 1
        temp_data.extend(curr_dics)
    if output_csv:
        df = pd.DataFrame(temp_data)
        df.to_csv(os.path.join(output_dir, 'train_clotho_raw.csv'), index=False)
    print('Clotho: Found {} audio files with {} annotations'.format(aud_cnt, cap_cnt))
    return temp_data


def prepare_misc_multi_csv(
    output_dir,
    prepare_fns,
    valid_pct=0,
    overfit_size=None,
):
    '''
        prepare_fns: partial function objects
    '''
    all_data = []
    for prepare_fn in prepare_fns:
        all_data += prepare_fn()
    # train/valid split
    random.seed(123)
    random.shuffle(all_data)
    if overfit_size is None:
        train_idx = int(len(all_data) * valid_pct)
        train_data = all_data[train_idx:]
        for idx, dic in enumerate(train_data):
            dic['ID'] = idx
        train_df = pd.DataFrame(train_data)
        train_df.to_csv(os.path.join(output_dir, 'train_curation_raw.csv'), index=False)
        if valid_pct > 0:
            valid_data = all_data[:train_idx]
            for idx, dic in enumerate(valid_data):
                dic['ID'] = idx
            valid_df = pd.DataFrame(valid_data)
            valid_df.to_csv(os.path.join(output_dir, 'valid_curation_raw.csv'), index=False)
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
        train_df.to_csv(os.path.join(output_dir, 'train_curation_raw.csv'), index=False)
        valid_data = deepcopy(all_data[:overfit_size])
        for idx, dic in enumerate(valid_data):
            dic['ID'] = idx
        valid_df = pd.DataFrame(valid_data)
        valid_df.to_csv(os.path.join(output_dir, 'valid_curation_raw.csv'), index=False)


def dataio_curation_prep(hparams, csv_path, is_train=True):
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

        #  audio_path = os.path.join(hparams['data_curation_folder'], wav_path)
        #  sig, read_sr = torchaudio.load(audio_path)
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
    #  @sb.utils.data_pipeline.takes("num_caps", "caption_1", "caption_2", "caption_3", "caption_4", "caption_5")
    @sb.utils.data_pipeline.takes("caption")
    @sb.utils.data_pipeline.provides("caption", "input_ids", "token_type_ids", "attention_mask")
    def label_pipeline(caption):
        #  captions = [caption_1, caption_2, caption_3, caption_4, caption_5]
        #  cap_idx = torch.randint(0, int(num_caps), (1,)).item()
        #  caption = captions[cap_idx]
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
        #  output_keys=["id", "sig1", "sig2", "caption", "input_ids", "token_type_ids", "attention_mask"],
        output_keys=["id", "sig1", "caption", "input_ids", "token_type_ids", "attention_mask"],
    )

    return ds


def zs_dataio_prep(hparams, csv_path, label_encoder):
    "Creates the datasets and their data processing pipelines."

    config_sample_rate = hparams["sample_rate"]
    # TODO  use SB implementation but need to make sure it give the same results as PyTorch
    # resampler = sb.processing.speech_augmentation.Resample(orig_freq=latest_file_sr, new_freq=config_sample_rate)
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav_path")
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
        if hparams['eval_duration'] * config_sample_rate >= sig.shape[0]:
            repeat_factor = int(np.ceil((hparams['eval_duration']*config_sample_rate)/sig.shape[0]))
            sig = sig.repeat(repeat_factor)
            sig = sig[0:int(hparams['eval_duration']*config_sample_rate)]
        else:
            start_index = random.randrange(
                sig.shape[0] - int(hparams['eval_duration'] * config_sample_rate)
            )
            sig = sig[start_index:start_index + int(hparams['eval_duration']*config_sample_rate)]
        #  # scaling
        #  max_amp = torch.abs(sig).max().item()
        #  #  assert max_amp > 0
        #  scaling = 1 / max_amp * 0.9
        #  sig = scaling * sig

        return sig

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("class_name")
    @sb.utils.data_pipeline.provides("class_name", "class_string_encoded")
    def label_pipeline(class_name):
        yield class_name
        class_string_encoded = label_encoder.encode_label_torch(class_name)
        yield class_string_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    ds = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=csv_path,
        dynamic_items=[audio_pipeline, label_pipeline],
        output_keys=["id", "sig1", "class_string_encoded"]
    )

    return ds


def test_prep():
    #  data = prepare_macs_csv(
    #      '/mnt/data3/insper',
    #      '/mnt/data3/insper/insper_macs_index.tsv',
    #      './',
    #      False,
    #  )
    data = prepare_clotho_csv(
        '/mnt/data3/insper',
        '/mnt/data3/insper/insper_clotho_index.tsv',
        './',
        False,
    )


if __name__ == '__main__':
    test_prep()
