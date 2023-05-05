import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import sys
import pdb


def register_txt_emb(hparams, device):
    # load pretrained weights
    if hparams['clap_ckpt_path'] is not None:
         model_state_dict = torch.load(hparams['clap_ckpt_path'], map_location=torch.device('cpu'))['model']
         hparams['clap'].load_state_dict(model_state_dict)
         print('==> Loading pretrained CLAP checkpoint from {}'.format(hparams['clap_ckpt_path']))
    if hparams['ssl_checkpointer'] is not None:
        if hparams['use_maxacc_ep']:
            hparams['ssl_checkpointer'].recover_if_possible(max_key='acc')
        elif hparams['use_minloss_ep']:
            hparams['ssl_checkpointer'].recover_if_possible(min_key='loss')
        else:
            #  hparams['ssl_checkpointer'].recover_if_possible()
            chosen_ckpts = hparams['ssl_checkpointer'].find_checkpoints()
            chosen_ckpt = chosen_ckpts[0]
            hparams['ssl_checkpointer'].load_checkpoint(chosen_ckpt)

    # get text prompt
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.load_or_create(hparams['label_encoder_path'])
    hparams["label_encoder"] = label_encoder

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    text_input = [hparams['text_prompt'] + e for e in class_labels]

    # get embeddings
    hparams['clap'] = hparams['clap'].to(device)
    hparams['clap'].caption_encoder.eval()
    with torch.no_grad():
        txt_encoding = hparams['txt_tokenizer'](
            text_input,
            max_length=hparams['text_max_length'],
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        ).to(device)
        txt_emb = hparams['clap'].caption_encoder(txt_encoding)
        txt_emb = txt_emb / torch.norm(txt_emb, dim=-1, keepdim=True)
        # saving as half precision
        txt_emb = txt_emb.half()

    return text_input, txt_emb


if __name__ == "__main__":
    """
        this script performs unsupervised curation by measuring the text
        similarity between two datasets (source and target)

        the source dataset should contain a set of labels

        the target dataset contains audio-caption pairs

        it keeps the examples from the target dataset that are relevant to the
        source dataset
    """
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # getting source label data
    device = torch.device('cuda:0')
    src_text, src_txt_emb = register_txt_emb(hparams, device)

    # getting target data
    target_bs = 100
    target_dataset = hparams['target_dataset']
    #  target_loader = DataLoader(target_dataset, batch_size=target_bs, shuffle=False, num_workers=16)
    target_txt_emb = torch.from_numpy(np.load(hparams['target_emb_path'])).to(device)

    res = []
    # plan 1: first find max for each target, then filter by threshold
    # plan 2: first find max for each target, then argsort
    # plan 3: find the topk target for each source
    with torch.no_grad():
        similarity = src_txt_emb @ target_txt_emb.T
        # for each target text, find its most relevant source text
        max_sim = similarity.amax(dim=0)
        keep_idx = torch.nonzero(max_sim >= hparams['sim_threshold']).cpu().numpy()

    # saving filtered embeddings
    target_txt_emb = target_txt_emb.detach().half().cpu().numpy()
    target_save_emb = target_txt_emb[keep_idx.flatten()]
    np.save(hparams['output_emb_path'], target_save_emb)

    # saving filtered df
    out_df = target_dataset.df.iloc[keep_idx.flatten().tolist()]
    # wav path contains absolute paths
    #  out_df['wav_path'] = out_df['wav_path'].map(lambda p: p[len(hparams['target_root_dir']):])
    out_df['ID'] = range(len(out_df))
    cols = ['ID', 'wav_path', 'caption']

    out_df = out_df[cols]
    out_path = hparams['output_csv_path']
    out_df.to_csv(out_path, index=False)
