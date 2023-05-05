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


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    aud_svpth = hparams['aud_emb_path']
    txt_svpth = hparams['txt_emb_path']

    audio_bs = 40000
    text_dataset = hparams['text_dataset']
    audio_dataset = hparams['audio_dataset']
    audio_loader = DataLoader(audio_dataset, batch_size=audio_bs, shuffle=False, num_workers=16)
    txt_emb = torch.from_numpy(np.load(txt_svpth))
    device = torch.device('cuda:0')
    txt_emb = txt_emb.to(device)

    res = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(audio_loader)):
            aud_emb = batch['emb'].to(device)
            similarity = aud_emb @ txt_emb.T
            #  txt_cand = similarity.argmax(-1)
            topk_val, topk_idx = torch.topk(similarity, 5, dim=-1)
            topk_idx = topk_idx.cpu().numpy()
            batch_captions = [[text_dataset[e]['text'] for e in l] for l in topk_idx.tolist()]
            aud_paths = batch['path']
            for i in range(5):
                res += [{
                    'wav_path': p,
                    'caption': c[i]
                } for (p, c) in zip(aud_paths, batch_captions)]
            #  res += [{
            #      #  'wav_path': os.path.basename(p),
            #      'wav_path': p,
            #      'caption_1': c[0],
            #      'caption_2': c[1],
            #      'caption_3': c[2],
            #      'caption_4': c[3],
            #      'caption_5': c[4],
            #  } for (p, c) in zip(aud_paths, batch_captions)]
            #  if i == 3:
            #      break
    out_df = pd.DataFrame(res)
    out_df['ID'] = range(len(out_df))
    #  out_df['num_caps'] = 5
    #  cols = ['ID', 'wav_path', 'caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5', 'num_caps']
    cols = ['ID', 'wav_path', 'caption']
    out_df = out_df[cols]
    out_path = hparams['output_path']
    out_df.to_csv(out_path, index=False)
