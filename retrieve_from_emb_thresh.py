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

    audio_bs = hparams['audio_bs']
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
            # find audio-text pairs with similarity higher than threshold
            keep_idx = torch.nonzero(similarity >= hparams['sim_threshold']).cpu().numpy()
            aud_paths = np.array(batch['path'])[keep_idx[:, 0]].tolist()
            captions = text_dataset[keep_idx[:, 1]]['text'].tolist()
            res += [{'wav_path': p, 'caption': c} for p, c in zip(aud_paths, captions)]
    # dump to csv
    out_df = pd.DataFrame(res)
    out_df['ID'] = range(len(out_df))
    cols = ['ID', 'wav_path', 'caption']
    out_df = out_df[cols]
    out_path = hparams['output_path']
    out_df.to_csv(out_path, index=False)
