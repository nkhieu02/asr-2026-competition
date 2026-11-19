import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class XLSREncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        kw = dict(mask_time_prob=args.mask_time_prob, mask_time_length=10,
                  mask_feature_prob=args.mask_feature_prob, mask_feature_length=64,
                  layerdrop=args.layerdrop, attention_dropout=args.enc_dropout,
                  hidden_dropout=args.enc_dropout, feat_proj_dropout=args.enc_dropout)
        cfg = getattr(args, "encoder_config", None)
        if cfg is not None:
            for k, v in kw.items():
                setattr(cfg, k, v)
            self.w2v = Wav2Vec2Model(cfg)
        else:
            self.w2v = Wav2Vec2Model.from_pretrained(args.encoder, **kw)
        self.w2v.freeze_feature_encoder()
        self.dim = self.w2v.config.hidden_size

    def forward(self, x, mask):
        h = self.w2v(x, attention_mask=mask).last_hidden_state
        hlen = self.w2v._get_feat_extract_output_lengths(mask.sum(-1)).long()
        return h, hlen


def lengths_to_padmask(lens, T):
    return torch.arange(T, device=lens.device)[None, :] >= lens[:, None]
