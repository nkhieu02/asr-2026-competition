import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import XLSREncoder, lengths_to_padmask


class PosEnc(nn.Module):
    def __init__(self, d, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class AED(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.vocab, self.args = vocab, args
        V, d = vocab.size, args.d_model
        self.encoder = XLSREncoder(args)
        self.enc_proj = nn.Sequential(nn.Linear(self.encoder.dim, d), nn.LayerNorm(d), nn.Dropout(args.dec_dropout))
        self.ctc_head = nn.Linear(self.encoder.dim, V)
        self.emb = nn.Embedding(V, d)
        self.pos = PosEnc(d)
        layer = nn.TransformerDecoderLayer(d, args.nhead, args.d_ff, args.dec_dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, args.dec_layers, norm=nn.LayerNorm(d))
        self.out = nn.Linear(d, V)
        self.drop = nn.Dropout(args.dec_dropout)

    def encode(self, batch):
        h, hlen = self.encoder(batch["x"], batch["mask"])
        return h, hlen, self.enc_proj(h), lengths_to_padmask(hlen, h.size(1))

    def dec_logits(self, ys_in, mem, mem_pad):
        L = ys_in.size(1)
        causal = torch.triu(torch.full((L, L), float("-inf"), device=ys_in.device), 1)
        x = self.drop(self.pos(self.emb(ys_in) * math.sqrt(self.args.d_model)))
        return self.out(self.decoder(x, mem, tgt_mask=causal, memory_key_padding_mask=mem_pad))

    def ctc_logp(self, h):
        return self.ctc_head(h).float().log_softmax(-1)

    def forward(self, batch):
        h, hlen, mem, mem_pad = self.encode(batch)
        y, ylen = batch["y"], batch["ylen"]
        B, U = y.shape
        sos = torch.full((B, 1), self.vocab.sos_eos, device=y.device)
        ys_in = torch.cat([sos, y], 1)  # padded positions are id 0 -> only seen after eos, ignored in loss
        ys_out = torch.cat([y, torch.zeros_like(sos)], 1)
        ys_out[torch.arange(B), ylen] = self.vocab.sos_eos
        ys_out = ys_out.masked_fill(torch.arange(U + 1, device=y.device)[None] > ylen[:, None], -100)
        logits = self.dec_logits(ys_in, mem, mem_pad).float()
        att = F.cross_entropy(logits.transpose(1, 2), ys_out, ignore_index=-100, label_smoothing=self.args.label_smoothing)
        acc = ((logits.argmax(-1) == ys_out) & (ys_out != -100)).sum() / (ys_out != -100).sum()
        ctc = F.ctc_loss(self.ctc_logp(h).transpose(0, 1), y, hlen, ylen, blank=0,
                         reduction="mean", zero_infinity=True)
        w = self.args.ctc_weight
        return {"loss": (1 - w) * att + w * ctc, "att_loss": att.detach(), "ctc_loss": ctc.detach(), "acc": acc.detach()}

    @torch.no_grad()
    def decode(self, batch, beam=1):
        h, hlen, mem, mem_pad = self.encode(batch)
        if beam == 1:
            return self._greedy(mem, mem_pad, hlen)
        ctc_lp = self.ctc_logp(h)
        return [self._beam(mem[b: b + 1, : hlen[b]], ctc_lp[b, : hlen[b]], beam) for b in range(h.size(0))]

    def _max_len(self, T):
        return int(T * self.args.max_len_ratio) + 10

    def _greedy(self, mem, mem_pad, hlen):
        B, eos = mem.size(0), self.vocab.sos_eos
        ys = torch.full((B, 1), eos, device=mem.device)
        done = torch.zeros(B, dtype=torch.bool, device=mem.device)
        for _ in range(self._max_len(int(hlen.max()))):
            lg = self.dec_logits(ys, mem, mem_pad)[:, -1]
            lg[:, 0] = float("-inf")
            nxt = lg.argmax(-1)
            nxt = nxt.masked_fill(done, eos)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == eos
            if done.all():
                break
        out = []
        for row in ys[:, 1:].tolist():
            out.append(self.vocab.decode(row[: row.index(eos)] if eos in row else row))
        return out

    def _search(self, mem, K):
        """Attention beam search -> finished [(tokens, attention log-prob)].

        Each step keeps only the K best extensions; an <eos> among them finishes that hypothesis and uses
        up its slot. (Accepting <eos> from outside the top K, or stopping once K hypotheses finished,
        returns truncated outputs.) The search ends when no hypothesis is alive, or when the K best finished
        ones all out-score every alive one (scores only decrease), or at max length.
        """
        eos, dev = self.vocab.sos_eos, mem.device
        hyps = torch.full((1, 1), eos, device=dev)
        scores = torch.zeros(1, device=dev)
        finished = []
        for _ in range(self._max_len(mem.size(1))):
            lp = self.dec_logits(hyps, mem.expand(hyps.size(0), -1, -1), None)[:, -1].float().log_softmax(-1)
            lp[:, 0] = -1e9
            cand = (scores[:, None] + lp).view(-1)
            top, idx = cand.topk(min(K, cand.numel()))
            new_h, new_s = [], []
            for s, i in zip(top.tolist(), idx.tolist()):
                src, tok = divmod(i, lp.size(1))
                if tok == eos:
                    finished.append((hyps[src, 1:].tolist(), s))
                else:
                    new_h.append(torch.cat([hyps[src], torch.tensor([tok], device=dev)]))
                    new_s.append(s)
            if not new_h:
                break
            hyps, scores = torch.stack(new_h), torch.tensor(new_s, device=dev)
            if len(finished) >= K and sorted(f[1] for f in finished)[-K] > max(new_s):
                break
        else:
            lp = self.dec_logits(hyps, mem.expand(hyps.size(0), -1, -1), None)[:, -1].float().log_softmax(-1)
            finished += [(hyps[i, 1:].tolist(), float(scores[i] + lp[i, eos])) for i in range(hyps.size(0))]
        return finished

    def nbest(self, mem, ctc_lp, K):
        """Beam search n-best as [(tokens, attention log-prob, CTC log-prob)] (for external-LM rescoring)."""
        T, out = ctc_lp.size(0), []
        for toks, att_s in self._search(mem, K):
            ctc_s = -1e9
            if toks:
                tgt = torch.tensor([toks], device=mem.device)
                nll = F.ctc_loss(ctc_lp[:, None], tgt, torch.tensor([T]), torch.tensor([len(toks)]),
                                 blank=0, reduction="sum").item()
                ctc_s = -nll if math.isfinite(nll) else -1e9
            out.append((toks, att_s, ctc_s))
        return out

    @torch.no_grad()
    def nbest_batch(self, batch, K):
        h, hlen, mem, _ = self.encode(batch)
        ctc = self.ctc_logp(h)
        return [self.nbest(mem[b: b + 1, : hlen[b]], ctc[b, : hlen[b]], K) for b in range(h.size(0))]

    def _beam(self, mem, ctc_lp, K):
        lam = self.args.ctc_weight_decode
        best = max(self.nbest(mem, ctc_lp, K),
                   key=lambda h: (1 - lam) * h[1] + lam * h[2] + self.args.length_bonus * len(h[0]))
        return self.vocab.decode(best[0])
