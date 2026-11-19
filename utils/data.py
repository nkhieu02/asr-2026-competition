import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

SR = 16_000


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s']", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_split(data_dir, name, durations=True):
    """data/geo/<name>.csv -> DataFrame with file, path, normalised transcript (and dur in seconds)."""
    df = pd.read_csv(Path(data_dir) / "geo" / f"{name}.csv", keep_default_na=False)
    df["path"] = df["file"].map(lambda f: str(Path(data_dir) / f))
    df["transcript"] = df["transcript"].map(normalize)
    if durations:
        df["dur"] = df["path"].map(lambda p: sf.info(p).frames / SR)
    return df


def read_audio(path):
    """Mono float32 waveform."""
    a, _ = sf.read(path, dtype="float32")
    return a.mean(1) if a.ndim > 1 else a


def write_predictions(df, path):
    """Competition submission format: file,transcript."""
    df[["file", "transcript"]].to_csv(path, index=False)


class CharVocab:
    """id 0 = CTC blank and padding; 1..N = characters (incl. space); N+1 = <sos/eos> (AED)."""

    def __init__(self, texts):
        self.chars = sorted(set("".join(texts)))
        self.c2i = {c: i + 1 for i, c in enumerate(self.chars)}
        self.blank = 0
        self.sos_eos = len(self.chars) + 1
        self.size = len(self.chars) + 2

    def encode(self, s):
        return [self.c2i[c] for c in s if c in self.c2i]

    def decode(self, ids):
        return re.sub(r"\s+", " ", "".join(self.chars[i - 1] for i in ids if 0 < i <= len(self.chars))).strip()

    def to_json(self, path):
        json.dump({"chars": self.chars}, open(path, "w"), ensure_ascii=False)

    @classmethod
    def from_json(cls, path):
        return cls(["".join(json.load(open(path))["chars"])])


class AudioSet(torch.utils.data.Dataset):
    def __init__(self, df, vocab):
        self.paths, self.texts, self.durs = df["path"].tolist(), df["transcript"].tolist(), df["dur"].tolist()
        self.vocab = vocab

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        a = read_audio(self.paths[i])
        a = (a - a.mean()) / np.sqrt(a.var() + 1e-7)  # same as Wav2Vec2FeatureExtractor(do_normalize=True)
        return torch.from_numpy(a), torch.tensor(self.vocab.encode(self.texts[i]), dtype=torch.long), i


def collate(items):
    wavs, labs, idx = zip(*items)
    B, L = len(wavs), max(len(w) for w in wavs)
    x = torch.zeros(B, L)
    mask = torch.zeros(B, L, dtype=torch.long)
    for b, w in enumerate(wavs):
        x[b, : len(w)] = w
        mask[b, : len(w)] = 1
    U = max(1, max(len(l) for l in labs))
    y = torch.zeros(B, U, dtype=torch.long)
    for b, l in enumerate(labs):
        y[b, : len(l)] = l
    return {"x": x, "mask": mask, "y": y, "ylen": torch.tensor([len(l) for l in labs]), "idx": list(idx)}


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


class DurationBatchSampler(torch.utils.data.Sampler):
    def __init__(self, durs, max_sec, shuffle, seed=0):
        self.durs, self.max_sec, self.shuffle, self.epoch, self.seed = np.asarray(durs), max_sec, shuffle, 0, seed

    def _batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.argsort(self.durs + (rng.uniform(0, 0.5, len(self.durs)) if self.shuffle else 0))
        batches, cur, cur_max = [], [], 0.0
        for i in order:
            m = max(cur_max, self.durs[i])
            if cur and m * (len(cur) + 1) > self.max_sec:
                batches.append(cur)
                cur, m = [], self.durs[i]
            cur.append(int(i))
            cur_max = m
        if cur:
            batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        b = self._batches()
        self.epoch += 1
        return iter(b)

    def __len__(self):
        return len(self._batches())


def loader(df, vocab, max_sec, shuffle, seed=0):
    ds = AudioSet(df, vocab)
    return torch.utils.data.DataLoader(ds, batch_sampler=DurationBatchSampler(ds.durs, max_sec, shuffle, seed),
                                       collate_fn=collate, num_workers=4, persistent_workers=True)
