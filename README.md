# GEO ASR

## Install

```sh
uv sync
brew install sox # speed perturbation
```

## Usage

```sh
uv run python -m scripts.train.ctc --run_name ctc_baseline
uv run python -m scripts.train.aed --run_name aed_baseline
uv run python -m scripts.decode.ctc_lm --model_dir runs/ctc_baseline/best   # evaluate CTC + 4-gram LM
uv run python -m scripts.decode.aed_beam --run runs/aed_baseline            # evaluate AED (beam search)
```

## Data

| Split | Utterances | Audio | Mean / max length | Words per utterance |
|---|---|---|---|---|
| train | 6 000 | 10.41 h | 6.2 s / 10.6 s | 8.4 |
| dev | 1 000 | 1.48 h | 5.3 s / 10.4 s | 7.6 |
| test | 1 000 | 1.62 h | 5.8 s / 10.6 s | (no transcripts) |

The clips are 16 kHz, 24-bit PCM. Speed perturbation triples the training set to 18 000 utterances (about 31 h).

## Method

### Text and vocabulary
Transcripts are normalised, lower-cased, stripped of punctuation (apostrophes are kept) and have whitespace
collapsed. The CTC vocabulary has 31 symbols: the 28 Esperanto letters (no q, w, x, y), the word delimiter `|`, `[UNK]`, 
and `[PAD]` (which doubles as the CTC blank). AED uses the same characters plus a space, a blank (id 0) and `<sos/eos>`.

### Pretrained encoder
Every model starts from `facebook/wav2vec2-xls-r-300m`: 300M parameters, self-supervised on 128 languages, with no
fine-tuning.

### CTC
Add a linear output layer, fine-tuned end to end. The CNN feature encoder is
frozen. Default settings:
- Training: AdamW, peak learning rate 3e-4, 500 warm-up steps then linear decay; 32 utterances per update
  (batch 16 × 2 accumulation), batches grouped by length; bf16.
- Regularisation: SpecAugment time masking (p = 0.05, span 10 frames), layerdrop 0.05, dropout 0.05.
- Model selection: dev WER, evaluated every 500 steps.

### Word language model and CTC decoding
- LM: a KenLM 4-gram over words.
- Decoding: pyctcdecode beam search (beam 100) with shallow fusion, scoring each hypothesis as
  `log P_ctc + α·log P_lm + β·#words`. The LM's vocabulary is passed as unigrams, which also applies pyctcdecode's
  penalty for unknown words.
- Tuning: α ∈ {0.3, 0.4, 0.5, 0.6, 0.8} × β ∈ {1, 2, 3, 4, 5} is grid-searched on dev with an LM trained on
  the train transcripts only. Test is decoded with the tuned α, β and an LM trained on **train + dev** text.

### Speed perturbation
Each clip gets two extra copies made with sox `speed` at 0.9× and 1.1× (tempo and pitch change together),
resampled back to 16 kHz (Ko et al., 2015). Test is never perturbed.

### Joint CTC/attention encoder-decoder, AED
- Encoder: pretrained XLS-R 300M (`--encoder`), then a linear projection to d_model = 384.
- Decoder: 4 Transformer layers (6 heads, feed-forward 1536, dropout 0.1 -- might be a bit low) over characters.
- CTC head: a second output head on the encoder.
- Loss: 0.7 × label-smoothed cross-entropy (smoothing 0.1) + 0.3 × CTC.
- Optimisation: AdamW with weight decay 0.01, learning rate 5e-5 for the encoder and 1e-3 for the new layers,
  cosine schedule with 1000 warm-up steps, 8000 steps. Batches are dynamic, up to 120 s of audio each, with
  2-step gradient accumulation.
- Decoding: attention beam search (beam 8). The final n-best is rescored as 0.7 × attention + 0.3 × CTC log-likelihood.
