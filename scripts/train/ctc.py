import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch

from utils.data import SR, load_split, read_audio, write_predictions
from utils.device import autocast


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--model", default="facebook/wav2vec2-xls-r-300m")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--max_seconds", type=float, default=15)
    p.add_argument("--mask_time_prob", type=float, default=0.05)
    p.add_argument("--mask_feature_prob", type=float, default=0.0)
    p.add_argument("--layerdrop", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.05, help="hidden/attention/projection dropout")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--train_on_dev", action="store_true", help="add dev to training (final model)")
    p.add_argument("--train_split", default="train", help="training CSV name, e.g. train_sp (speed-perturbed)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def build_processor(texts, out_dir):
    """Character vocab ("|" = word delimiter) + feature extractor, saved to <out_dir>/{vocab.json,processor}."""
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
    vocab = {c: i for i, c in enumerate(sorted(set("".join(texts))))}
    vocab["|"] = vocab.pop(" ")
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    json.dump(vocab, open(out_dir / "vocab.json", "w"), ensure_ascii=False)
    print(len(vocab), "symbols:", "".join(k for k in vocab if len(k) == 1))
    tokenizer = Wav2Vec2CTCTokenizer(str(out_dir / "vocab.json"), unk_token="[UNK]",
                                     pad_token="[PAD]", word_delimiter_token="|")
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tokenizer)
    processor.save_pretrained(out_dir / "processor")
    return processor


@dataclass
class Collator:
    processor: object

    def __call__(self, features):
        batch = self.processor.pad([{"input_values": f["input_values"]} for f in features],
                                   padding="longest", return_tensors="pt")
        lab = self.processor.pad(labels=[{"input_ids": f["labels"]} for f in features],
                                 padding="longest", return_tensors="pt")
        batch["labels"] = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        return batch


def jsonl_logger(path):
    from transformers import TrainerCallback

    class JsonlLogger(TrainerCallback):
        def __init__(self):
            self.f, self.t0 = open(path, "a"), time.time()

        def on_log(self, args_, state, control, logs=None, **kw):
            rec = {"step": state.global_step, "epoch": round(state.epoch or 0, 3),
                   "elapsed_min": round((time.time() - self.t0) / 60, 2), **(logs or {})}
            self.f.write(json.dumps(rec) + "\n")
            self.f.flush()

    return JsonlLogger()


def main(argv=None):
    from datasets import Dataset
    from transformers import AutoModelForCTC, Trainer, TrainingArguments

    args = parser().parse_args(argv)
    out_dir = Path(args.runs_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / "args.json", "w"), indent=2)

    split = lambda name: load_split(args.data_dir, name, durations=False)
    train_df, dev_df, test_df = split(args.train_split), split("dev"), split("test")
    if args.quick:
        train_df, dev_df = train_df.head(256), dev_df.head(64)
    processor = build_processor(split("train")["transcript"].tolist() + split("dev")["transcript"].tolist(), out_dir)

    def prepare(batch):
        audio = read_audio(batch["path"])
        batch["input_values"] = processor(audio, sampling_rate=SR).input_values[0]
        batch["input_length"] = len(audio)
        batch["labels"] = processor(text=batch["transcript"]).input_ids
        return batch

    cols = ["path", "transcript"]
    tr = pd.concat([train_df, dev_df]) if args.train_on_dev else train_df
    train_ds = Dataset.from_pandas(tr[cols].reset_index(drop=True)).map(prepare, remove_columns=cols, num_proc=16)
    dev_ds = Dataset.from_pandas(dev_df[cols].reset_index(drop=True)).map(prepare, remove_columns=cols, num_proc=16)
    n0 = len(train_ds)
    train_ds = train_ds.filter(lambda n: n < args.max_seconds * SR, input_columns="input_length")
    print(f"train {len(train_ds)} (dropped {n0 - len(train_ds)} > {args.max_seconds}s), dev {len(dev_ds)}")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        hyp = processor.batch_decode(pred_ids)
        ref = processor.batch_decode(pred.label_ids, group_tokens=False)
        return {"wer": jiwer.wer(ref, hyp), "cer": jiwer.cer(ref, hyp)}

    model = AutoModelForCTC.from_pretrained(
        args.model,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        attention_dropout=args.dropout, hidden_dropout=args.dropout,
        feat_proj_dropout=args.dropout, final_dropout=args.dropout,
        layerdrop=args.layerdrop,
        mask_time_prob=args.mask_time_prob, mask_time_length=10,
        mask_feature_prob=args.mask_feature_prob, mask_feature_length=64,
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir=str(out_dir / "ckpt"),
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup,
        max_steps=args.max_steps,
        lr_scheduler_type="linear",
        gradient_checkpointing=args.grad_ckpt,
        bf16=torch.cuda.is_available(),
        train_sampling_strategy="group_by_length",
        length_column_name="input_length",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=4,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=dev_ds,
                      processing_class=processor, data_collator=Collator(processor),
                      compute_metrics=compute_metrics, callbacks=[jsonl_logger(out_dir / "metrics.jsonl")])

    ckpts = sorted((out_dir / "ckpt").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)

    trainer.save_model(out_dir / "best")
    processor.save_pretrained(out_dir / "best")
    final = trainer.evaluate()
    print("FINAL DEV", final)
    json.dump(final, open(out_dir / "final_dev.json", "w"), indent=2)

    # ---- greedy test predictions
    model = trainer.model.eval()
    preds = []
    with torch.no_grad():
        for path in test_df["path"]:
            inp = processor(read_audio(path), sampling_rate=SR, return_tensors="pt").to(model.device)
            with autocast(model.device):
                logits = model(**inp).logits
            preds.append(processor.batch_decode(logits.float().argmax(-1))[0])
    test_df["transcript"] = preds
    write_predictions(test_df, out_dir / "test_predictions.csv")
    print("DONE")


if __name__ == "__main__":
    main()
