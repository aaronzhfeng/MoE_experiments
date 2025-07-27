#!/usr/bin/env python
import argparse
import glob
import logging
import numpy as np
import os
import sys
import time
import torch
from torch.utils.data import DataLoader

from models.graph2seq_series_rel import Graph2SeqSeriesRel
from models.seq2seq import Seq2Seq
from utils import parsing
from utils.data_utils import canonicalize_smiles, load_vocab, S2SDataset, G2SDataset
from utils.train_utils import param_count, set_seed, setup_logger

def get_validate_parser():
    parser = argparse.ArgumentParser("validate")
    parsing.add_common_args(parser)
    parsing.add_preprocess_args(parser)
    parsing.add_train_args(parser)
    parsing.add_predict_args(parser)
    return parser

def main(args):
    start_time = time.time()
    parsing.log_args(args)

    # make sure results directory exists
    os.makedirs(os.path.join("./results", args.data_name), exist_ok=True)

    # pick up all checkpoint files in the range
    checkpoint_paths = glob.glob(os.path.join(args.load_from, "*.pt"))
    checkpoint_paths = sorted(
        checkpoint_paths,
        key=lambda p: int(p.split(".")[-2].split("_")[-1]),
        reverse=True
    )
    # filter by step number embedded in filename
    def step_num(p):
        return int(os.path.basename(p).split(".")[-2].split("_")[0])
    checkpoint_paths = [
        p for p in checkpoint_paths
        if args.checkpoint_step_start <= step_num(p) <= args.checkpoint_step_end
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    vocab_tokens = None
    smis_tgt = []

    # load all target SMILES once
    with open(args.val_tgt, "r") as f:
        for line in f:
            sm = "".join(line.split())
            smis_tgt.append(canonicalize_smiles(sm))

    total = len(smis_tgt)

    for idx, ckpt in enumerate(checkpoint_paths):
        logging.info(f"==== CHECKPOINT {idx} → {ckpt} ====")
        state = torch.load(ckpt, map_location=device)
        pre_args = state["args"]
        state_dict = state["state_dict"]

        # build model once
        if model is None:
            # propagate mpn_type and rel_pos if missing
            for attr in ("mpn_type", "rel_pos"):
                if not hasattr(pre_args, attr):
                    setattr(pre_args, attr, getattr(args, attr))

            assert args.model == pre_args.model, \
                f"Requested {args.model} but checkpoint was {pre_args.model}"
            if args.model == "s2s":
                model_cls, dataset_cls = Seq2Seq, S2SDataset
            elif args.model == "g2s_series_rel":
                model_cls, dataset_cls = Graph2SeqSeriesRel, G2SDataset
                pre_args.compute_graph_distance = True
            else:
                raise ValueError(f"Unknown model type: {args.model}")

            # vocab & model init
            vocab = load_vocab(pre_args.vocab_file)
            vocab_tokens = [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]
            model = model_cls(pre_args, vocab)
            logging.info(model)
            logging.info(f"Total parameters: {param_count(model)}")

            # prepare validation dataset & loader
            val_dataset = dataset_cls(pre_args, file=args.valid_bin)
            val_dataset.batch(
                batch_type=args.batch_type,
                batch_size=args.predict_batch_size
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=lambda batch: batch[0],
                pin_memory=True
            )

        # load weights (ignore unexpected keys from relative-PE)
        model.load_state_dict(state_dict, strict=False)
        logging.info("Loaded pretrained weights (strict=False)")

        model.to(device)
        model.eval()

        # run beam search predictions
        all_preds = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i % args.log_iter == 0:
                    logging.info(f"Inference on batch {i}/{len(val_loader)}, "
                                 f"elapsed {time.time() - start_time:.1f}s")
                batch.to(device)
                out = model.predict_step(
                    reaction_batch=batch,
                    batch_size=batch.size,
                    beam_size=args.beam_size,
                    n_best=args.n_best,
                    temperature=args.temperature,
                    min_length=args.predict_min_len,
                    max_length=args.predict_max_len
                )
                # collect n-best
                for preds in out["predictions"]:
                    toks = []
                    for p in preds:
                        idxs = p.detach().cpu().numpy()
                        seq = " ".join(vocab_tokens[i] for i in idxs[:-1])
                        toks.append(seq)
                    all_preds.append(",".join(toks) + "\n")

        # write raw predictions
        res_file = f"{args.result_file}.{idx}"
        with open(res_file, "w") as fo:
            fo.writelines(all_preds)

        # scoring: top-n accuracy
        invalid_count = 0
        acc_matrix = np.zeros((total, args.n_best), dtype=np.float32)

        with open(res_file, "r") as fo:
            for i, (gold, pred_line) in enumerate(zip(smis_tgt, fo)):
                # skip problematic CC entries
                if gold == "CC":
                    continue
                preds = "".join(pred_line.split()).split(",")
                preds = [
                    canonicalize_smiles(p, trim=False, suppress_warning=True)
                    for p in preds
                ]
                if not preds[0]:
                    invalid_count += 1
                preds = [p for p in preds if p and p != "CC"]
                for j, p in enumerate(preds):
                    if p == gold:
                        acc_matrix[i, j:] = 1.0
                        break

        # write stats
        stat_file = f"{args.result_file}.stat.{idx}"
        with open(stat_file, "w") as fo:
            top1_inv = invalid_count / total * 100
            fo.write(f"Total: {total}, top-1 invalid: {top1_inv:.2f} %\n")
            mean_accs = acc_matrix.mean(axis=0)
            for n in range(args.n_best):
                fo.write(f"Top {n+1} accuracy: {mean_accs[n]*100:.2f} %\n")

        logging.info(f"Finished checkpoint {idx}: stats in {stat_file}")

if __name__ == "__main__":
    parser = get_validate_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    setup_logger(args, warning_off=True)
    torch.set_printoptions(profile="full")
    main(args)

