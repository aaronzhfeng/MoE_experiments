# models/graph2seq_series_rel.py
# Modified for Mixture-of-Experts decoder feed-forward layers
# Author: 2025-07-17 – Option-1 MoE integration

import math
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.chem_utils import ATOM_FDIM, BOND_FDIM

from onmt.decoders.transformer import TransformerDecoder
from onmt.encoders.transformer import TransformerEncoder
from models.graphfeat import GraphFeatEncoder
from onmt.modules import PositionalEncoding, Embeddings
from onmt.utils.misc import sequence_mask
from utils.data_utils import G2SBatch
from onmt.translate import BeamSearch, GNMTGlobalScorer, GreedySearch
import numpy as np
from models.attention_xl import AttnEncoderXL


# ----------------------------------------------------------------------
#               Mixture-of-Experts Feed-Forward sub-layer
# ----------------------------------------------------------------------
class MoEFeedForward(nn.Module):
    r"""
    Position-wise Feed-Forward layer with **top-k sparse routing**.

    Parameters
    ----------
    d_model : int
        Transformer hidden size.
    d_ff : int
        Hidden size of each expert feed-forward MLP (the same as the original FFN).
    num_experts : int
        Total number of experts in this layer.
    topk : int, default 1
        How many experts each token is sent to (top-1 is Switch Transformer style).
    gate_temperature : float, default 1.0
        Temperature for the softmax gating distribution.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        topk: int = 1,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        assert 1 <= topk <= num_experts
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.topk = topk
        self.temperature = gate_temperature

        # ------------- experts -------------
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(d_ff, d_model, bias=True),
                    nn.Dropout(0.1),
                )
                for _ in range(num_experts)
            ]
        )

        # ------------- gating network -------------
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self._gate_probs: Optional[torch.Tensor] = None
        self._gate_logits: Optional[torch.Tensor] = None
        self._topk_idx: Optional[torch.Tensor] = None

    @property
    def last_routing_probs(self) -> Optional[torch.Tensor]:
        return self._gate_probs

    @property
    def last_routing_logits(self) -> Optional[torch.Tensor]:
        return self._gate_logits

    @property
    def last_topk_indices(self) -> Optional[torch.Tensor]:
        return self._topk_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, b, h = x.shape
        x_flat = x.reshape(s * b, h)
        logits = self.gate(x_flat)
        self._gate_logits = logits
        probs = F.softmax(logits / self.temperature, dim=-1)
        self._gate_probs = probs

        if self.topk == 1:
            topk_probs, indices = torch.topk(probs, self.topk, dim=-1)
            self._topk_idx = indices
            all_expert_out = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, h).to(all_expert_out.device)
            y_flat = all_expert_out.gather(1, indices_expanded).squeeze(1)
        else:
            topk_probs, topk_idx = torch.topk(probs, self.topk, dim=-1)
            self._topk_idx = topk_idx
            all_expert_out = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
            tokens = x_flat.size(0)
            gathered = all_expert_out[torch.arange(tokens, device=x_flat.device).unsqueeze(1), topk_idx]
            weighted = gathered * topk_probs.unsqueeze(-1)
            y_flat = weighted.sum(dim=1)

        y = y_flat.reshape(s, b, h)
        return y


class Graph2SeqSeriesRel(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.vocab_size = len(vocab)
        while args.enable_amp and not self.vocab_size % 8 == 0:
            self.vocab_size += 1

        self.encoder = GraphFeatEncoder(
            args,
            n_atom_feat=sum(ATOM_FDIM),
            n_bond_feat=BOND_FDIM
        )

        if args.attn_enc_num_layers > 0:
            self.attention_encoder = AttnEncoderXL(args)
        else:
            self.attention_encoder = None

        self.decoder_embeddings = Embeddings(
            word_vec_size=args.embed_size,
            word_vocab_size=self.vocab_size,
            word_padding_idx=self.vocab["_PAD"],
            position_encoding=True,
            dropout=args.dropout,
        )

        self.decoder = TransformerDecoder(
            num_layers=args.decoder_num_layers,
            d_model=args.decoder_hidden_size,
            heads=args.decoder_attn_heads,
            d_ff=args.decoder_filter_size,
            copy_attn=False,
            self_attn_type="scaled-dot",
            dropout=args.dropout,
            attention_dropout=args.attn_dropout,
            embeddings=self.decoder_embeddings,
            max_relative_positions=args.max_relative_positions,
            aan_useffn=False,
            full_context_alignment=False,
            alignment_layer=-3,
            alignment_heads=0,
        )

        self._inject_moe_into_decoder()

        if not args.attn_enc_hidden_size == args.decoder_hidden_size:
            self.bridge_layer = nn.Linear(args.attn_enc_hidden_size, args.decoder_hidden_size, bias=True)
        else:
            self.bridge_layer = None

        self.output_layer = nn.Linear(args.decoder_hidden_size, self.vocab_size, bias=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab["_PAD"], reduction="mean")

    def _inject_moe_into_decoder(self):
        for layer in self.decoder.transformer_layers:
            d_ff = layer.feed_forward.w_1.weight.size(0)
            layer.feed_forward = MoEFeedForward(
                d_model=self.args.decoder_hidden_size,
                d_ff=d_ff,
                num_experts=self.args.moe_num_experts,
                topk=self.args.moe_topk,
                gate_temperature=self.args.moe_gating_temperature,
            )

    def _get_moe_aux_loss(self):
        load_balancing_loss = 0.0
        router_z_loss = 0.0
        num_ff_layers = 0

        for layer in self.decoder.transformer_layers:
            if isinstance(layer.feed_forward, MoEFeedForward):
                num_ff_layers += 1
                ff_layer = layer.feed_forward
                
                # router_z_loss
                logits = ff_layer.last_routing_logits
                if logits is not None:
                    router_z_loss += torch.mean(torch.logsumexp(logits, dim=-1)**2)
                
                # load balancing loss
                probs = ff_layer.last_routing_probs
                topk_idx = ff_layer.last_topk_indices
                if probs is not None and topk_idx is not None:
                    num_tokens = probs.size(0)
                    num_experts = ff_layer.num_experts
                    
                    topk_one_hot = F.one_hot(topk_idx, num_classes=num_experts).sum(dim=1).float()
                    
                    tokens_per_expert = topk_one_hot.sum(dim=0)
                    fraction_tokens_per_expert = tokens_per_expert / num_tokens
                    
                    probs_of_routed_tokens = (probs * topk_one_hot).sum(dim=0)
                    mean_probs_for_routed_tokens = probs_of_routed_tokens / (tokens_per_expert + 1e-8)
                    
                    l_aux = num_experts * (fraction_tokens_per_expert * mean_probs_for_routed_tokens).sum()
                    load_balancing_loss += l_aux
        
        if num_ff_layers > 0:
            load_balancing_loss /= num_ff_layers
            router_z_loss /= num_ff_layers

        z_loss_factor = 1e-2
        
        total_aux_loss = self.args.moe_aux_loss_factor * load_balancing_loss + z_loss_factor * router_z_loss
        return total_aux_loss

    def encode_and_reshape(self, reaction_batch: G2SBatch):
        hatom, _ = self.encoder(reaction_batch)

        atom_scope = reaction_batch.atom_scope
        memory_lengths = [scope[-1][0] + scope[-1][1] - scope[0][0] for scope in atom_scope]
        assert 1 + sum(memory_lengths) == hatom.size(0), "Memory lengths calculation error"

        memory_bank = torch.split(hatom, [1] + memory_lengths, dim=0)
        max_length = max(memory_lengths) if memory_lengths else 0
        
        padded_memory_bank = []
        for length, h in zip(memory_lengths, memory_bank[1:]):
            m = nn.ZeroPad2d((0, 0, 0, max_length - length))
            padded_memory_bank.append(m(h))

        padded_memory_bank = torch.stack(padded_memory_bank, dim=1)
        memory_lengths = torch.tensor(memory_lengths, dtype=torch.long, device=padded_memory_bank.device)

        if self.attention_encoder is not None:
            padded_memory_bank = self.attention_encoder(
                padded_memory_bank,
                memory_lengths,
                reaction_batch.distances
            )

        if self.bridge_layer is not None:
            padded_memory_bank = self.bridge_layer(padded_memory_bank)
        
        self.decoder.state["src"] = np.zeros(max_length)
        return padded_memory_bank, memory_lengths

    def forward(self, reaction_batch: G2SBatch):
        padded_memory_bank, memory_lengths = self.encode_and_reshape(reaction_batch)

        dec_in = reaction_batch.tgt_token_ids[:, :-1]
        m = nn.ConstantPad1d((1, 0), self.vocab["_SOS"])
        dec_in = m(dec_in)
        dec_in = dec_in.transpose(0, 1).unsqueeze(-1)

        dec_outs, _ = self.decoder(
            tgt=dec_in,
            memory_bank=padded_memory_bank,
            memory_lengths=memory_lengths
        )

        dec_outs = self.output_layer(dec_outs)
        dec_outs = dec_outs.permute(1, 2, 0)
        
        main_loss = self.criterion(input=dec_outs, target=reaction_batch.tgt_token_ids)
        aux_loss = self._get_moe_aux_loss()
        loss = main_loss + aux_loss

        predictions = torch.argmax(dec_outs, dim=1)
        mask = (reaction_batch.tgt_token_ids != self.vocab["_PAD"]).long()
        accs = (predictions == reaction_batch.tgt_token_ids).float() * mask
        acc = accs.sum() / mask.sum()

        return loss, acc

    def predict_step(self, reaction_batch: G2SBatch,
                     batch_size: int, beam_size: int, n_best: int, temperature: float,
                     min_length: int, max_length: int) -> Dict[str, Any]:
        if beam_size == 1:
            decode_strategy = GreedySearch(
                pad=self.vocab["_PAD"], bos=self.vocab["_SOS"], eos=self.vocab["_EOS"],
                batch_size=batch_size, min_length=min_length, max_length=max_length,
                block_ngram_repeat=0, exclusion_tokens=set(), return_attention=False,
                sampling_temp=0.0, keep_topk=1
            )
        else:
            global_scorer = GNMTGlobalScorer(alpha=0.0, beta=0.0, length_penalty="none", coverage_penalty="none")
            decode_strategy = BeamSearch(
                beam_size=beam_size, batch_size=batch_size, pad=self.vocab["_PAD"],
                bos=self.vocab["_SOS"], eos=self.vocab["_EOS"], n_best=n_best,
                global_scorer=global_scorer, min_length=min_length, max_length=max_length,
                return_attention=False, block_ngram_repeat=0, exclusion_tokens=set(),
                stepwise_penalty=None, ratio=0.0
            )

        padded_memory_bank, memory_lengths = self.encode_and_reshape(reaction_batch=reaction_batch)
        results = {"predictions": None, "scores": None, "attention": None}

        src_map = None
        target_prefix = None
        fn_map_state, memory_bank, memory_lengths, src_map = decode_strategy.initialize(
            memory_bank=padded_memory_bank, src_lengths=memory_lengths,
            src_map=src_map, target_prefix=target_prefix
        )

        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
            dec_out, dec_attn = self.decoder(
                tgt=decoder_input, memory_bank=memory_bank,
                memory_lengths=memory_lengths, step=step
            )
            attn = dec_attn.get("std")
            dec_out = self.output_layer(dec_out) / temperature
            log_probs = F.log_softmax(dec_out.squeeze(0), dim=-1)

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break
                
                select_indices = decode_strategy.select_indices
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)
                self.map_state(lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        results["alignment"] = [[] for _ in range(batch_size)]

        return results

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)
        if self.decoder.state["cache"] is not None:
            _recursive_map(self.decoder.state["cache"])