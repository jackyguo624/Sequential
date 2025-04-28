import logging
from pathlib import Path
from typing import Callable, Union

import torch
import pytorch_lightning as pl
from models.asr.fireredasr.models.module.adapter import Adapter
from models.asr.fireredasr.models.module.conformer_encoder import ConformerEncoder
from models.asr.fireredasr.tokenizer.llm_tokenizer import (
    LlmTokenizerWrapper,
    DEFAULT_SPEECH_TOKEN,
    IGNORE_TOKEN_ID
)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FireRedAsrLlm(pl.LightningModule):
    """
    FireRed ASR with LLM
    Args:
        encoder: ConformerEncoder
        llm: Any
        encoder_projector: Adapter
        tokenizer: AutoTokenizer
        freeze_encoder: bool
        freeze_llm: bool
        use_lora: bool
    """
    def __init__(self,
                 encoder: ConformerEncoder,
                 llm: Callable,
                 encoder_projector: Adapter, #  Callable[[int], Adapter],
                 tokenizer: Callable,
                 freeze_encoder: bool,
                 freeze_llm: bool,
                 use_lora: bool,
                 max_len: int = 4096,
                 model_path: Union[str, Path] = None,
                 ):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector # (llm.config.hidden_size)
        self.tokenizer = tokenizer

        # args
        self.freeze_encoder = freeze_encoder
        self.freeze_llm = freeze_llm
        self.use_lora = use_lora
        self.max_len = max_len
        self.model_path = model_path

        if self.freeze_encoder:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
            self.encoder.eval()

        # LLM Freeze or LoRA
        if self.freeze_llm:
            logging.info(f"Frezee LLM")
            for name, param in self.llm.named_parameters():
                param.requires_grad = False
            self.llm.eval()
        else:
            if self.use_lora:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=64,
                    lora_alpha=16,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "up_proj",
                        "gate_proj",
                        "down_proj",
                    ],
                    lora_dropout=0.05,
                    task_type="CAUSAL_LM",
                )
                self.llm = get_peft_model(self.llm, lora_config)
                self.llm.print_trainable_parameters()

        if self.model_path:
            logging.info(f"Loading model from {self.model_path}")
            package = torch.load(self.model_path, map_location=lambda storage, loc: storage)
            self.load_state_dict(package["model_state_dict"], strict=False)

        # Tokenizer
        assert self.tokenizer.pad_token_id == self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.config.bos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.llm.config.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.llm.config.default_speech_token_id = self.tokenizer.convert_tokens_to_ids(
            DEFAULT_SPEECH_TOKEN
        )


        self.save_hyperparameters()

    def forward(self, batch, batch_idx):

        import numpy as np

        device = next(self.encoder.parameters()).device
        dtype = next(self.encoder.parameters()).dtype

        pad_feats, supervisions = batch['inputs'], batch['supervisions']

        feat_lengths = supervisions['num_frames']
        pad_feats = pad_feats.to(device).to(dtype)

        encoder_outs, enc_lengths, enc_mask = self.encoder(pad_feats, feat_lengths)

        speech_features, speech_lens = self.encoder_projector(encoder_outs, enc_lengths)

        '''
        # Tokenizer processing
        padded_input_ids, attention_mask, _, _ = LlmTokenizerWrapper.preprocess_texts(
            [""] * len(supervisions['text']),
            self.tokenizer,
            self.max_len,
            decode=True
            )
        padded_input_ids = padded_input_ids.to(device)
        attention_mask = attention_mask.to(device)


        generated_ids = self.transcribe(
                pad_feats, feat_lengths, padded_input_ids, attention_mask,
                1,
                0,
                0,
                1.0,
                0.0,
                1.0
            )
        decoded_texts = self.tokenizer.batch_decode(generated_ids,
                                                skip_special_tokens=True)
        print(f"decoded texts: {decoded_texts}")
        breakpoint()
        '''

        text = supervisions['text']
        padded_input_ids, attention_mask, target_ids, _ = LlmTokenizerWrapper.preprocess_texts(text,
            self.tokenizer,
            self.max_len,
            decode=False
            )

        padded_input_ids = padded_input_ids.to(device)
        attention_mask = attention_mask.to(device)

        target_ids = target_ids.to(device)
        inputs_embeds = self.llm.get_input_embeddings()(padded_input_ids)

        # Merge speech features and text
        inputs_embeds, attention_mask, labels = \
            self._merge_input_ids_with_speech_features(
                speech_features.to(inputs_embeds.dtype),
                inputs_embeds,
                padded_input_ids,
                attention_mask,
                labels=target_ids,
                speech_lens=speech_lens
            )
        # LLM forward
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Loss and accuracy
        loss = outputs.loss
        acc = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            mask = labels != IGNORE_TOKEN_ID
            preds = torch.argmax(outputs.logits, -1)
            acc = torch.sum(preds[mask] == labels[mask]) / len(labels[mask])

        print(f"loss: {loss} acc: {acc}")

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer

    def transcribe(self,
                  padded_feat: torch.Tensor,
                  feat_lengths: torch.Tensor,
                  padded_input_ids: torch.Tensor,
                  attention_mask: torch.Tensor,
                  beam_size: int = 1,
                  decode_max_len: int = 0,
                  decode_min_len: int = 0,
                  repetition_penalty: float = 1.0,
                  llm_length_penalty: float = 1.0,
                  temperature: float = 1.0):

        encoder_outs, enc_lengths, enc_mask = self.encoder(padded_feat, feat_lengths)
        speech_features, speech_lens = self.encoder_projector(encoder_outs, enc_lengths)
        inputs_embeds = self.llm.get_input_embeddings()(padded_input_ids)

        inputs_embeds, attention_mask, _ = \
            self._merge_input_ids_with_speech_features(
                speech_features.to(inputs_embeds.dtype), inputs_embeds, padded_input_ids, attention_mask,
                speech_lens=speech_lens
            )

        max_new_tokens = speech_features.size(1) if decode_max_len < 1 else decode_max_len
        max_new_tokens = max(1, max_new_tokens)

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            do_sample=False,
            min_length=decode_min_len,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            length_penalty=llm_length_penalty,
            temperature=temperature,
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.llm.config.pad_token_id,
        )

        return generated_ids


    def _merge_input_ids_with_speech_features(
        self,
        speech_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        speech_lens: torch.Tensor = None
    ):
        """
        Modified from: https://github.com/k2-fsa/icefall/blob/master/egs/speech_llm/ASR_LLM/whisper_llm_zh/model.py
        """
        speech_lens = None
        num_speechs, speech_len, embed_dim = speech_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(self.llm.config.pad_token_id)
        )
        # 1. Create a mask to know where special speech tokens are
        special_speech_token_mask = input_ids == self.llm.config.default_speech_token_id
        num_special_speech_tokens = torch.sum(special_speech_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (
            num_special_speech_tokens.max() * (speech_len - 1)
        ) + sequence_length
        batch_indices, non_speech_indices = torch.where(
            input_ids != self.llm.config.default_speech_token_id
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged speech-text sequence.
        # `special_speech_token_mask` identifies speech tokens. Each speech token will be replaced by `nb_text_tokens_per_speechs - 1` text tokens.
        # `torch.cumsum` computes how each speech token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (
            torch.cumsum((special_speech_token_mask * (speech_len - 1) + 1), -1) - 1
        )  # (N,U)
        nb_speech_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_speech_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_speech_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                IGNORE_TOKEN_ID,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_speech_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_speech_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<speech>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the speech features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_speech_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_speech_indices
        ]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_speech_indices
            ]

        # 5. Fill the embeddings corresponding to the speechs. Anything that is not `text_positions` needs filling (#29835)
        speech_to_overwrite = torch.full(
            (batch_size, max_embed_dim),
            True,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        speech_to_overwrite[batch_indices, text_to_overwrite] = False
        if speech_lens is not None:
            speech_pad_position = speech_to_overwrite.cumsum(-1) <= speech_lens[:, None]
        speech_to_overwrite &= speech_to_overwrite.cumsum(-1) - 1 >= nb_speech_pad[
            :, None
        ].to(target_device)

        if speech_to_overwrite.sum() != speech_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of speech tokens is {torch.sum(special_speech_token_mask)} while"
                f" the number of speech given to the model is {num_speechs}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[speech_to_overwrite] = (
            speech_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        if speech_lens is not None:
            speech_to_overwrite &= speech_pad_position
        final_attention_mask |= speech_to_overwrite

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(
            input_ids == self.llm.config.pad_token_id
        )
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels #, position_ids
