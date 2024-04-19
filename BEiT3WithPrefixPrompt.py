# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.nn as nn

from torchscale.architecture.encoder import Encoder
from torchscale.component.embedding import (
    PositionalEmbedding,
    TextEmbedding,
    VisionEmbedding,
)
from torchscale.component.multiway_network import MutliwayEmbedding


class PromptLearner(nn.Module):
    def __init__(self, beit3_text_embed: TextEmbedding, args, **kwargs):
        super().__init__()

        ctx_init = kwargs.get("ctx_init", None)
        n_ctx = kwargs.get("n_ctx", 2)
        ctx_dim = beit3_text_embed.weight.shape[-1]

        self.num_soft_token = 0

        # Init hard prompt if existed
        if "ctx_init" in kwargs:
            # use given words to initialize context vectors
            n_ctx = len(ctx_init[0])
            print(n_ctx)
            with torch.no_grad():
                embedding = beit3_text_embed(ctx_init)
            ctx_vectors = embedding[0, : n_ctx - 1, :]
            prompt_prefix = kwargs["ori_ctx_init"]

        # Soft prompt
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors, requires_grad=True)  # to be optimized

        self.ctx_init = ctx_init
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim

    def forward(self, text_embeds: torch.Tensor):
        ctx = self.ctx.unsqueeze(0).expand(text_embeds.shape[0], -1, -1)
        print(ctx.shape)
        prompts = torch.cat((ctx, text_embeds[:, 1 : 64 - (ctx.shape[1] - 1), :]), 1)
        print(prompts.shape)
        return prompts


class BEiT3WithPrefixPrompt(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        assert args.multiway
        assert args.vocab_size > 0
        assert not args.share_encoder_input_output_embed
        self.text_embed = TextEmbedding(args.vocab_size, args.encoder_embed_dim)
        self.vision_embed = VisionEmbedding(
            args.img_size,
            args.patch_size,
            args.in_chans,
            args.encoder_embed_dim,
            contain_mask_token=True,
            prepend_cls_token=True,
        )
        # being consistent with Fairseq, which starts from 2 for position embedding
        embed_positions = MutliwayEmbedding(
            modules=[
                PositionalEmbedding(self.vision_embed.num_position_embeddings() + 2, args.encoder_embed_dim),
                PositionalEmbedding(args.max_source_positions, args.encoder_embed_dim),
            ],
            dim=1,
        )
        self.encoder = Encoder(
            args,
            embed_tokens=None,
            embed_positions=embed_positions,
            output_projection=None,
            is_encoder_decoder=False,
        )

    def set_prompter(
            self, prompter: PromptLearner
    ):
        self.prompter = prompter

    def forward(
        self,
        textual_tokens=None,
        visual_tokens=None,
        text_padding_position=None,
        attn_mask=None,
        vision_masked_position=None,
        incremental_state=None,
        positions=None,
    ):
        assert textual_tokens is not None or visual_tokens is not None

        if textual_tokens is None:
            x = self.vision_embed(visual_tokens, vision_masked_position)
            encoder_padding_mask = None
            multiway_split_position = -1
        elif visual_tokens is None:
            _x = self.text_embed(textual_tokens)
            x = self.prompter(_x)
            encoder_padding_mask = text_padding_position
            multiway_split_position = 0
        else:
            x1 = self.vision_embed(visual_tokens, vision_masked_position)
            multiway_split_position = x1.size(1)
            _x2 = self.text_embed(textual_tokens)
            x2 = self.prompter(_x2)
            x = torch.cat([x1, x2], dim=1)

            if text_padding_position is not None:
                encoder_padding_mask = torch.cat(
                    [
                        torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                        text_padding_position,
                    ],
                    dim=1,
                )
            else:
                encoder_padding_mask = None

        encoder_out = self.encoder(
            src_tokens=None,
            encoder_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            token_embeddings=x,
            multiway_split_position=multiway_split_position,
            incremental_state=incremental_state,
            positions=positions,
        )
        encoder_out["multiway_split_position"] = multiway_split_position

        return encoder_out
