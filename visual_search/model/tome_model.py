# import tome
#
# from typing import Tuple, Optional
#
# import torch
# from torch.nn.functional import _in_projection_packed, linear
#
# from tome.merge import merge_source, merge_wavg, bipartite_soft_matching
# from tome.utils import parse_r
#
# from open_clip.transformer import ResidualAttentionBlock, VisionTransformer
#
#
# # Since we don't necessarily have the swag code available, this patch is a little bit more involved
#
#
# class ToMeResidualAttentionBlock(ResidualAttentionBlock):
#     """
#     Modifications:
#     - Apply ToMe between the attention and mlp blocks
#     - Compute and propogate token size and potentially the token sources.
#     """
#
#     def forward(
#             self,
#             q_x: torch.Tensor,
#             k_x: Optional[torch.Tensor] = None,
#             v_x: Optional[torch.Tensor] = None,
#             attn_mask: Optional[torch.Tensor] = None,
#     ):
#         k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
#         v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
#
#         x_attn, metric = self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
#         x = q_x + self.ls_1(x_attn)
#
#         r = self._tome_info["r"].pop(0)
#         if r > 0:
#             # Apply ToMe here
#             merge, _ = bipartite_soft_matching(
#                 metric,
#                 r,
#                 self._tome_info["class_token"],
#                 self._tome_info["distill_token"],
#             )
#             x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
#
#         x = x + self.ls_2(self.mlp(self.ln_2(x)))
#         return x
#
#     def get_metric(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
#         # set up shape vars
#         bsz, tgt_len, embed_dim = query.shape
#         q, k, v = _in_projection_packed(query, key, value, self.attn.in_proj_weight, self.attn.in_proj_bias)
#         k = k.view(bsz, tgt_len, self.attn.num_heads, self.attn.embed_dim // self.attn.num_heads)
#         return k.mean(2)
#
#     def attention(
#             self,
#             q_x: torch.Tensor,
#             k_x: Optional[torch.Tensor] = None,
#             v_x: Optional[torch.Tensor] = None,
#             attn_mask: Optional[torch.Tensor] = None,
#     ):
#         k_x = k_x if k_x is not None else q_x
#         v_x = v_x if v_x is not None else q_x
#
#         attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
#         scores = self.attn(
#             q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
#         )[0]
#         metric = self.get_metric(q_x, k_x, v_x)
#         return scores, metric
#
#
# def make_transformer_class(transformer_class):
#     class ToMeVisionTransformer(transformer_class):
#         """
#         Modifications:
#         - Initialize r, token size, and token sources.
#         """
#
#         def forward(self, x: torch.Tensor) -> torch.Tensor:
#             self._tome_info["r"] = parse_r(len(self.transformer.resblocks), self.r)
#             self._tome_info["size"] = None
#             self._tome_info["source"] = None
#
#             # return super().forward(x)
#
#             # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
#             if self.input_patchnorm:
#                 # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
#                 x = x.reshape(x.shape[0], x.shape[1], self.grid_size[0], self.patch_size[0], self.grid_size[1], self.patch_size[1])
#                 x = x.permute(0, 2, 4, 1, 3, 5)
#                 x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
#                 x = self.patchnorm_pre_ln(x)
#                 x = self.conv1(x)
#             else:
#                 x = self.conv1(x)  # shape = [*, width, grid, grid]
#                 x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#                 x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#
#             # class embeddings and positional embeddings
#             x = torch.cat(
#                 [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
#                  x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#             x = x + self.positional_embedding.to(x.dtype)
#
#             # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
#             x = self.patch_dropout(x)
#             x = self.ln_pre(x)
#             x = self.transformer(x)
#             if self.attn_pool is not None:
#                 x = self.attn_pool(x)
#                 x = self.ln_post(x)
#                 pooled, tokens = self._global_pool(x)
#             else:
#                 pooled, tokens = self._global_pool(x)
#                 pooled = self.ln_post(pooled)
#
#             if self.proj is not None:
#                 pooled = pooled @ self.proj
#
#             if self.output_tokens:
#                 return pooled, tokens
#
#             return pooled
#
#     return ToMeVisionTransformer
#
#
# def apply_patch(model, trace_source: bool = False, prop_attn: bool = False):
#     """
#     Applies ToMe to this transformer. Afterward, set r using model.r.
#
#     If you want to know the source of each token (e.g., for visualization), set trace_source = true.
#     The sources will be available at model._tome_info["source"] afterward.
#
#     For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
#     the shelf. For trianing and for evaluating MAE models off the self set this to be False.
#     """
#
#     if model.__class__.__name__ == "ToMeVisionTransformer":
#         # This model was already patched!
#         return
#
#     ToMeVisionTransformer = make_transformer_class(model.__class__)
#
#     model.__class__ = ToMeVisionTransformer
#     model.r = 0
#     model._tome_info = {
#         "r": model.r,
#         "size": None,
#         "source": None,
#         "trace_source": trace_source,
#         "prop_attn": prop_attn,
#         "class_token": True,
#         "distill_token": False,
#     }
#
#     for module in model.modules():
#         if isinstance(module, ResidualAttentionBlock):
#             module.__class__ = ToMeResidualAttentionBlock
#             module._tome_info = model._tome_info
#         if isinstance(module, torch.nn.MultiheadAttention):
#             module.batch_first = True