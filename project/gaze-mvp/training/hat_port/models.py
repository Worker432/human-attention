# Adapted from cvlab-stonybrook/HAT, MIT license in LICENSE.
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional
from . import position_encoding as pe
from .compat import weight_init
from .transformer import TransformerEncoder, TransformerEncoderLayer
from .image_encoder import ImageFeatureEncoder
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SelfAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self,
                    tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt2,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self,
                tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask,
                                    query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask,
                                 query_pos)

class CrossAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn_weights

    def forward_pre(self,
                    tgt,
                    memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self,
                tgt,
                memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class FFNLayer(nn.Module):
    def __init__(self,
                 d_model,
                 dim_feedforward=2048,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class HumanAttnTransformer(nn.Module):
    def __init__(
        self,
        pa,
        num_decoder_layers: int,
        hidden_dim: int,
        nhead: int,
        ntask: int,
        tgt_vocab_size: int,
        num_output_layers: int,
        separate_fix_arch: bool = False,
        train_encoder: bool = False,
        train_foveator: bool = True,
        train_pixel_decoder: bool = False,
        use_dino: bool = True,
        pre_norm: bool = False,
        dropout: float = 0.1,
        dim_feedforward: int = 512,
        parallel_arch: bool = False,
        dorsal_source: list = ["P2"],
        num_encoder_layers: int = 3,
        output_centermap: bool = False,
        output_saliency: bool = False,
        output_target_map: bool = False,
        combine_pos_emb: bool = True,
        combine_all_emb: bool = False,
        transfer_learning_setting: str = 'none',
        project_queries: bool = True,
        is_pretraining: bool = False,
        output_feature_map_name: str = 'P4',
    ):
        super(HumanAttnTransformer, self).__init__()
        self.pa = pa
        self.num_decoder_layers = num_decoder_layers
        self.is_pretraining = is_pretraining
        self.combine_pos_emb = combine_pos_emb
        self.combine_all_emb = combine_all_emb
        self.output_feature_map_name = output_feature_map_name
        self.parallel_arch = parallel_arch
        self.dorsal_source = dorsal_source
        assert len(dorsal_source) > 0, "need to specify dorsal source: P1, P2!"
        self.output_centermap = output_centermap
        self.output_saliency = output_saliency
        self.output_target_map = output_target_map

        # Encoder: Deformable Attention Transformer
        self.train_encoder = train_encoder
        self.encoder = ImageFeatureEncoder(pa.backbone_config, dropout,
                                           pa.pixel_decoder)
        self.symbol_offset = len(self.pa.special_symbols)
        if not train_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.train_pixel_decoder = train_pixel_decoder
        if train_pixel_decoder:
            self.encoder.pixel_decoder.train()
            for param in self.encoder.pixel_decoder.parameters():
                param.requires_grad = True
        featmap_channels = 256
        if hidden_dim != featmap_channels:
            self.input_proj = nn.Conv2d(featmap_channels,
                                         hidden_dim,
                                         kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()

        # Transfer learning setting (only support COCO-Search18 for now, where
        # we assume 18 search targets).
        self.transfer_learning_setting = transfer_learning_setting
        self.project_queries = project_queries
        if transfer_learning_setting == 'freeview2search':
            if project_queries:
                assert ntask == 18, "only support 18 task for freeview2search"
                # TODO: Create a new query for every new search task
                ntask = 1
                self.new_query_embed = nn.Embedding(18, hidden_dim)
                self.freeview2search_mlp = MLP(1, 64, 18, 3)
        elif transfer_learning_setting == 'search2freeview':
            if project_queries:
                assert ntask == 1, "only support 1 task for search2freeview"
                # Create a new query for the new free-viewing task
                ntask = 18
                self.new_query_embed = nn.Embedding(1, hidden_dim)
                self.search2freeview_mlp = MLP(18, 64, 1, 3)
        elif transfer_learning_setting == 'none' or transfer_learning_setting == 'finetune':
            pass
        else:
            raise ValueError(
                f"transfer_learning_setting {transfer_learning_setting} not supported."
            )

        # Queries
        self.ntask = ntask
        self.aux_queries = 0
        if self.output_centermap:
            self.aux_queries += 80
        if self.output_saliency:
            self.aux_queries += 1
        self.query_embed = nn.Embedding(ntask + self.aux_queries, hidden_dim)
        self.query_pos = nn.Embedding(ntask + self.aux_queries, hidden_dim)

        # Decoder
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers_dorsal = nn.ModuleList()
        self.transformer_cross_attention_layers_ventral = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))
            self.transformer_cross_attention_layers_dorsal.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))
            if not self.parallel_arch:
                self.transformer_cross_attention_layers_ventral.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nhead,
                        dropout=dropout,
                        normalize_before=pre_norm,
                    ))
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))

        self.num_encoder_layers = num_encoder_layers
        if self.parallel_arch and num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                normalize_before=pre_norm)
            encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
            self.working_memory_encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm)
            if not train_foveator:
                self.working_memory_encoder.eval()
                for param in self.working_memory_encoder.parameters():
                    param.requires_grad = False

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Task heads
        self.termination_predictor = MLP(hidden_dim + 1, hidden_dim, 1,
                                         num_output_layers)
        self.fixation_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                  num_output_layers)
        if self.output_target_map:
            self.target_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                    num_output_layers)
        if self.output_centermap:
            self.centermap_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                       num_output_layers)
        if self.output_saliency:
            self.saliency_embed = MLP(hidden_dim, hidden_dim, hidden_dim,
                                      num_output_layers)

        # Positional embedding
        self.pixel_loc_emb = pe.PositionalEncoding2D(pa,
                                                     hidden_dim,
                                                     height=pa.im_h // 4,
                                                     width=pa.im_w // 4,
                                                     dropout=dropout)
        if self.output_feature_map_name == 'P4':
            self.pos_scale = 1
        elif self.output_feature_map_name == 'P2':
            self.pos_scale = 4
        else:
            raise NotImplementedError
        # self.positional_encoding = pe.PositionalEncoding(hidden_dim)

        self.fix_ind_emb = nn.Embedding(pa.max_traj_length, hidden_dim)

        # Embedding for distinguishing dorsal or ventral embeddings
        self.dorsal_ind_emb = nn.Embedding(2, hidden_dim)  # P1 and P2
        self.ventral_ind_emb = nn.Embedding(1, hidden_dim)

        # self.visual_pos_fuse_layer = nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=1, padding=0)
        if self.is_pretraining:
            self.pretrain_task_head = MLP(hidden_dim, 32, 1, 3)

    def forward(self,
                img: torch.Tensor,
                tgt_seq: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                tgt_seq_high: torch.Tensor,
                task_ids: torch.Tensor = None,
                return_attn_weights=False,
                img_ids: torch.Tensor = None):

        # Prepare dorsal embeddings
        img_embs_s4, img_embs_s1, img_embs_s2 = self.encoder(img)
        high_res_featmaps = self.input_proj(img_embs_s4)
        if self.output_feature_map_name == 'P4':
            output_featmaps = high_res_featmaps
        elif self.output_feature_map_name == 'P2':
            output_featmaps = self.input_proj(img_embs_s2)
        else:
            raise NotImplementedError

        dorsal_embs, dorsal_pos, scale_embs = [], [], []
        if "P1" in self.dorsal_source:
            # C x 10 x 16
            img_embs = self.input_proj(img_embs_s1)
            bs, c, h, w = img_embs.shape
            pe = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:],
                                                     scale=8)
            img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)
            scale_embs.append(
                self.dorsal_ind_emb.weight[0].unsqueeze(0).unsqueeze(0).expand(
                    img_embs.size(0), bs, c))
            dorsal_embs.append(img_embs)
            dorsal_pos.append(
                pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))
        if "P2" in self.dorsal_source:
            # C x 20 x 32
            img_embs = self.input_proj(img_embs_s2)
            bs, c, h, w = img_embs.shape
            pe = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:],
                                                     scale=4)
            img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)
            scale_embs.append(
                self.dorsal_ind_emb.weight[1].unsqueeze(0).unsqueeze(0).expand(
                    img_embs.size(0), bs, c))
            dorsal_embs.append(img_embs)
            dorsal_pos.append(
                pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))

        dorsal_embs = torch.cat(dorsal_embs, dim=0)
        dorsal_pos = torch.cat(dorsal_pos, dim=0)
        scale_embs = torch.cat(scale_embs, dim=0)

        if img_ids is not None and self.is_pretraining:
            # Pre-training: combine all images with every scanpath and
            # label each image-scanpath pair with image name index
            bs = img.shape[0]
            high_res_featmaps = high_res_featmaps.unsqueeze(0).expand(bs, bs, *high_res_featmaps.shape[1:])
            high_res_featmaps = high_res_featmaps.reshape(bs * bs, *high_res_featmaps.shape[2:])
            l, bs, ch = dorsal_embs.shape
            dorsal_embs = dorsal_embs.unsqueeze(1).expand(l, bs, bs, ch).reshape(l, bs * bs, ch)
            dorsal_pos = dorsal_pos.unsqueeze(1).expand(l, bs, bs, ch).reshape(l, bs * bs, ch)
            scale_embs =  scale_embs.unsqueeze(1).expand(l, bs, bs, ch).reshape(l, bs * bs, ch)
            tgt_len = tgt_seq_high.shape[1]
            tgt_seq = tgt_seq.unsqueeze(1).expand(bs, bs, tgt_len).reshape(bs * bs, tgt_len)
            tgt_seq_high = tgt_seq_high.unsqueeze(1).expand(bs, bs, tgt_len).reshape(bs * bs, tgt_len)
            tgt_padding_mask = tgt_padding_mask.unsqueeze(1).expand(bs, bs, tgt_len).reshape(bs * bs, tgt_len)

        bs = high_res_featmaps.size(0)
        # Prepare ventral embeddings
        if tgt_seq_high is None:
            tgt_seq = tgt_seq.transpose(0, 1)
            ventral_embs = torch.gather(
                torch.cat([
                    torch.zeros(1, *img_embs.shape[1:],
                                device=img_embs.device), img_embs
                ],
                          dim=0), 0,
                tgt_seq.unsqueeze(-1).expand(*tgt_seq.shape,
                                             img_embs.size(-1)))
            ventral_pos = self.pixel_loc_emb(
                tgt_seq)  # Pos for fixation location
        else:
            tgt_seq_high = tgt_seq_high.transpose(0, 1)
            highres_embs = high_res_featmaps.view(bs, c, -1).permute(2, 0, 1)
            ventral_embs = torch.gather(
                torch.cat([
                    torch.zeros(1,
                                *highres_embs.shape[1:],
                                device=img_embs.device), highres_embs
                ],
                          dim=0), 0,
                tgt_seq_high.unsqueeze(-1).expand(*tgt_seq_high.shape,
                                                  highres_embs.size(-1)))
            # Pos for fixation location
            ventral_pos = self.pixel_loc_emb(tgt_seq_high)

        # Add pos into embeddings for attention prediction
        if self.combine_pos_emb:
            # Dorsal embeddings
            dorsal_embs += dorsal_pos
            dorsal_pos.fill_(0)
            # Ventral embeddings
            ventral_embs += ventral_pos
            ventral_pos.fill_(0)
            # High-res featmap
            # high_res_featmaps = self.visual_pos_fuse_layer(
            #     torch.cat([high_res_featmaps,
            #                self.pixel_loc_emb.forward_featmaps(
            #                    high_res_featmaps.shape[-2:]).expand_as(high_res_featmaps)], dim=-3))
            output_featmaps += self.pixel_loc_emb.forward_featmaps(
                output_featmaps.shape[-2:], scale=self.pos_scale)

        # Add embedding indicator embedding into pos embedding
        dorsal_pos += scale_embs
        ventral_pos += self.ventral_ind_emb.weight.unsqueeze(0).expand(
            *ventral_pos.shape)

        # Temporal embedding for fixations
        # ventral_pos += self.positional_encoding(
        #     ventral_embs).repeat(1, bs, 1)
        ventral_pos += self.fix_ind_emb.weight[:ventral_embs.
                                               size(0)].unsqueeze(1).repeat(
                                                   1, bs, 1)
        ventral_pos[tgt_padding_mask.transpose(0, 1)] = 0

        if self.combine_all_emb:
            dorsal_embs += dorsal_pos
            ventral_embs += ventral_pos
            dorsal_pos = ventral_pos = None

        # Update working memory
        if self.parallel_arch:
            working_memory = torch.cat([dorsal_embs, ventral_embs], dim=0)
            padding_mask = torch.cat([
                torch.zeros(bs, dorsal_embs.size(0),
                            device=dorsal_embs.device).bool(), tgt_padding_mask
            ],
                                     dim=1)
            if self.combine_all_emb:
                working_memory_pos = None
            else:
                working_memory_pos = torch.cat([dorsal_pos, ventral_pos],
                                               dim=0)
            # working_memory = dorsal_embs # ventral_embs
            # padding_mask = None # tgt_padding_mask
            # working_memory_pos = dorsal_pos # ventral_pos

            if self.num_encoder_layers > 0:
                working_memory = self.working_memory_encoder(
                    working_memory,
                    src_key_padding_mask=padding_mask,
                    pos=working_memory_pos)
            dorsal_embs = working_memory
            dorsal_pos = working_memory_pos
        else:
            padding_mask = None

        # Update queries with attention
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, bs, 1)
        query_pos = None

        # Let model know which fixation it's going to predict
        num_fixs = (tgt_padding_mask.size(1) -
                    torch.sum(tgt_padding_mask, dim=1)).unsqueeze(0).expand(
                        self.ntask, bs)
        # num_fix_embed = self.fix_ind_emb.weight[num_fixs - 1]
        # query_embed += num_fix_embed

        attn_weights_all_layers = []
        for i in range(self.num_decoder_layers):
            # Dorsal cross attention
            query_embed, attn_weights = self.transformer_cross_attention_layers_dorsal[
                i](query_embed,
                   dorsal_embs,
                   memory_mask=None,
                   memory_key_padding_mask=padding_mask,
                   pos=dorsal_pos,
                   query_pos=query_pos)
            if return_attn_weights:
                attn_weights_all_layers.append(attn_weights)

            if not self.parallel_arch:
                # Ventral cross attention
                query_embed, _ = self.transformer_cross_attention_layers_ventral[
                    i](query_embed,
                       ventral_embs,
                       memory_mask=None,
                       memory_key_padding_mask=tgt_padding_mask,
                       pos=ventral_pos,
                       query_pos=query_pos)

            # if self.ntask > 1:
            #     # Self attention
            #     query_embed = self.transformer_self_attention_layers[i](
            #         query_embed,
            #         query_pos=query_pos,
            #     )

            # FFN
            query_embed = self.transformer_ffn_layers[i](query_embed)

        if img_ids is not None:
            num_imgs = len(img_ids)
            sp_labels = img_ids.unsqueeze(1).expand(num_imgs, num_imgs)
            img_labels = img_ids.unsqueeze(0).expand(num_imgs, num_imgs)
            pretrain_labels = (sp_labels == img_labels).to(torch.float32).view(-1, 1)
            return self.pretrain_task_head(query_embed.squeeze(0)), pretrain_labels

        # Prediction
        if self.transfer_learning_setting == 'none' or not self.project_queries:
            pred_queries = query_embed[:self.ntask]
        elif self.transfer_learning_setting == 'search2freeview':
            pred_queries = query_embed[:self.ntask].permute(1, 2, 0)
            pred_queries = self.search2freeview_mlp(pred_queries).permute(
                2, 0, 1)
            pred_queries += self.new_query_embed.weight.unsqueeze(1)
            num_fixs = num_fixs[:1]
        elif self.transfer_learning_setting == 'freeview2search':
            pred_queries = query_embed[:self.ntask].permute(1, 2, 0)
            pred_queries = self.freeview2search_mlp(pred_queries).permute(
                2, 0, 1)
            pred_queries += self.new_query_embed.weight.unsqueeze(1)
            num_fixs = num_fixs.expand(18, bs)
        else:
            raise NotImplementedError
        x = torch.cat([pred_queries, num_fixs.unsqueeze(-1)], dim=-1)
        output_termination = self.termination_predictor(x)
        out = {
            "pred_termination": output_termination.squeeze(-1).transpose(0, 1)
        }

        fixation_embed = self.fixation_embed(pred_queries)
        outputs_fixation_map = torch.einsum("lbc,bchw->lbhw", fixation_embed,
                                            output_featmaps)
        if self.training:
            out["pred_fixation_map"] = outputs_fixation_map.transpose(0, 1)
        else:
            outputs_fixation_map = outputs_fixation_map[task_ids,
                                                        torch.arange(bs)]
            outputs_fixation_map = torch.sigmoid(outputs_fixation_map)
            outputs_fixation_map = F.interpolate(
                outputs_fixation_map.unsqueeze(1),
                size=(self.pa.im_h, self.pa.im_w)).squeeze(1)
            out["pred_fixation_map"] = outputs_fixation_map

        # Auxiliary tasks
        if self.training:
            ind = self.ntask
            if self.output_centermap:
                centermap_embed = self.centermap_embed(query_embed[ind:ind +
                                                                   80])
                out["pred_centermap"] = torch.einsum(
                    "lbc,bchw->lbhw", centermap_embed,
                    high_res_featmaps).transpose(0, 1)
                ind += 80
            if self.output_saliency:
                saliency_embed = self.saliency_embed(
                    # query_embed[ind: ind + 1])
                    query_embed[:self.ntask])
                outputs_saliency = torch.einsum("lbc,bchw->lbhw",
                                                saliency_embed,
                                                high_res_featmaps)
                out["pred_saliency"] = outputs_saliency.transpose(0, 1)
                ind += 1
            if self.output_target_map:
                target_embed = self.target_embed(query_embed[:self.ntask])
                out["pred_target_map"] = torch.einsum(
                    "lbc, bchw->lbhw", target_embed,
                    high_res_featmaps).transpose(0, 1)

        if return_attn_weights:
            out['cross_attn_weights'] = attn_weights_all_layers
        return out

    def encode(self, img: torch.Tensor):
        # Prepare dorsal embeddings
        img_embs_s4, img_embs_s1, img_embs_s2 = self.encoder(img)
        high_res_featmaps = self.input_proj(img_embs_s4)
        if self.output_feature_map_name == 'P4':
            output_featmaps = high_res_featmaps
        elif self.output_feature_map_name == 'P2':
            output_featmaps = self.input_proj(img_embs_s2)
        else:
            raise NotImplementedError
        dorsal_embs, dorsal_pos, scale_embs = [], [], []
        if "P1" in self.dorsal_source:
            # C x 10 x 16
            img_embs = self.input_proj(img_embs_s1)
            bs, c, h, w = img_embs.shape
            pe = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:],
                                                     scale=8)
            img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)
            scale_embs.append(
                self.dorsal_ind_emb.weight[0].unsqueeze(0).unsqueeze(0).expand(
                    img_embs.size(0), bs, c))
            dorsal_embs.append(img_embs)
            dorsal_pos.append(
                pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))
        if "P2" in self.dorsal_source:
            # C x 20 x 32
            img_embs = self.input_pro1(img_embs_s2)
            bs, c, h, w = img_embs.shape
            pe = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:],
                                                     scale=4)
            img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)
            scale_embs.append(
                self.dorsal_ind_emb.weight[1].unsqueeze(0).unsqueeze(0).expand(
                    img_embs.size(0), bs, c))
            dorsal_embs.append(img_embs)
            dorsal_pos.append(
                pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))

        dorsal_embs = torch.cat(dorsal_embs, dim=0)
        dorsal_pos = torch.cat(dorsal_pos, dim=0)
        scale_embs = torch.cat(scale_embs, dim=0)
        dorsal_pos = (dorsal_pos, scale_embs)

        return dorsal_embs, dorsal_pos, None, (high_res_featmaps, output_featmaps)

        # high_res_featmaps, img_embs = self.encoder(img)
        # high_res_featmaps = self.input_proj(high_res_featmaps)
        # img_embs = self.input_proj(img_embs)
        # bs, c, h, w = high_res_featmaps.shape
        # img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)

        # # Add position encoding to feature maps
        # fm_pos = self.pixel_loc_emb(
        #     torch.arange(h * w).to(img.device).unsqueeze(1).expand(
        #         h * w, bs) + self.symbol_offset).permute(1, 2, 0).view(
        #             bs, c, h, w)
        # high_res_featmaps += fm_pos
        # return img_embs, high_res_featmaps

    def decode_and_predict(self,
                           dorsal_embs: torch.Tensor,
                           dorsal_pos: tuple,
                           dorsal_mask: torch.Tensor,
                           high_res_featmaps: tuple[torch.Tensor, torch.Tensor],
                           tgt_seq: torch.Tensor,
                           tgt_padding_mask: torch.Tensor,
                           tgt_seq_high: torch.Tensor,
                           task_ids: torch.Tensor = None,
                           return_attn_weights: bool = False):
        # Prepare ventral embeddings
        dorsal_pos, scale_embs = dorsal_pos
        high_res_featmaps, output_featmaps = high_res_featmaps
        bs, c = high_res_featmaps.shape[:2]
        highres_embs = high_res_featmaps.view(bs, c, -1).permute(2, 0, 1)
        # tgt_seq = tgt_seq.transpose(0,1)
        tgt_seq_high = tgt_seq_high.transpose(0, 1)
        if dorsal_mask is None:
            dorsal_mask = torch.zeros(1,
                                      *highres_embs.shape[1:],
                                      device=dorsal_embs.device)
        ventral_embs = torch.gather(
            torch.cat([dorsal_mask, highres_embs], dim=0), 0,
            tgt_seq_high.unsqueeze(-1).expand(*tgt_seq_high.shape,
                                              highres_embs.size(-1)))
        ventral_pos = self.pixel_loc_emb(
            tgt_seq_high)  # Pos for fixation location

        # Add pos into embeddings for attention prediction
        if self.combine_pos_emb:
            # Dorsal embeddings
            dorsal_embs += dorsal_pos
            dorsal_pos = torch.zeros_like(dorsal_pos)
            # Ventral embeddings
            ventral_embs += ventral_pos
            ventral_pos = torch.zeros_like(ventral_pos)
            # High-res featmap
            # high_res_featmaps = self.visual_pos_fuse_layer(
            #     torch.cat([high_res_featmaps,
            #                self.pixel_loc_emb.forward_featmaps(
            #                    high_res_featmaps.shape[-2:]).expand_as(high_res_featmaps)], dim=-3))
            output_featmaps_wpos = output_featmaps + self.pixel_loc_emb.forward_featmaps(
                output_featmaps.shape[-2:], self.pos_scale)

        # Add embedding indicator embedding into pos embedding
        dorsal_pos += scale_embs
        ventral_pos += self.ventral_ind_emb.weight.unsqueeze(0).expand(
            *ventral_pos.shape)

        # Temporal embedding for fixations
        # ventral_pos += self.positional_encoding(
        #     ventral_embs).repeat(1, bs, 1)
        ventral_pos += self.fix_ind_emb.weight[:ventral_embs.
                                               size(0)].unsqueeze(1).repeat(
                                                   1, bs, 1)

        if self.combine_all_emb:
            dorsal_embs += dorsal_pos
            ventral_embs += ventral_pos
            dorsal_pos = ventral_pos = None

        # Update working memory with both dorsal and ventral memory
        # if using parallel architecture
        if self.parallel_arch:
            working_memory = torch.cat([dorsal_embs, ventral_embs], dim=0)
            padding_mask = torch.cat(
                [
                    torch.zeros(
                        bs, dorsal_embs.size(0),
                        device=dorsal_embs.device).bool(), tgt_padding_mask
                ],
                dim=1) if tgt_padding_mask is not None else None
            if self.combine_all_emb:
                working_memory_pos = None
            else:
                working_memory_pos = torch.cat([dorsal_pos, ventral_pos],
                                               dim=0)
            # working_memory = dorsal_embs # ventral_embs
            # padding_mask = None # tgt_padding_mask
            # working_memory_pos = dorsal_pos # ventral_pos

            if self.num_encoder_layers > 0:
                working_memory = self.working_memory_encoder(
                    working_memory,
                    src_key_padding_mask=padding_mask,
                    pos=working_memory_pos)
            dorsal_embs = working_memory
            dorsal_pos = working_memory_pos
        else:
            padding_mask = None

        # Update queries with attention
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, bs, 1)
        query_pos = None
        num_fixs = torch.ones(self.ntask, bs).to(
            query_embed.device) * tgt_seq_high.size(0)
        # num_fix_embed = self.fix_ind_emb.weight[num_fixs.to(torch.long) - 1]
        # query_embed += num_fix_embed

        attn_weights_all_layers = []
        for i in range(self.num_decoder_layers):
            # Dorsal cross attention
            query_embed, attn_weights = self.transformer_cross_attention_layers_dorsal[
                i](query_embed,
                   dorsal_embs,
                   memory_mask=None,
                   memory_key_padding_mask=padding_mask,
                   pos=dorsal_pos,
                   query_pos=query_pos)
            if return_attn_weights:
                attn_weights_all_layers.append(attn_weights)

            if not self.parallel_arch:
                # Ventral cross attention
                query_embed, _ = self.transformer_cross_attention_layers_ventral[
                    i](query_embed,
                       ventral_embs,
                       memory_mask=None,
                       memory_key_padding_mask=tgt_padding_mask,
                       pos=ventral_pos,
                       query_pos=query_pos)
            # if self.ntask > 1:
            #     # Self attention
            #     query_embed = self.transformer_self_attention_layers[i](
            #         query_embed,
            #         query_pos=query_pos,
            #     )

            # FFN
            query_embed = self.transformer_ffn_layers[i](query_embed)

        # Prediction
        if self.transfer_learning_setting == 'none' or not self.project_queries:
            pred_queries = query_embed[:self.ntask]
        elif self.transfer_learning_setting == 'search2freeview':
            pred_queries = query_embed[:self.ntask].permute(1, 2, 0)
            pred_queries = self.search2freeview_mlp(pred_queries).permute(
                2, 0, 1)
            pred_queries += self.new_query_embed.weight.unsqueeze(1)
            num_fixs = num_fixs[:1]
        elif self.transfer_learning_setting == 'freeview2search':
            pred_queries = query_embed[:self.ntask].permute(1, 2, 0)
            pred_queries = self.freeview2search_mlp(pred_queries).permute(
                2, 0, 1)
            pred_queries += self.new_query_embed.weight.unsqueeze(1)
            num_fixs = num_fixs.expand(18, bs)
        else:
            raise NotImplementedError

        x = torch.cat([pred_queries, num_fixs.unsqueeze(-1)], dim=-1)
        output_termination = torch.sigmoid(self.termination_predictor(x))
        out = {
            "pred_termination":
            output_termination.squeeze(-1)[task_ids,
                                           torch.arange(bs)]
        }

        fixation_embed = self.fixation_embed(pred_queries)
        outputs_fixation_map = torch.einsum("lbc,bchw->lbhw", fixation_embed,
                                            output_featmaps_wpos)
        outputs_fixation_map = torch.sigmoid(
            outputs_fixation_map[task_ids, torch.arange(bs)])
        outputs_fixation_map = F.interpolate(outputs_fixation_map.unsqueeze(1),
                                             size=(self.pa.im_h,
                                                   self.pa.im_w)).squeeze(1)
        out["pred_fixation_map"] = outputs_fixation_map.view(bs, -1)

        if return_attn_weights:
            out['cross_attn_weights'] = attn_weights_all_layers
        return out
