# ultralytics/nn/modules/anti_uav.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, batched_nms

# ===============================
# 1) QueryConvGate（孪生查询门控）
# ===============================
class QueryConvGate(nn.Module):
    """
    模板 RoI -> 通道权重向量 -> 对搜索特征逐通道放大/抑制
    mode='scale' 仅做通道缩放；mode='dwconv' 叠加深度可分卷积的调制项
    """
    def __init__(self, c, reduce=16, mode="scale"):
        super().__init__()
        hid = max(c // reduce, 8)
        self.fc1 = nn.Linear(c, hid)
        self.fc2 = nn.Linear(hid, c)
        self.mode = mode
        if mode == "dwconv":
            self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
            nn.init.constant_(self.dw.weight, 0.)

    def forward(self, x, tpl_roi_feat):
        # tpl_roi_feat: [B,C,S,S] (模板 RoI 特征)
        w = F.adaptive_avg_pool2d(tpl_roi_feat, 1).flatten(1)         # [B,C]
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w)))).view(x.size(0), x.size(1), 1, 1)
        return x * w if self.mode == "scale" else (self.dw(x) + x * w)

# ==============
# 2) DSRPN（简化）
# ==============
class DSRPN(nn.Module):
    """
    Dual-Semantic RPN（简化）：在每个金字塔层做 obj + box 回归，解码到原图坐标后做 TopK+NMS。
    可视为“语义先验（obj） + 定位”二元，“dual semantic”的语义先验分支用于背景抑制/先验概率。
    """
    def __init__(self, in_channels=(256, 512, 1024), anchors=(1.0,), strides=(8, 16, 32), topk=1000, nms_thr=0.7):
        super().__init__()
        self.strides = strides
        self.topk = topk
        self.nms_thr = nms_thr
        A = len(anchors)
        self.anchors = anchors

        # 轻量 RPN 头：3x3 conv -> obj(A) + box(4A)
        self.rpn = nn.ModuleList()
        for c in in_channels:
            self.rpn.append(nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True),
            ))
        self.obj_heads = nn.ModuleList([nn.Conv2d(c, len(anchors), 1) for c in in_channels])
        self.box_heads = nn.ModuleList([nn.Conv2d(c, 4*len(anchors), 1) for c in in_channels])

    @staticmethod
    def _grid_anchors(H, W, stride, device, scales):
        # 以 (cx,cy,w,h) 生成 anchors（归一为像素单位）
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        base = torch.stack([cx, cy], -1)  # [H,W,2]
        out = []
        for s in scales:
            w = torch.full_like(cx, float(stride * s))
            h = torch.full_like(cy, float(stride * s))
            out.append(torch.stack([base[...,0], base[...,1], w, h], -1))
        # [A,H,W,4]
        return torch.stack(out, 0).permute(0,2,1,3)  # [A,W,H,4] -> 保持一致

    @staticmethod
    def _delta_decode(anchors, deltas):
        # anchors, deltas: [...,4]  (cx,cy,w,h) & (tx,ty,tw,th)
        cx = anchors[...,0] + deltas[...,0] * anchors[...,2]
        cy = anchors[...,1] + deltas[...,1] * anchors[...,3]
        w  = anchors[...,2] * torch.exp(deltas[...,2])
        h  = anchors[...,3] * torch.exp(deltas[...,3])
        x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
        return torch.stack([x1,y1,x2,y2], -1)

    def forward(self, feats, img_shapes):
        """
        feats: [P3,P4,P5]  每个[B,C,H,W]
        img_shapes: list[(H,W)]
        return: proposals(list[B, K, 4]), scores(list[B,K]), levels(list[B,K])，以及语义先验 obj_map 供可视化
        """
        B = feats[0].size(0)
        device = feats[0].device
        all_props, all_scores, all_lvls = [], [], []
        # 各层生成候选
        for lvl, (f, head, box_head, stride) in enumerate(zip(self.rpn, self.obj_heads, self.box_heads, self.strides)):
            x = head(feats[lvl])
            obj = torch.sigmoid(self.obj_heads[lvl](x))              # [B,A,H,W]
            box = self.box_heads[lvl](x)                             # [B,4A,H,W]
            A, H, W = obj.size(1), obj.size(2), obj.size(3)
            anchors = self._grid_anchors(H, W, stride, device, self.anchors)  # [A,W,H,4]
            anchors = anchors.permute(0,2,1,3).reshape(1, A, H, W, 4)         # [1,A,H,W,4]

            # decode
            box = box.view(B, A, 4, H, W).permute(0,1,3,4,2)                  # [B,A,H,W,4]
            props = self._delta_decode(anchors, box)                           # [B,A,H,W,4]

            # 展平 + Topk
            scores = obj.reshape(B, -1)
            props  = props.reshape(B, -1, 4)
            # 先取每张图 top (3 * topk) 再 NMS
            k = min(scores.shape[1], self.topk * 3)
            sc, idx = torch.topk(scores, k, dim=1)
            gather_props = torch.gather(props, 1, idx.unsqueeze(-1).expand(-1,-1,4))
            lvls = torch.div(idx, A*H*W, rounding_mode='trunc')*0 + lvl  # 粗略层索引（仅用于调试）

            # NMS
            props_nms, scores_nms, lvls_nms = [], [], []
            for b in range(B):
                keep = batched_nms(gather_props[b], sc[b], lvls[b], self.nms_thr)
                keep = keep[:self.topk]
                props_nms.append(gather_props[b][keep])
                scores_nms.append(sc[b][keep])
                lvls_nms.append(lvls[b][keep])
            all_props.append(torch.stack([p for p in props_nms], 0))
            all_scores.append(torch.stack([s for s in scores_nms], 0))
            all_lvls.append(torch.stack([l for l in lvls_nms], 0))

        # 合并三个尺度，再做一次全局 TopK
        props_cat  = torch.cat(all_props, 1)   # [B, K', 4]
        scores_cat = torch.cat(all_scores, 1)  # [B, K']
        lvls_cat   = torch.cat(all_lvls, 1)
        K_final = min(self.topk, scores_cat.shape[1])
        sc, idx = torch.topk(scores_cat, K_final, dim=1)
        props_final = torch.gather(props_cat, 1, idx.unsqueeze(-1).expand(-1,-1,4))
        lvls_final  = torch.gather(lvls_cat, 1, idx)

        return props_final, sc, lvls_final

# ===========================
# 3) Versatile R-CNN Head（多分支）
# ===========================
class VersatileRCNNHead(nn.Module):
    """
    多分支 VR-CNN 头：
      - 语义先验（semantic prior）分支：cls_logit（是否为目标）
      - 背景抑制（background suppression）分支：利用 Query 门控权重对 RoI 特征的“可见度”打分
      - 相似性（similarity）分支：模板RoI vs 候选RoI 的匹配分
      - 回归分支：IoU/GIoU/L1 回归用于 bbox 细化
    最终融合：sigmoid( α*sem + β*sim + γ*bkg )
    """
    def __init__(self, c=256, roi=7, num_classes=1, alpha=0.5, beta=0.4, gamma=0.1):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        self.shared = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)
        )
        # 语义先验（分类）
        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, num_classes))
        # 相似性：RoI 与 模板拼接后卷积
        self.sim_conv = nn.Sequential(nn.Conv2d(c*2, c, 3, 1, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True))
        self.sim_fc   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 1))
        # 回归
        self.reg_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 4))

    def forward(self, roi_search, roi_tpl, gate_weight=None):
        """
        roi_search: [B*K, C, S, S]
        roi_tpl   : [B*K, C, S, S]
        gate_weight（可选）: [B, C, 1, 1] broadcast 后用于估计 background 抑制分
        """
        h = self.shared(roi_search)                      # 候选特征
        # A) 语义先验
        sem_logit = self.cls_head(h).squeeze(-1) if self.cls_head[-1].out_features==1 else self.cls_head(h)  # [B*K] or [B*K,C]
        sem_score = torch.sigmoid(sem_logit.squeeze(-1))

        # B) 相似性
        sim = torch.cat([roi_search, roi_tpl], 1)
        sim_h = self.sim_conv(sim)
        sim_score = torch.sigmoid(self.sim_fc(sim_h).squeeze(1))

        # C) 背景抑制（使用门控幅度的 proxy，越大说明模板越“激活”候选）
        if gate_weight is not None:
            # gate_weight: [B, C, 1, 1] -> broadcast to [B*K, C, 1, 1]
            # 这里取均值作为抑制得分 proxy
            bkg_score = gate_weight.flatten(1).mean(1)    # [B]
            bkg_score = bkg_score.repeat_interleave(int(roi_search.shape[0] / gate_weight.shape[0]))
            bkg_score = torch.sigmoid(bkg_score)
        else:
            bkg_score = torch.zeros_like(sim_score)

        # 融合
        fused = torch.sigmoid(self.alpha*sem_score + self.beta*sim_score + self.gamma*bkg_score)

        # 回归
        delta = self.reg_head(h)                          # [B*K,4]
        return fused, sem_score, sim_score, bkg_score, delta

# ==================================
# 4) 包装器：SiamTwoStage（替代 Detect）
# ==================================
class SiamTwoStage(nn.Module):
    """
    接在 FPN 输出后，执行：
      (1) DSRPN 生成 proposals（含语义先验 obj，隐式“dual semantic”）
      (2) 模板 RoI -> Query 门控 -> 门控后的 P3/P4/P5
      (3) ROIAlign + VersatileRCNNHead 三分支评分 + bbox 微调
    前向多两个参数：
      template_boxes: list[Tensor(N_i,4)]   模板框（像素坐标，训练=GT或上帧GT；推理=上帧预测）
      img_shapes:     list[(H,W)]           原图尺寸
    """
    def __init__(self, nc=1, ch=(256, 512, 1024), rpn_topk=1000, head_topk=300, roi=7, mode="scale"):
        super().__init__()
        self.strides = (8,16,32)
        self.rpn = DSRPN(ch, strides=self.strides, topk=rpn_topk)
        self.qg3 = QueryConvGate(ch[0], reduce=16, mode=mode)
        self.qg4 = QueryConvGate(ch[1], reduce=16, mode=mode)
        self.qg5 = QueryConvGate(ch[2], reduce=16, mode=mode)
        self.head_topk = head_topk
        self.roi = roi
        self.vr = VersatileRCNNHead(c=ch[0], roi=roi, num_classes=nc)  # 以 P3 做 ROI（也可做多层融合）

    def _roi_align_p3(self, p3, boxes):
        # boxes: [B,K,4] (像素坐标)
        B, K = boxes.shape[:2]
        rois = []
        for b in range(B):
            if K == 0:
                rois.append(torch.tensor([b,0.,0.,1.,1.], device=p3.device).view(1,5))
            else:
                idx = torch.full((K,1), b, device=p3.device)
                rois.append(torch.cat([idx, boxes[b]], 1))
        rois = torch.cat(rois, 0)  # [B*K,5]
        feat = roi_align(p3, rois, output_size=self.roi, spatial_scale=1.0/8, aligned=True)
        return feat  # [B*K, C, roi, roi]

    def forward(self, feats, template_boxes=None, img_shapes=None):
        """
        feats: [P3,P4,P5]
        """
        p3, p4, p5 = feats
        B = p3.size(0)
        # === (1) RPN proposals ===
        props, scores, lvls = self.rpn([p3,p4,p5], img_shapes)     # [B,K,4], [B,K]
        # Top-M 送入 VR 头
        K = min(self.head_topk, props.shape[1])
        scores_t, idx = torch.topk(scores, K, dim=1)
        props_t = torch.gather(props, 1, idx.unsqueeze(-1).expand(-1,-1,4))

        # === (2) 模板 -> Query 门控 ===
        if (template_boxes is not None) and (img_shapes is not None):
            # P3/P4/P5 的模板 RoI（各取 1 个）
            tpl3 = self._template_roi(p3, template_boxes, img_shapes, stride=8)
            tpl4 = self._template_roi(p4, template_boxes, img_shapes, stride=16)
            tpl5 = self._template_roi(p5, template_boxes, img_shapes, stride=32)
            p3 = self.qg3(p3, tpl3); p4 = self.qg4(p4, tpl4); p5 = self.qg5(p5, tpl5)
            gate_weight = F.adaptive_avg_pool2d(tpl3, 1)  # 用 P3 门控幅度作为背景抑制 proxy
        else:
            gate_weight = None

        # === (3) ROIAlign + VR-Head ===
        roi_s = self._roi_align_p3(p3, props_t)            # [B*K,C,roi,roi]
        if (template_boxes is not None) and (img_shapes is not None):
            tpl3 = self._template_roi(p3, template_boxes, img_shapes, stride=8)
            roi_t = tpl3.repeat_interleave(K, dim=0)       # [B*K,C,roi,roi]
        else:
            roi_t = torch.zeros_like(roi_s)

        fused, sem, sim, bkg, delta = self.vr(roi_s, roi_t, gate_weight)  # [B*K] x3, [B*K,4]
        # 这里保持与 Ultralytics 的输出风格一致：返回一个 dict，Trainer 可忽略或你自定义 Loss 使用
        return {
            "proposals": props_t,      # [B,K,4]
            "score_rpn": scores_t,     # [B,K]
            "score_fused": fused.view(B, K),
            "score_sem": sem.view(B, K),
            "score_sim": sim.view(B, K),
            "score_bkg": bkg.view(B, K),
            "delta": delta.view(B, K, 4)  # 二阶段微调量（可选参与loss）
        }

    def _template_roi(self, feat, boxes, img_shapes, stride=8):
        B, C, H, W = feat.shape
        rois = []
        for b in range(B):
            if (boxes is None) or (boxes[b] is None) or (boxes[b].numel()==0):
                rois.append(torch.tensor([b,0.,0.,img_shapes[b][1]-1.,img_shapes[b][0]-1.], device=feat.device))
            else:
                rois.append(torch.cat([torch.tensor([b], device=feat.device), boxes[b][0].to(feat.device)], 0))
        rois = torch.stack(rois,0)
        return roi_align(feat, rois, output_size=self.roi, spatial_scale=1.0/stride, aligned=True)
