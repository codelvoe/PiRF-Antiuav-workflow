# # # ultralytics/nn/modules/qsiam.py
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torchvision.ops import roi_align

# # # ---------- 1) 轻量查询门：模板 RoI -> 通道权重 -> 门控搜索特征 ----------
# # class QueryConvGate(nn.Module):
# #     """
# #     模板 RoI 特征 -> 通道权重向量 -> 对搜索特征逐通道放大/抑制
# #     mode='scale': 仅通道缩放；mode='dwconv': 叠加深度可分卷积调制
# #     """
# #     def __init__(self, c, reduce=16, mode="scale"):
# #         super().__init__()
# #         hid = max(c // reduce, 8)
# #         self.fc1 = nn.Linear(c, hid)
# #         self.fc2 = nn.Linear(hid, c)
# #         self.mode = mode
# #         if mode == "dwconv":
# #             self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
# #             nn.init.constant_(self.dw.weight, 0.)

# #     def forward(self, x, tpl_roi_feat):
# #         # x: [B,C,H,W] 搜索特征; tpl_roi_feat: [B,C,S,S] 模板 RoI 特征
# #         w = F.adaptive_avg_pool2d(tpl_roi_feat, 1).flatten(1)        # [B,C]
# #         w = torch.sigmoid(self.fc2(F.relu(self.fc1(w)))).view(x.size(0), x.size(1), 1, 1)
# #         return x * w if self.mode == "scale" else (self.dw(x) + x * w)

# # # ---------- 2) 单尺度 ROIAlign：模板框坐标是原图像素坐标 ----------
# # def roi_pool_one_level(feat, boxes, hw, out_size=7, stride=8):
# #     """
# #     feat   : [B,C,H,W]  某一尺度特征 (例如 P3)
# #     boxes  : list[Tensor(N_i,4)]  每张图的模板框 x1y1x2y2（原图坐标）
# #     hw     : list[(H,W)] 原图尺寸
# #     return : [B,C,out_size,out_size] 每张图取一个模板 RoI（没框则用整图兜底）
# #     """
# #     B, C, _, _ = feat.shape
# #     rois = []
# #     for b in range(B):
# #         if (not boxes) or (boxes[b] is None) or (boxes[b].numel() == 0):
# #             H, W = hw[b]
# #             rois.append(torch.tensor([b, 0., 0., W - 1., H - 1.], device=feat.device))
# #         else:
# #             # 这里只取第一个框，你可以按面积/置信度来选
# #             rois.append(torch.cat([torch.tensor([b], device=feat.device),
# #                                    boxes[b][0].to(feat.device)], 0))
# #     rois = torch.stack(rois, 0)  # [B,5]
# #     pooled = roi_align(feat, rois, output_size=out_size, spatial_scale=1.0/stride, aligned=True)
# #     return pooled  # [B,C,S,S]

# # # ---------- 3) SiamDetect：包装原生 Detect，支持孪生门控 ----------
# # class SiamDetect(nn.Module):
# #     """
# #     与原生 Detect 用法保持兼容：
# #         - 构造: SiamDetect(nc, ch)
# #         - 前向: forward([P3,P4,P5]) 返回与 Detect 相同的输出格式
# #     额外增加:
# #         - set_template(template_boxes, img_shapes): Trainer 在 preprocess_batch 里写入模板
# #         - forward(..., template_boxes, img_shapes): 也支持显式传入模板
# #     """
# #     def __init__(self, nc=80, ch=()):
# #         super().__init__()
# #         from ultralytics.nn.modules.head import Detect
# #         self.detect = Detect(nc, ch=ch)

# #         c3, c4, c5 = ch  # 例如 (256, 512, 1024)
# #         self.qg3 = QueryConvGate(c3, reduce=16, mode="scale")
# #         self.qg4 = QueryConvGate(c4, reduce=16, mode="scale")
# #         self.qg5 = QueryConvGate(c5, reduce=16, mode="scale")

# #         # 如果没显式传入，我们用缓存的模板信息
# #         self._tmpl_boxes = None
# #         self._img_shapes = None

# #         # Detect 在 build 阶段会写入 stride，这里给个兜底
# #         self.register_buffer("_stride_p3", torch.tensor(8.0))
# #         self.register_buffer("_stride_p4", torch.tensor(16.0))
# #         self.register_buffer("_stride_p5", torch.tensor(32.0))

# #     # ------- 让 Trainer 能写入模板 --------
# #     def set_template(self, template_boxes, img_shapes):
# #         """
# #         template_boxes: list[Tensor(N_i,4)] 每张图的模板框（像素坐标）
# #         img_shapes    : list[(H,W)]
# #         """
# #         self._tmpl_boxes = template_boxes
# #         self._img_shapes = img_shapes

# #     @property
# #     def stride(self):
# #         if hasattr(self.detect, "stride") and len(self.detect.stride):
# #             return self.detect.stride
# #         return torch.tensor([float(self._stride_p3), float(self._stride_p4), float(self._stride_p5)])

# #     def forward(self, x, template_boxes=None, img_shapes=None):
# #         """
# #         x: list[P3,P4,P5]
# #         如果不传 template_boxes/img_shapes，则使用 set_template 写入的缓存。
# #         """
# #         assert isinstance(x, (list, tuple)) and len(x) == 3, "SiamDetect expects [P3,P4,P5]"
# #         p3, p4, p5 = x

# #         # 选择使用入参还是缓存
# #         template_boxes = template_boxes if template_boxes is not None else self._tmpl_boxes
# #         img_shapes     = img_shapes     if img_shapes     is not None else self._img_shapes

# #         if (template_boxes is not None) and (img_shapes is not None):
# #             s3, s4, s5 = [float(v) for v in self.stride]
# #             tpl3 = roi_pool_one_level(p3, template_boxes, img_shapes, out_size=7, stride=int(s3))
# #             tpl4 = roi_pool_one_level(p4, template_boxes, img_shapes, out_size=7, stride=int(s4))
# #             tpl5 = roi_pool_one_level(p5, template_boxes, img_shapes, out_size=7, stride=int(s5))
# #             p3 = self.qg3(p3, tpl3)
# #             p4 = self.qg4(p4, tpl4)
# #             p5 = self.qg5(p5, tpl5)

# #         # 调用原生 Detect，输出完全一致，loss / NMS / 导出都不用改
# #         return self.detect([p3, p4, p5])

# # ultralytics/nn/modules/qsiam.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.ops import roi_align

# # ---------- 1) 轻量查询门：模板 RoI -> 通道权重 -> 门控搜索特征 ----------
# class QueryConvGate(nn.Module):
#     """
#     模板 RoI 特征 -> 通道权重向量 -> 对搜索特征逐通道放大/抑制
#     mode='scale': 仅通道缩放；mode='dwconv': 叠加深度可分卷积调制
#     """
#     def __init__(self, c, reduce=16, mode="scale"):
#         super().__init__()
#         hid = max(c // reduce, 8)
#         self.fc1 = nn.Linear(c, hid)
#         self.fc2 = nn.Linear(hid, c)
#         self.mode = mode
#         if mode == "dwconv":
#             self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
#             nn.init.constant_(self.dw.weight, 0.)

#     def forward(self, x, tpl_roi_feat):
#         # x: [B,C,H,W] 搜索特征; tpl_roi_feat: [B,C,S,S] 模板 RoI 特征
#         w = F.adaptive_avg_pool2d(tpl_roi_feat, 1).flatten(1)  # [B,C]
#         w = torch.sigmoid(self.fc2(F.relu(self.fc1(w)))).view(x.size(0), x.size(1), 1, 1)
#         return x * w if self.mode == "scale" else (self.dw(x) + x * w)


# # ---------- 2) 单尺度 ROIAlign：模板框坐标是原图像素坐标 ----------
# def roi_pool_one_level(feat, boxes, hw, out_size=7, stride=8):
#     """
#     feat   : [B,C,H,W]  某一尺度特征 (例如 P3)
#     boxes  : list[Tensor(N_i,4)]  每张图的模板框 x1y1x2y2（原图坐标）
#     hw     : list[(H,W)] 原图尺寸
#     return : [B,C,out_size,out_size] 每张图取一个模板 RoI（没框则用整图兜底）
#     """
#     B, C, _, _ = feat.shape
#     rois = []
#     for b in range(B):
#         if (not boxes) or (boxes[b] is None) or (boxes[b].numel() == 0):
#             H, W = hw[b]
#             rois.append(torch.tensor([b, 0., 0., W - 1., H - 1.], device=feat.device))
#         else:
#             # 这里只取第一个框，你可以按面积/置信度来选
#             rois.append(torch.cat([torch.tensor([b], device=feat.device),
#                                    boxes[b][0].to(feat.device)], 0))
#     rois = torch.stack(rois, 0)  # [B,5]
#     pooled = roi_align(feat, rois, output_size=out_size, spatial_scale=1.0/stride, aligned=True)
#     return pooled  # [B,C,S,S]


# # ---------- 3) SiamDetect：包装原生 Detect，支持孪生门控 ----------
# class SiamDetect(nn.Module):
#     """
#     与原生 Detect 用法保持兼容：
#         - 构造: SiamDetect(nc, ch)
#         - 前向: forward([P3,P4,P5]) 返回与 Detect 相同的输出格式
#     额外增加:
#         - set_template(template_boxes, img_shapes): Trainer 在 preprocess_batch 里写入模板
#         - forward(..., template_boxes, img_shapes): 也支持显式传入模板
#     """
#     def __init__(self, nc=80, ch=()):
#         super().__init__()
#         from ultralytics.nn.modules.head import Detect
#         self.detect = Detect(nc, ch=ch)

#         c3, c4, c5 = ch  # 例如 (256, 512, 1024)
#         self.qg3 = QueryConvGate(c3, reduce=16, mode="scale")
#         self.qg4 = QueryConvGate(c4, reduce=16, mode="scale")
#         self.qg5 = QueryConvGate(c5, reduce=16, mode="scale")

#         # 如果没显式传入，我们用缓存的模板信息
#         self._tmpl_boxes = None
#         self._img_shapes = None

#         # Detect 在 build 阶段会写入 stride，这里给个兜底
#         self.register_buffer("_stride_p3", torch.tensor(8.0))
#         self.register_buffer("_stride_p4", torch.tensor(16.0))
#         self.register_buffer("_stride_p5", torch.tensor(32.0))

#     # ------- 让 Trainer 能写入模板 --------
#     def set_template(self, template_boxes, img_shapes):
#         """
#         template_boxes: list[Tensor(N_i,4)] 每张图的模板框（像素坐标）
#         img_shapes    : list[(H,W)]
#         """
#         self._tmpl_boxes = template_boxes
#         self._img_shapes = img_shapes

#     # ========== 这些属性是给 loss / 导出 用的壳属性，全部转发到内部 Detect ==========
#     @property
#     def nc(self):
#         """类别数，供 v8DetectionLoss 使用。"""
#         return self.detect.nc

#     @property
#     def reg_max(self):
#         """DFL 的 reg_max 参数。"""
#         return self.detect.reg_max

#     @property
#     def no(self):
#         """每个锚点输出维度 = nc + reg_max * 4。"""
#         return self.detect.no

#     @property
#     def dfl(self):
#         """DFL 模块（边框回归用）。"""
#         return self.detect.dfl

#     @property
#     def box(self):
#         """负责回归的卷积层列表。"""
#         return self.detect.box

#     @property
#     def cls(self):
#         """负责分类的卷积层列表。"""
#         return self.detect.cls

#     @property
#     def nl(self):
#         """检测层数量（通常为 3: P3/P4/P5）。"""
#         return self.detect.nl

#     @property
#     def anchors(self):
#         """anchor 配置，部分工具/导出可能会用到。"""
#         return self.detect.anchors

#     @property
#     def stride(self):
#         """步长，loss / decode 都会用到。"""
#         if hasattr(self.detect, "stride") and len(self.detect.stride):
#             return self.detect.stride
#         return torch.tensor([float(self._stride_p3), float(self._stride_p4), float(self._stride_p5)])

#     # ---------- 前向 ----------
#     def forward(self, x, template_boxes=None, img_shapes=None):
#         """
#         x: list[P3,P4,P5]
#         如果不传 template_boxes/img_shapes，则使用 set_template 写入的缓存。
#         """
#         assert isinstance(x, (list, tuple)) and len(x) == 3, "SiamDetect expects [P3,P4,P5]"
#         p3, p4, p5 = x

#         # 选择使用入参还是缓存
#         template_boxes = template_boxes if template_boxes is not None else self._tmpl_boxes
#         img_shapes     = img_shapes     if img_shapes     is not None else self._img_shapes

#         if (template_boxes is not None) and (img_shapes is not None):
#             s3, s4, s5 = [float(v) for v in self.stride]
#             tpl3 = roi_pool_one_level(p3, template_boxes, img_shapes, out_size=7, stride=int(s3))
#             tpl4 = roi_pool_one_level(p4, template_boxes, img_shapes, out_size=7, stride=int(s4))
#             tpl5 = roi_pool_one_level(p5, template_boxes, img_shapes, out_size=7, stride=int(s5))
#             p3 = self.qg3(p3, tpl3)
#             p4 = self.qg4(p4, tpl4)
#             p5 = self.qg5(p5, tpl5)

#         # 调用原生 Detect，输出完全一致，loss / NMS / 导出都不用改
#         return self.detect([p3, p4, p5])


# # ultralytics/nn/modules/qsiam.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.ops import roi_align

# from ultralytics.nn.modules.head import Detect  # 直接继承原生 Detect 头


# # ---------- 1) 轻量查询门：模板 RoI -> 通道权重 -> 门控搜索特征 ----------
# class QueryConvGate(nn.Module):
#     """
#     模板 RoI 特征 -> 通道权重向量 -> 对搜索特征逐通道放大/抑制
#     mode='scale': 仅通道缩放；mode='dwconv': 叠加深度可分卷积调制
#     """
#     def __init__(self, c, reduce=16, mode="scale", gate_scale=0.1):
#         super().__init__()
#         hid = max(c // reduce, 8)
#         self.fc1 = nn.Linear(c, hid)
#         self.fc2 = nn.Linear(hid, c)
#         self.mode = mode
#         self.gate_scale = gate_scale  # 控制门控强度，初期建议很小（0.1）

#         if mode == "dwconv":
#             self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
#             nn.init.constant_(self.dw.weight, 0.)

#     def forward(self, x, tpl_roi_feat):
#         # x: [B,C,H,W] 搜索特征; tpl_roi_feat: [B,C,S,S] 模板 RoI 特征
#         w = F.adaptive_avg_pool2d(tpl_roi_feat, 1).flatten(1)  # [B,C]
#         w = torch.sigmoid(self.fc2(F.relu(self.fc1(w)))).view(x.size(0), x.size(1), 1, 1)
#         # 为避免训练初期把特征压死，把缩放限制在 [0.9,1.1] 左右
#         w = 1.0 + self.gate_scale * (w - 1.0)
#         return x * w if self.mode == "scale" else (self.dw(x) + x * w)


# # ---------- 2) 单尺度 ROIAlign：模板框坐标是原图像素坐标 ----------
# def roi_pool_one_level(feat, boxes, hw, out_size=7, stride=8):
#     """
#     feat   : [B,C,H,W]  某一尺度特征 (例如 P3)
#     boxes  : list[Tensor(N_i,4)]  每张图的模板框 x1y1x2y2（当前输入图的像素坐标）
#     hw     : list[(H,W)] 当前输入图尺寸
#     return : [B,C,out_size,out_size] 每张图取一个模板 RoI（没框则用整图兜底）
#     """
#     B, C, _, _ = feat.shape
#     rois = []
#     for b in range(B):
#         if (not boxes) or (boxes[b] is None) or (boxes[b].numel() == 0):
#             H, W = hw[b]
#             rois.append(torch.tensor([b, 0., 0., W - 1., H - 1.], device=feat.device))
#         else:
#             # 这里只取第一个框，你可以按面积/置信度来选
#             rois.append(torch.cat([torch.tensor([b], device=feat.device),
#                                    boxes[b][0].to(feat.device)], 0))
#     rois = torch.stack(rois, 0)  # [B,5]
#     # 注意 spatial_scale = 1/stride
#     pooled = roi_align(feat, rois, output_size=out_size, spatial_scale=1.0 / float(stride), aligned=True)
#     return pooled  # [B,C,S,S]


# # ---------- 3) SiamDetect：继承 Detect，保持所有头部行为完全一致 ----------
# class SiamDetect(Detect):
#     """
#     直接继承 Detect：
#       - 所有 init_criterion / stride 计算 / bias_init / AMP 等逻辑全部保持原样，
#         框架里凡是 isinstance(m, Detect) 的地方都把它当 Detect 用。
#       - 仅在 forward 时（可选）对 P3/P4/P5 施加 QueryConvGate 门控。

#     重要：
#       - 目前默认“门控完全关闭”（只是预留接口），确保训练行为先恢复到接近原始 YOLO11。
#       - 等验证原生指标正常后，再一点点打开门控逻辑。
#     """
#     def __init__(self, nc=80, ch=(), *args, **kwargs):
#         # 调用原生 Detect 构造，nc 与 ch 由 parse_model 提供
#         super().__init__(nc=nc, ch=ch, *args, **kwargs)

#         # ch: [C_P3, C_P4, C_P5]
#         if len(ch) != 3:
#             # 保底，某些配置可能不完全一样，那就不做门控
#             self.qg3 = self.qg4 = self.qg5 = None
#         else:
#             c3, c4, c5 = ch
#             self.qg3 = QueryConvGate(c3, reduce=16, mode="scale", gate_scale=0.1)
#             self.qg4 = QueryConvGate(c4, reduce=16, mode="scale", gate_scale=0.1)
#             self.qg5 = QueryConvGate(c5, reduce=16, mode="scale", gate_scale=0.1)

#         # 模板信息缓存（由 Trainer 通过 set_template 写入）
#         self._tmpl_boxes = None   # list[Tensor(N_i,4)]，当前 batch 的模板框（像素坐标）
#         self._img_shapes = None   # list[(H,W)]，当前输入尺寸

#     # ------- 让 Trainer 能写入模板 --------
#     def set_template(self, template_boxes, img_shapes):
#         """
#         template_boxes: list[Tensor(N_i,4)] 每张图的模板框（像素坐标，基于当前 batch["img"] 尺寸）
#         img_shapes    : list[(H,W)]
#         """
#         self._tmpl_boxes = template_boxes
#         self._img_shapes = img_shapes

#     # ---------- 前向 ----------
#     def forward(self, x, template_boxes=None, img_shapes=None):
#         """
#         x: list[P3,P4,P5] （因为 YAML 中 from = [P3,P4,P5]）
#         template_boxes/img_shapes: 可选，若 None 则使用 set_template 写入的缓存。
#         """

#         # 兼容：如果 x 不是 list（例如某些导出/特殊路径），直接走父类 Detect.forward
#         if not isinstance(x, (list, tuple)):
#             return super().forward(x)

#         assert len(x) == 3, "SiamDetect expects 3 feature maps [P3,P4,P5]"
#         p3, p4, p5 = x

#         # （目前默认关闭门控，只走 super()，先验证训练/评估链路完全正常）
#         # 如果你已经确认 SiamDetect 与原生 Detect 效果一致，再打开下面的门控代码块。

#         # ========= 现在：完全禁用门控 =========
#         return super().forward([p3, p4, p5])

#         # ========= 将来：你可以按下面的方式一点点打开门控 =========
#         # template_boxes = template_boxes if template_boxes is not None else self._tmpl_boxes
#         # img_shapes     = img_shapes     if img_shapes     is not None else self._img_shapes
#         #
#         # if (template_boxes is not None) and (img_shapes is not None) and self.qg3 is not None:
#         #     # Detect.stride 在 DetectionModel.__init__ 中会正确设置，这里直接用
#         #     s3, s4, s5 = [float(v) for v in self.stride]
#         #     s3 = max(s3, 1.0); s4 = max(s4, 1.0); s5 = max(s5, 1.0)
#         #     tpl3 = roi_pool_one_level(p3, template_boxes, img_shapes, out_size=7, stride=int(s3))
#         #     tpl4 = roi_pool_one_level(p4, template_boxes, img_shapes, out_size=7, stride=int(s4))
#         #     tpl5 = roi_pool_one_level(p5, template_boxes, img_shapes, out_size=7, stride=int(s5))
#         #     p3 = self.qg3(p3, tpl3)
#         #     p4 = self.qg4(p4, tpl4)
#         #     p5 = self.qg5(p5, tpl5)
#         #
#         # return super().forward([p3, p4, p5])


# ultralytics/nn/modules/qsiam.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from ultralytics.nn.modules.head import Detect  # 直接继承原生 Detect 头


# ---------- 1) 轻量查询门：模板 RoI -> 通道权重 -> 门控搜索特征 ----------
class QueryConvGate(nn.Module):
    """
    模板 RoI 特征 -> 通道权重向量 -> 对搜索特征逐通道放大/抑制
    mode='scale': 仅通道缩放；mode='dwconv': 叠加深度可分卷积调制
    """
    def __init__(self, c, reduce=16, mode="scale", gate_scale=0.05):
        super().__init__()
        hid = max(c // reduce, 8)
        self.fc1 = nn.Linear(c, hid)
        self.fc2 = nn.Linear(hid, c)
        self.mode = mode
        self.gate_scale = gate_scale  # 控制门控强度，初期建议很小（0.05）

        if mode == "dwconv":
            self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
            nn.init.constant_(self.dw.weight, 0.)

    def forward(self, x, tpl_roi_feat):
        # x: [B,C,H,W] 搜索特征; tpl_roi_feat: [B,C,S,S] 模板 RoI 特征
        w = F.adaptive_avg_pool2d(tpl_roi_feat, 1).flatten(1)  # [B,C]
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w)))).view(x.size(0), x.size(1), 1, 1)
        # 为避免训练初期把特征压死，把缩放限制在 [1-g,1+g] 左右
        w = 1.0 + self.gate_scale * (w - 1.0)
        return x * w if self.mode == "scale" else (self.dw(x) + x * w)


# ---------- 2) 单尺度 ROIAlign：模板框坐标是原图像素坐标 ----------
def roi_pool_one_level(feat, boxes, hw, out_size=7, stride=8):
    """
    feat   : [B,C,H,W]  某一尺度特征 (例如 P3)
    boxes  : list[Tensor(N_i,4)]  每张图的模板框 x1y1x2y2（当前输入图的像素坐标）
    hw     : list[(H,W)] 当前输入图尺寸
    return : [B,C,out_size,out_size] 每张图取一个模板 RoI（没框则用整图兜底）
    """
    B, C, _, _ = feat.shape
    rois = []
    for b in range(B):
        if (not boxes) or (boxes[b] is None) or (boxes[b].numel() == 0):
            H, W = hw[b]
            rois.append(torch.tensor([b, 0., 0., W - 1., H - 1.], device=feat.device))
        else:
            # 这里只取第一个框，你可以按面积/置信度来选
            rois.append(torch.cat([torch.tensor([b], device=feat.device),
                                   boxes[b][0].to(feat.device)], 0))
    rois = torch.stack(rois, 0)  # [B,5]
    # 注意 spatial_scale = 1/stride
    pooled = roi_align(feat, rois, output_size=out_size, spatial_scale=1.0 / float(stride), aligned=True)
    return pooled  # [B,C,S,S]


# ---------- 3) SiamDetect：继承 Detect，保持所有头部行为完全一致 ----------
class SiamDetect(Detect):
    """
    直接继承 Detect：
      - 所有 init_criterion / stride 计算 / bias_init / AMP 等逻辑全部保持原样，
        框架里凡是 isinstance(m, Detect) 的地方都把它当 Detect 用。
      - 在 forward 时对 P3 施加 QueryConvGate 门控（P4/P5 先不动，逐步扩展）。

    使用方式：
      - 训练时在 BaseTrainer/DetectionTrainer 的 preprocess_batch 里，
        根据 batch["bboxes"] 和 batch["img"].shape 计算像素模板框，并调用 head.set_template(...)。
    """
    def __init__(self, nc=80, ch=(), *args, **kwargs):
        # 调用原生 Detect 构造，nc 与 ch 由 parse_model 提供
        super().__init__(nc=nc, ch=ch, *args, **kwargs)

        # ch: [C_P3, C_P4, C_P5]
        if len(ch) >= 1:
            c3 = ch[0]
            # 先只对 P3 做门控，P4/P5 保持不动
            self.qg3 = QueryConvGate(c3, reduce=16, mode="scale", gate_scale=0.05)
        else:
            self.qg3 = None

        # 模板信息缓存（由 Trainer 通过 set_template 写入）
        self._tmpl_boxes = None   # list[Tensor(N_i,4)]，当前 batch 的模板框（像素坐标）
        self._img_shapes = None   # list[(H,W)]，当前输入尺寸（注意应为当前 batch["img"] 的尺寸）

    # ------- 让 Trainer 能写入模板 --------
    def set_template(self, template_boxes, img_shapes):
        """
        template_boxes: list[Tensor(N_i,4)] 每张图的模板框（像素坐标，基于当前 batch["img"] 尺寸）
        img_shapes    : list[(H,W)]
        """
        self._tmpl_boxes = template_boxes
        self._img_shapes = img_shapes

    # ---------- 前向 ----------
    def forward(self, x, template_boxes=None, img_shapes=None):
        """
        x: list[P3,P4,P5] （因为 YAML 中 from = [P3,P4,P5]）
        template_boxes/img_shapes: 可选，若 None 则使用 set_template 写入的缓存。
        """

        # 兼容：如果 x 不是 list（例如某些导出/特殊路径），直接走父类 Detect.forward
        if not isinstance(x, (list, tuple)):
            return super().forward(x)

        assert len(x) == 3, "SiamDetect expects 3 feature maps [P3,P4,P5]"
        feats = list(x)
        p3, p4, p5 = feats

        # 选择模板来源
        template_boxes = template_boxes if template_boxes is not None else self._tmpl_boxes
        img_shapes     = img_shapes     if img_shapes     is not None else self._img_shapes

        # 仅在训练阶段且模板有效时启用 P3 门控；推理阶段先关闭门控，保证评估口径清晰
        # (Self-correction: Enable for training to learn the gating)
        if self.training and (self.qg3 is not None) and (template_boxes is not None) and (img_shapes is not None):
            # Detect.stride 在 DetectionModel.__init__ 中会正确设置，这里直接用
            s3 = float(self.stride[0].item() if isinstance(self.stride, torch.Tensor) else self.stride[0])
            s3 = max(s3, 1.0)
            tpl3 = roi_pool_one_level(p3, template_boxes, img_shapes, out_size=7, stride=int(s3))
            p3 = self.qg3(p3, tpl3)
            feats[0] = p3

        # 仍然调用父类 Detect.forward，保持输出格式不变
        return super().forward(feats)
