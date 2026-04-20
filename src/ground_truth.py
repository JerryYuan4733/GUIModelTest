"""
Ground Truth 标注数据模块

所有 bbox 均使用 **相对坐标 0-1000**（与模型输出格式一致），
便于跨图片尺寸复用。评估时由评估器按 COORDINATE_SCALE 转换为像素。

bbox 格式：(x1, y1, x2, y2)  左上-右下
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GTItem:
    """单个标注目标"""
    label: str                  # 目标名称（用于对齐预测）
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in 0-1000 space

    @property
    def center(self) -> tuple[int, int]:
        """返回 bbox 中心点（0-1000 空间）"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


# ============================================================================
# Ground Truth 数据集（初始近似值，运行 scripts/visualize_gt.py 可视化后再微调）
# ============================================================================

# 左侧搜索结果列表的 avatar 布局参数（0-1000 空间）
# 基于多模型 POS_002 输出共识标定
_ROW_STEP = 66          # 行间距（观测 seed-1-6-vision 实际步长 ~66）
_ROW_Y0 = 150           # 第一行中心 y
_AVATAR_X1 = 2          # avatar 左边界
_AVATAR_X2 = 60         # avatar 右边界
_AVATAR_HALF_H = 26     # avatar 高度的一半（略放宽容差）


def _row_bbox(index: int) -> tuple[int, int, int, int]:
    """按行号（从 0 开始）生成 avatar bbox"""
    cy = _ROW_Y0 + index * _ROW_STEP
    return (_AVATAR_X1, cy - _AVATAR_HALF_H, _AVATAR_X2, cy + _AVATAR_HALF_H)


# 左侧搜索列表中哪些行是「群组」类型（用户视觉标注基于条目名称前的 👥 两人 icon）
# 索引对应下方 _LEFT_LIST_LABELS 的 0-9
GROUP_TYPE_INDICES: list[int] = [4, 6, 9]
# 说明：
#   index 4 = League of Legends Community Chat (@lolc_chat)
#   index 6 = League Of Legends Iran (@LeagueOfLegends_Iran)
#   index 9 = «League of legends» (@League_of_legends1)


# 左侧搜索列表的 avatar 条目（基于图片实际可见的 10 个 LoL 相关条目）
# 说明：EXPECTED_GROUPS 中 "League off Legend" 与 "League of legend" 实为同一行
# （前者疑似 @handle 误解析），图片中只有 10 个 LoL 相关行可见
_LEFT_LIST_LABELS = [
    "League of Legends Community",
    "Starky | League of Legends + IRL",
    "League of Legends",
    "League of Legends: Wild Rift",
    "League of Legends Community Chat",
    "League of Legends News",
    "League Of Legends Iran",
    "League of legend",
    "League of legends",
    "«League of legends»",
]


# ============================================================================
# 按 test_id 索引的 GT
# ============================================================================
GROUND_TRUTH: dict[str, list[GTItem]] = {
    # POS_001a: 右侧信息面板的频道大头像（紫色圆形，位于订阅数上方）
    # 基于 seed-1-6-vision 点击 (925, 95) → 像素 (1184, 61)
    "POS_001a": [
        GTItem(
            label="League of Legends Community (right panel)",
            bbox=(880, 55, 970, 145),
        ),
    ],
    # POS_001b: 左侧搜索结果列表第一项的头像
    # 与 POS_002 第 0 行同一目标
    "POS_001b": [
        GTItem(
            label="League of Legends Community (left list row 1)",
            bbox=_row_bbox(0),
        ),
    ],
    # POS_002: 左侧 10 个 LoL 相关条目头像（群组+频道+bot混合）
    "POS_002": [
        GTItem(label=label, bbox=_row_bbox(i))
        for i, label in enumerate(_LEFT_LIST_LABELS)
    ],
    # POS_003: 仅「群组」类型条目（用户标注：row 4, 6, 9）
    # row 4: League of Legends Community Chat (@lolc_chat)
    # row 6: League Of Legends Iran (@LeagueOfLegends_Iran)
    # row 9: «League of legends» (@League_of_legends1)
    "POS_003": [
        GTItem(label=_LEFT_LIST_LABELS[i], bbox=_row_bbox(i))
        for i in GROUP_TYPE_INDICES
    ],
    # POS_004: 顶部搜索输入框（image2.png, 原分辨率 1920×960）
    # 用户在 image2.png 中用红框标注的 Search 输入框区域
    # 红框像素范围 ≈ (100, 55) - (1870, 170)
    # 换算到 0-1000 相对坐标：
    #   x1 = 100/1920*1000  ≈ 52
    #   y1 =  55/960 *1000  ≈ 57
    #   x2 = 1870/1920*1000 ≈ 974
    #   y2 = 170/960 *1000  ≈ 177
    # 该 bbox 覆盖整个可点击输入区域，点击任一位置均视作命中
    "POS_004": [
        GTItem(
            label="Telegram Search Input Box",
            bbox=(52, 57, 974, 177),
        ),
    ],
}


def get_ground_truth(test_id: str) -> list[GTItem] | None:
    """获取指定测试用例的 GT 标注"""
    return GROUND_TRUTH.get(test_id)
