from __future__ import annotations

"""Runtime-authoritative taxonomy for the layered petroleum knowledge base.

This module defines the first-version controlled vocabulary used by:

- rule-based labeling
- metadata construction during ingest
- tag-aware retrieval and filtering

The current version intentionally keeps the taxonomy compact and editable.
Later versions can extend this file with:

- domain-specific aliases
- hierarchical parent/child tag mappings
- ontology IDs for knowledge graph construction
"""

SOURCE_GROUPS = [
    "文献",
    "标准",
    "手册",
    "报告",
    "数据说明",
    "内部笔记",
    "未分组",
]

TASK_GROUPS = [
    "样品表征",
    "分子组成推断",
    "结构解析",
    "性质预测",
    "工艺机理",
    "仪器操作",
    "数据处理",
    "方法评价",
    "未分组",
]

EVIDENCE_TYPES = [
    "定义性知识",
    "标准规定",
    "实验结论",
    "经验描述",
    "操作建议",
    "未分类",
]

# Common petroleum-domain entities. Keys are normalized canonical labels and
# values are keyword/alias lists used by the rule labeler.
ENTITY_LEXICON = {
    "原油": ["原油", "crude oil", "feedstock"],
    "馏分": ["馏分", "petroleum fraction", "减压瓦斯油", "柴油馏分"],
    "饱和烃": ["饱和烃", "saturates", "正构烷烃", "异构烷烃"],
    "芳香烃": ["芳香烃", "aromatics", "苯系物"],
    "含硫化合物": ["含硫化合物", "thiophenic", "sulfidic", "硫醚"],
    "GC": ["gc", "气相色谱", "保留时间", "峰面积"],
    "GC×GC": ["gc×gc", "gcxgc", "二维气相色谱"],
    "FT-ICR MS": ["ft-icr", "orbitrap", "高分辨质谱"],
    "密度": ["密度", "density"],
    "API": ["api", "api gravity"],
    "DBE": ["dbe", "double bond equivalent"],
    "碳数分布": ["碳数分布", "carbon number"],
}

METHOD_LEXICON = {
    "GC": ["gc", "气相色谱", "保留时间", "fid", "ms"],
    "GC×GC": ["gc×gc", "gcxgc", "二维气相色谱"],
    "FT-ICR MS": ["ft-icr", "orbitrap", "高分辨质谱"],
    "SARA": ["sara", "饱和烃", "芳香烃", "胶质", "沥青质"],
    "NMR": ["nmr", "核磁"],
    "元素分析": ["元素分析", "元素组成", "ea"],
}

# Path-level prior hints. These are intentionally weak priors and should be
# treated as anchors, not absolute truth.
PATH_GROUP_HINTS = {
    "papers": "文献",
    "literature": "文献",
    "standards": "标准",
    "manuals": "手册",
    "reports": "报告",
    "notes": "内部笔记",
    "data": "数据说明",
}

FILE_TASK_HINTS = {
    "gc": "数据处理",
    "色谱": "数据处理",
    "质谱": "结构解析",
    "api": "样品表征",
    "密度": "性质预测",
    "机理": "工艺机理",
    "仪器": "仪器操作",
    "方法": "方法评价",
}
