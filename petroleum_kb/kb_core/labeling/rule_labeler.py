from __future__ import annotations

from .taxonomy import ENTITY_LEXICON, FILE_TASK_HINTS, METHOD_LEXICON, PATH_GROUP_HINTS


def build_rule_labels(
    chunk_text: str,
    source_path: str,
    file_name: str,
    *,
    source_group_hint: str | None = None,
    task_group_hint: str | None = None,
) -> dict:
    """
    Build hierarchical labels through rule anchors.

    Rule anchors come from:
    1. path rules
    2. file-name rules
    3. body keywords
    4. petroleum term lexicons
    """

    source_group = source_group_hint or detect_source_group(source_path)
    task_group = task_group_hint or detect_task_group(chunk_text, file_name)
    entity_tags = collect_tags(chunk_text, ENTITY_LEXICON)
    method_tags = collect_tags(chunk_text, METHOD_LEXICON)
    evidence_type = detect_evidence_type(chunk_text, source_group)
    confidence = estimate_confidence(source_group, task_group, entity_tags, method_tags, evidence_type)
    return {
        "source_group": source_group,
        "task_group": task_group,
        "entity_tags": entity_tags,
        "method_tags": method_tags,
        "evidence_type": evidence_type,
        "confidence": confidence,
    }


def detect_source_group(source_path: str) -> str:
    lower_path = source_path.lower()
    for hint, label in PATH_GROUP_HINTS.items():
        if hint in lower_path:
            return label
    return "未分组"


def detect_task_group(chunk_text: str, file_name: str) -> str:
    searchable = f"{file_name.lower()} {chunk_text.lower()}"
    for keyword, task_group in FILE_TASK_HINTS.items():
        if keyword in searchable:
            return task_group
    if any(word in searchable for word in ["组成", "分子式", "芳香烃", "饱和烃"]):
        return "分子组成推断"
    if any(word in searchable for word in ["结构", "dbe", "质谱"]):
        return "结构解析"
    if any(word in searchable for word in ["api", "密度", "黏度"]):
        return "样品表征"
    if any(word in searchable for word in ["操作步骤", "操作建议", "注意事项"]):
        return "仪器操作"
    return "未分组"


def collect_tags(chunk_text: str, lexicon: dict[str, list[str]]) -> list[str]:
    searchable = chunk_text.lower()
    tags: list[str] = []
    for canonical, keywords in lexicon.items():
        if any(keyword.lower() in searchable for keyword in keywords):
            tags.append(canonical)
    return sorted(tags)


def detect_evidence_type(chunk_text: str, source_group: str) -> str:
    searchable = chunk_text.lower()
    if source_group == "标准":
        return "标准规定"
    if any(token in searchable for token in ["定义", "是指", "可定义为"]):
        return "定义性知识"
    if any(token in searchable for token in ["实验结果", "结果表明", "测得", "observed"]):
        return "实验结论"
    if any(token in searchable for token in ["建议", "推荐", "应当", "注意"]):
        return "操作建议"
    if any(token in searchable for token in ["通常", "一般认为", "经验上"]):
        return "经验描述"
    return "未分类"


def estimate_confidence(
    source_group: str,
    task_group: str,
    entity_tags: list[str],
    method_tags: list[str],
    evidence_type: str,
) -> float:
    score = 0.20
    if source_group != "未分组":
        score += 0.20
    if task_group != "未分组":
        score += 0.20
    score += min(0.20, len(entity_tags) * 0.05)
    score += min(0.10, len(method_tags) * 0.05)
    if evidence_type != "未分类":
        score += 0.10
    return round(min(score, 0.95), 2)
