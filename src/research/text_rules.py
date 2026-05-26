from __future__ import annotations

from dataclasses import dataclass
import re

from .io_utils import has_url


QUESTION_MARKS = ("?", "？")
ZH_DENY = ("谣言", "假的", "不实", "别信", "辟谣", "并非", "不是", "假消息", "传言")
ZH_QUERY = ("求证", "真的吗", "真的假的", "是否属实", "有依据吗", "有没有", "谁知道")
ZH_EVIDENCE = ("官方", "通报", "公告", "来源", "证据", "依据", "研究", "数据", "链接", "截图", "新闻")
ZH_EMOTION = ("震惊", "可怕", "愤怒", "气死", "害怕", "太惨", "离谱", "天啊", "支持", "恐怖")

EN_DENY = ("fake", "false", "not true", "hoax", "debunk", "rumour", "rumor", "misleading")
EN_QUERY = ("is this true", "really", "any source", "source?", "confirmed", "can anyone verify")
EN_EVIDENCE = ("official", "report", "according", "source", "evidence", "study", "data", "link", "statement")
EN_EMOTION = ("omg", "wow", "terrible", "scary", "awful", "angry", "sad", "horrible", "shocking")
ZH_OFFICIAL = ("官方", "警方", "卫健委", "政府", "央视", "人民日报", "通报", "公告", "辟谣平台")
ZH_EXPERT = ("专家", "教授", "医生", "院士", "研究员", "疾控", "医院")
ZH_MEDIA = ("媒体", "新闻", "记者", "电视台", "报社", "客户端")
ZH_ACTION = ("不要", "请", "建议", "立即", "尽快", "及时", "拨打", "佩戴", "就医", "上报")
ZH_FIVEW = {
    "who": ("谁", "专家", "警方", "部门", "患者", "居民", "医生"),
    "what": ("事件", "情况", "病例", "消息", "通报", "传言"),
    "when": ("今日", "今天", "昨日", "目前", "时间", "截至", "当日"),
    "where": ("在", "于", "地区", "医院", "学校", "社区", "武汉", "北京"),
    "why": ("因为", "由于", "原因", "为何", "因此", "导致"),
    "how": ("如何", "怎么", "方式", "步骤", "方法", "处理"),
}

EN_OFFICIAL = ("official", "government", "police", "ministry", "department", "statement", "agency")
EN_EXPERT = ("expert", "professor", "doctor", "scientist", "researcher", "cdc", "who")
EN_MEDIA = ("news", "reporter", "media", "bbc", "cnn", "times", "press")
EN_ACTION = ("please", "avoid", "do not", "should", "advised", "immediately", "wear", "call", "report")
EN_FIVEW = {
    "who": ("who", "expert", "doctor", "police", "agency", "official"),
    "what": ("what", "case", "incident", "report", "claim", "rumor"),
    "when": ("when", "today", "yesterday", "currently", "as of", "timeline"),
    "where": ("where", "hospital", "school", "city", "community", "at "),
    "why": ("why", "because", "due to", "reason", "cause"),
    "how": ("how", "method", "step", "process", "guidance"),
}

TOKEN_PATTERN = re.compile(r"https?://\S+|www\.\S+|[\w']+|[\u4e00-\u9fff]+", re.UNICODE)


@dataclass(frozen=True)
class ResponseRuleResult:
    label: str
    scores: dict[str, int]


@dataclass(frozen=True)
class InterventionSignalResult:
    stance_type: str
    inferred_correction_signal: str
    publisher_type: str
    publisher_type_source: str
    explicit_correction: bool
    inferred_correction: bool
    is_correction: bool
    correction_type: str | None
    official_flag: int
    evidence_flag: int
    deny_flag: int
    query_flag: int
    emotion_flag: int
    action_flag: int
    fivew_score: int
    scores: dict[str, int]


def _score_by_keywords(text: str, keywords: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def classify_response(text: str | None, language: str) -> ResponseRuleResult:
    content = (text or "").strip()
    lowered = content.lower()
    scores = {
        "deny": 0,
        "query": 0,
        "evidence": 0,
        "emotion": 0,
        "other": 0,
    }

    if language == "zh":
        scores["deny"] += _score_by_keywords(content, ZH_DENY)
        scores["query"] += _score_by_keywords(content, ZH_QUERY)
        scores["evidence"] += _score_by_keywords(content, ZH_EVIDENCE)
        scores["emotion"] += _score_by_keywords(content, ZH_EMOTION)
    else:
        scores["deny"] += _score_by_keywords(lowered, EN_DENY)
        scores["query"] += _score_by_keywords(lowered, EN_QUERY)
        scores["evidence"] += _score_by_keywords(lowered, EN_EVIDENCE)
        scores["emotion"] += _score_by_keywords(lowered, EN_EMOTION)

    if any(mark in content for mark in QUESTION_MARKS):
        scores["query"] += 1
    if has_url(content):
        scores["evidence"] += 2
    if content.count("!") + content.count("！") >= 2:
        scores["emotion"] += 1

    if not any(scores.values()):
        return ResponseRuleResult(label="other", scores=scores)

    priority = ("evidence", "deny", "query", "emotion", "other")
    label = max(priority, key=lambda key: (scores[key], -priority.index(key)))
    return ResponseRuleResult(label=label, scores=scores)


def simple_tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())


def extract_intervention_signals(
    text: str | None,
    language: str,
    *,
    metadata: dict | None = None,
    explicit_publisher_type: str | None = None,
    explicit_is_correction: bool | None = None,
    explicit_correction_type: str | None = None,
) -> InterventionSignalResult:
    content = (text or "").strip()
    metadata = metadata or {}
    response = classify_response(content, language)

    if language == "zh":
        official_flag = int(_score_by_keywords(content, ZH_OFFICIAL) > 0)
        expert_flag = int(_score_by_keywords(content, ZH_EXPERT) > 0)
        media_flag = int(_score_by_keywords(content, ZH_MEDIA) > 0)
        action_flag = int(_score_by_keywords(content, ZH_ACTION) > 0)
        fivew_score = _fivew_score(content, ZH_FIVEW)
    else:
        lowered = content.lower()
        official_flag = int(_score_by_keywords(lowered, EN_OFFICIAL) > 0)
        expert_flag = int(_score_by_keywords(lowered, EN_EXPERT) > 0)
        media_flag = int(_score_by_keywords(lowered, EN_MEDIA) > 0)
        action_flag = int(_score_by_keywords(lowered, EN_ACTION) > 0)
        fivew_score = _fivew_score(lowered, EN_FIVEW)

    evidence_flag = int(response.label == "evidence" or has_url(content))
    deny_flag = int(response.label == "deny")
    query_flag = int(response.label == "query")
    emotion_flag = int(response.label == "emotion")

    publisher_type = _infer_publisher_type(
        explicit_publisher_type=explicit_publisher_type,
        metadata=metadata,
        official_flag=official_flag,
        expert_flag=expert_flag,
        media_flag=media_flag,
    )
    publisher_type_source = "explicit" if explicit_publisher_type else ("metadata" if metadata.get("user_verified") is True else "rule")
    explicit_correction = bool(explicit_is_correction)
    inferred_correction = bool(deny_flag or evidence_flag or action_flag or metadata.get("fact_check_available"))
    is_correction = (
        explicit_is_correction
        if explicit_is_correction is not None
        else inferred_correction
    )
    correction_type = (
        explicit_correction_type
        if explicit_correction_type is not None
        else _infer_correction_type(
            evidence_flag=evidence_flag,
            deny_flag=deny_flag,
            action_flag=action_flag,
            query_flag=query_flag,
        )
    )

    return InterventionSignalResult(
        stance_type=response.label,
        inferred_correction_signal=response.label,
        publisher_type=publisher_type,
        publisher_type_source=publisher_type_source,
        explicit_correction=explicit_correction,
        inferred_correction=inferred_correction,
        is_correction=is_correction,
        correction_type=correction_type,
        official_flag=official_flag,
        evidence_flag=evidence_flag,
        deny_flag=deny_flag,
        query_flag=query_flag,
        emotion_flag=emotion_flag,
        action_flag=action_flag,
        fivew_score=fivew_score,
        scores=response.scores,
    )


def _infer_publisher_type(
    *,
    explicit_publisher_type: str | None,
    metadata: dict,
    official_flag: int,
    expert_flag: int,
    media_flag: int,
) -> str:
    if explicit_publisher_type:
        return explicit_publisher_type
    if metadata.get("user_verified") is True:
        return "official"
    if expert_flag:
        return "expert"
    if media_flag:
        return "media"
    if official_flag:
        return "official"
    return "user"


def _infer_correction_type(
    *,
    evidence_flag: int,
    deny_flag: int,
    action_flag: int,
    query_flag: int,
) -> str | None:
    if evidence_flag:
        return "evidence_explanation"
    if deny_flag and action_flag:
        return "denial_action"
    if deny_flag:
        return "denial"
    if action_flag:
        return "action_guidance"
    if query_flag:
        return "query"
    return None


def _fivew_score(text: str, keywords_by_dim: dict[str, tuple[str, ...]]) -> int:
    return sum(
        1
        for keywords in keywords_by_dim.values()
        if any(keyword in text for keyword in keywords)
    )
