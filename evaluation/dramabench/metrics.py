"""Compute content-label statistics without relying on model-reported scores."""

from collections import Counter

def metrics(dimension, value):
    while set(value) == {"result"} and isinstance(value["result"], dict):
        value = value["result"]
    if dimension == "narrative_efficiency":
        labels = [beat["classification"].upper() for beat in value["beats"]]
        if not labels or any(label not in {"DRIVER", "STATIC", "REDUNDANT"} for label in labels):
            raise ValueError("Invalid narrative labels")
        counts = Counter(labels)
        return {"enr": counts["DRIVER"] / len(labels), "driver_beats": counts["DRIVER"], "annotated_beats": len(labels)}
    if dimension == "character_consistency":
        labels = [line["classification"].upper() for line in value["dialogue_analysis"]]
        if not labels:
            return {"ooc_rate": None, "voice_distinctiveness": None, "annotated_dialogues": 0}
        if any(label not in {"IN_CHARACTER", "NEUTRAL", "OOC"} for label in labels):
            raise ValueError("Invalid character labels")
        counts = Counter(labels)
        return {"ooc_rate": counts["OOC"] / len(labels), "voice_distinctiveness": counts["IN_CHARACTER"] / len(labels), "annotated_dialogues": len(labels)}
    if dimension == "emotional_depth":
        arc = value["emotional_arc"]["arc_type"].upper()
        if arc not in {"SHIFT", "STATIC"}:
            raise ValueError("Invalid emotional arc label")
        moments = value["complex_emotions"]
        if not isinstance(moments, list):
            raise ValueError("Invalid complex-emotion list")
        return {"arc_score": int(arc == "SHIFT"), "complex_emotion_count": len(moments), "complex_emotion_present": int(bool(moments))}
    if dimension == "logic_consistency":
        labels = [fact["status"].upper() for fact in value["facts"]
                  if fact.get("fact_kind", "WORLD_FACT") == "WORLD_FACT"]
        if any(label not in {"VIOLATED", "MAINTAINED", "IRRELEVANT"} for label in labels):
            raise ValueError("Invalid continuity labels")
        counts = Counter(labels)
        relevant = counts["VIOLATED"] + counts["MAINTAINED"]
        return {"logic_break_rate": counts["VIOLATED"] / relevant if relevant else None, "maintained_facts": counts["MAINTAINED"], "relevant_facts": relevant}
    label = value["conflict_handling"]["classification"].upper()
    weight = {"ESCALATION": 2, "TWIST": 2, "PAUSE": 1, "RESOLUTION": 0, "DROPPED": -5}[label]
    return {"conflict_score": weight, "conflict_label": label, "drop_indicator": int(label == "DROPPED")}
