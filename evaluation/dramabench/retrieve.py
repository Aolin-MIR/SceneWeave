"""Retrieve human TRAIN demonstrations using only the writing request."""

import math
import re
from collections import Counter

def retrieve_examples(cases, training, count):
    """Retrieve only from human TRAIN, using requests rather than candidate drafts."""
    corpus = [row for row in training if (
        str(row.get("label_provenance", "")).lower().startswith("original human")
        or str(row.get("label_provenance", "")).startswith("Arena actual user preference"))]
    def tokens(value):
        text = str(value).lower()
        words = re.findall(r"[a-z][a-z0-9_]+", text)
        for phrase in re.findall(r"[\u4e00-\u9fff]+", text):
            words.extend(phrase[i:i+2] for i in range(max(0,len(phrase)-1)))
        return Counter(words)
    documents = [tokens(row["prompt"]) for row in corpus]
    frequencies = Counter(word for document in documents for word in document)
    idf = {word:math.log((1+len(documents))/(1+frequency))+1 for word,frequency in frequencies.items()}
    def vector(document):
        weights = {word:(1+math.log(n))*idf.get(word,1) for word,n in document.items()}
        length = math.sqrt(sum(v*v for v in weights.values())) or 1
        return {word:v/length for word,v in weights.items()}
    vectors = [vector(document) for document in documents]
    assignments = []
    for case in cases:
        last_request = case["request"].get("last_instruction", case["request"]) if isinstance(case["request"],dict) else case["request"]
        query = vector(tokens(last_request))
        scores = [sum(v*query.get(word,0) for word,v in vec.items()) for vec in vectors]
        indices = sorted(range(len(corpus)),key=lambda i:(-scores[i],corpus[i]["pair_id"]))[:count]
        examples = []
        for index in indices:
            row=corpus[index]
            examples.append({"writing_request":row["prompt"],"a":row["chosen"],"b":row["rejected"],
                             "human_probability_a":row["preference_probability"],
                             "released_human_explanations":row.get("original_human_annotations",[])})
        case["_examples"]=examples
        assignments.append({"case_id":case["id"],"human_train_example_ids":[corpus[i]["pair_id"] for i in indices],"retrieval_uses_candidate_text_or_preferences":False})
    return assignments
