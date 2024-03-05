import re
import json
import string
import collections

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

data_0doc_llama2 = []
with open('/home/caizf/projects/RagDecoding/results/results_v35/musique_llama2chat_20doc
          .json', 'r') as fp:
    for line in fp.readlines():
        data_0doc_llama2.append(json.loads(line))

data_0doc_llama2 = data_0doc_llama2[0]

total_f1_scores = 0

for index in range(len(data_0doc_llama2)):
    completion = data_0doc_llama2[index]["completion"]
    prediction = data_0doc_llama2[index]["prediction"]

    f1_scores = compute_f1(completion, prediction)
    total_f1_scores += f1_scores

print("total_f1_scores: ", total_f1_scores / len(data_0doc_llama2) * 100, "%")