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

def special_compute_f1(a_gold, a_pred, length=-1):
    # gold_toks = get_tokens(a_gold)
    # pred_toks = get_tokens(a_pred)
    gold_toks = a_gold
    pred_toks = a_pred
    
    # for index_pred in range(len(pred_toks)):
    #     for index_gold in range(len(gold_toks)):
    #         if gold_toks[index_gold] in pred_toks[index_pred]:
    #             pred_toks[index_pred] = gold_toks[index_gold]
    
    # if length == -1: pass
    # else:
    #     if len(pred_toks) < length: 
    #         for id in range(length - len(pred_toks)):
    #             pred_toks.append('0')
    #     else:
    #         pass
    
    output_length = len(pred_toks)
    
    # print(f"debug gold_toks {gold_toks}")
    # print(f"debug pred_toks {pred_toks}")
    
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks), output_length, gold_toks, pred_toks
    if num_same == 0:
        return 0, output_length, gold_toks, pred_toks
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    
    output_length = len(pred_toks)
    # print(f"debug f1 {f1} ")
    # print(f"debug output_length {output_length} ")
    return f1, output_length, gold_toks, pred_toks

completion = []
with open('./encoded_inputs_completion.json', 'r') as fp:
    completion = json.load(fp)

prediction = []
with open('./encoded_inputs_prediction.json', 'r') as fp:
    prediction = json.load(fp)

prediction_rag = []
with open('./encoded_inputs_prediction_rag.json', 'r') as fp:
    prediction_rag = json.load(fp)

total_f1_scores = 0
total_f1_scores_rag = 0

increase = 0
decrease = 0

for index in range(len(completion)):
    
    
    this_completion = completion[index]
    this_prediction = prediction[index]
    this_prediction_rag = prediction_rag[index]

    
    
    filtered_completion = [element for element in this_completion if element != 0]
    # filtered_prediction = [element for element in this_prediction if element != 0]
    filtered_prediction_rag = [element for element in this_prediction_rag if element != 0]
    
    
    print(f"debug filtered_completion {len(filtered_completion)}")
    
    print(f"debug filtered_prediction_rag {len(filtered_prediction_rag)}")
    f1_scores_rag, length, gold_toks_rag, pred_toks_rag = special_compute_f1(filtered_completion, filtered_prediction_rag, length=-1)
    
    filtered_prediction = this_prediction[:length]
    
    print(f"debug filtered_prediction {len(filtered_prediction)}")
    f1_scores, _, gold_toks, pred_toks = special_compute_f1(filtered_completion, filtered_prediction, length=length)
    
    if f1_scores < f1_scores_rag:
        increase += 1
        # print(f"debug\n prompt {prompt}\n completion {completion}\n prediction {prediction}\n f1_scores {f1_scores}\n prediction_rag {prediction_rag}\n  f1_scores_rag {f1_scores_rag} \n")
    
    if f1_scores > f1_scores_rag:
        decrease += 1
        # print(f"debug\n completion {completion}\n prediction {prediction}\n f1_scores {f1_scores}\n prediction_rag {prediction_rag}\n  f1_scores_rag {f1_scores_rag} \n")
        # print(f"gold_toks_rag {gold_toks_rag}\n pred_toks_rag {pred_toks_rag}\n")
        # print(f"gold_toks {gold_toks}\n pred_toks {pred_toks}\n")
        # print(f"debug length {length} \n")
    
    total_f1_scores += f1_scores
    total_f1_scores_rag += f1_scores_rag

print(f"\n")
print(f"increase {increase}")
print(f"decrease {decrease}")
print("total_f1_scores: ", total_f1_scores / len(completion) * 100, "%")
print("total_f1_scores_rag: ", total_f1_scores_rag / len(completion) * 100, "%")