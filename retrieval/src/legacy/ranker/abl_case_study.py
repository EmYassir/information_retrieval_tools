import json
from tqdm import tqdm

def select_answer_with_high_rank(ifile,ofile):
    selected = []
    with open(ifile, 'r') as ifn:
        dat = ifn.read().splitlines()
        for line in tqdm(dat):
            que = json.loads(line)
            if que[0]["has_answers"][0] and len(que[0]["question"]) > 10 and len(que[0]["answers_tok"][0]) == 1:
                selected.append(line)
    print("Selected", len(selected), "out of", len(dat))
    with open(ofile, 'w+') as ofn:
        for line in selected:
            ofn.write(line+'\n')

if __name__ == "__main__":
    ifile = '/u/pandu/data/openQA/data/datasets/quasart/test_utf8.norm_tok_rank.json'
    ofile = '/u/pandu/data/openQA/data/datasets/quasart/case_utf8.norm_tok_rank.json'
    select_answer_with_high_rank(ifile, ofile)
    print("Done!")