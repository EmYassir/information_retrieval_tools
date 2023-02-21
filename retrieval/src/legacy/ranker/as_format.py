import json
from tqdm import tqdm
import io
from utils.data import *


def as2quasart(ifn, ofn_json, ofn_txt):
    with open(ifn, 'r') as ifr:
        with io.open(ofn_txt, 'w+', encoding='utf-8') as oftr:
            with io.open(ofn_json, 'w+', encoding='utf-8') as ofjr:
                dat = ifr.read().splitlines()
                for q_id, line in tqdm(enumerate(dat)):
                    que = json.loads(line)
                    # if q_id > 5:
                    #     break
                    quest = que["question"]
                    quest_tok = normalize(quest)
                    quest_tok = PROCESS_TOK.tokenize(quest_tok)
                    quest_tok = quest_tok.words(uncased=True)

                    ctxs = que["candidates"]
                    oftr.write(json.dumps({"question": quest, "answers": []}) + '\n')
                    pairs = []
                    for c_id, ctx in enumerate(ctxs):
                        p_id = ['q'+ str(q_id), c_id]
                        doc_tok = normalize(ctx["sentence"])
                        doc_tok = PROCESS_TOK.tokenize(doc_tok)
                        doc_tok = doc_tok.words(uncased=True)
                        ans = [True, []] if ctx['label'] == 0 else [False, []]
                        pairs.append({"document": doc_tok, "id": p_id, "question": quest_tok, "has_answers": ans})
                    ofjr.write(json.dumps(pairs) + '\n')


def quasart2nq(quasart_json, nq_json, ofn):
    tot_len = sum(1 for i in open(quasart_json, 'r'))
    with open(nq_json, 'r') as njr:
        nq_dat = json.load(njr)
        new_nq_dat = []
        with io.open(quasart_json, 'r', encoding='utf-8') as qjr:
            q_id = 0
            for line in tqdm(qjr, total=tot_len):
                que = json.loads(line)
                nq_i = nq_dat[q_id]
                id_ctxs = {ctx['id']: ctx for ctx in nq_i['ctxs']}
                new_ctxs = []
                for doc in que:
                    if doc['id'][1] in id_ctxs:
                        new_ctx = id_ctxs[doc['id'][1]]
                        new_ctx['score'] = str(doc['score'])
                        new_ctxs.append(new_ctx)
                new_nq_dat.append({
                    "question": nq_i["question"],
                    "answers": nq_i["answers"],
                    "ctxs": new_ctxs
                })

                q_id += 1
        with open(ofn, 'w+') as ofw:
            ofw.write(json.dumps(new_nq_dat, indent=4)+'\n')


if __name__ == '__main__':

    '''trecqa_as to squart'''
    splt = 'train'
    # splt = 'test'
    # splt = 'dev'
    ifn = '/u/pandu/data/openQA/data/datasets/trecqa_as/' + splt + '-filtered.jsonl'
    ofn_json = '/u/pandu/data/openQA/data/datasets/trecqa_as_abl/' + splt + '.json'
    ofn_txt = '/u/pandu/data/openQA/data/datasets/trecqa_as_abl/' + splt + '.txt'
    as2quasart(ifn, ofn_json, ofn_txt)

    '''squart to nq'''
    # splt = 'test'
    # nq_json = '/u/pandu/data/openQA/data/datasets/nq/tfidf/nq-'+splt+'.json'
    # quasart_json = '/u/pandu/data/openQA/data/datasets/nqsub/'+splt+'_utf8.norm_tok_rank.json'
    # nn_json = '/u/pandu/data/openQA/data/datasets/nq/tfidf/nq-'+splt+'-rank.json'
    # quasart2nq(quasart_json, nq_json, nn_json)

    print("Done!")

    '''debug set cmd'''
    # head -n 100 $ofn_json$ > $debug_ofn_json$