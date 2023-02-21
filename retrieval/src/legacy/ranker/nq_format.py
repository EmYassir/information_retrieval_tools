import json
from tqdm import tqdm
import io
from utils.data import *
from joblib import Parallel, delayed

def nq2quasart(ifn, ofn_json, ofn_txt):

    with open(ifn, 'r') as ifr:
        with io.open(ofn_txt, 'w+', encoding='utf-8') as oftr:
            with io.open(ofn_json, 'w+', encoding='utf-8') as ofjr:
                dat = json.load(ifr)
                for q_id, que in tqdm(enumerate(dat), total=len(dat)):
                    # if q_id > 5:
                    #     break
                    quest = que["question"]
                    quest_tok = normalize(quest)
                    quest_tok = PROCESS_TOK.tokenize(quest_tok)
                    quest_tok = quest_tok.words(uncased=True)
                    ans = que["answers"]
                    ctxs = que["ctxs"]
                    oftr.write(json.dumps({"question": quest, "answers": ans}) + '\n')
                    pairs = []
                    for ctx in ctxs:
                        p_id = ['s'+ str(q_id), ctx["id"]]
                        doc_tok = normalize(ctx["text"])
                        doc_tok = PROCESS_TOK.tokenize(doc_tok)
                        if 'score' in ctx:
                            score = ctx["score"]
                        else:
                            score = '0.0'
                        doc_tok = doc_tok.words(uncased=True)
                        pairs.append({"document": doc_tok, "id": p_id, "question": quest_tok, "score":score})
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

    '''nq to squart'''
    iset = 'nq'
    # retr = 'tfidf'
    # retr = 'dpr_retrieval_sub'  # 50
    retr = 'dpr_retrieval_sub200'  # equal to 'dpr_retrieval'
    retr = 'dpr_retrieval_sub50'

    # oset = 'nqsub'
    oset = 'nqsub200'
    oset = 'nqsub50'

    splt = 'train'
    # splt = 'test'
    # splt = 'dev'
    # splt = 'debug'
    ifn = '/u/pandu/data/openQA/data/datasets/'+iset+'/'+retr+'/nq-'+splt+'.json'
    ofn_json = '/u/pandu/data/openQA/data/datasets/'+oset+'/'+splt+'.json'
    ofn_txt = '/u/pandu/data/openQA/data/datasets/'+oset+'/'+splt+'.txt'
    nq2quasart(ifn, ofn_json, ofn_txt)

    '''squart to nq'''
    # # splt = 'test'
    # nq_json = '/u/pandu/data/openQA/data/datasets/'+iset+'/'+retr+'/nq-'+splt+'.json'
    # quasart_json = '/u/pandu/data/openQA/data/datasets/'+oset+'/'+splt+'_utf8.norm_tok_rank.json'
    # nn_json = '/u/pandu/data/openQA/data/datasets/'+iset+'/'+retr+'/nq-'+splt+'-rank.json'
    # quasart2nq(quasart_json, nq_json, nn_json)

    '''trec to nq'''
    # splt = 'test'
    # nq_json = '/u/pandu/data/openQA/data/datasets/trec/' + retr + '/nq-' + splt + '.json'
    # quasart_json = '/u/pandu/data/openQA/data/datasets/trec100_full/test_utf8.norm_tok_rank.json'
    # nn_json = '/u/pandu/data/openQA/data/datasets/trec100_full/nq-' + splt + '-rank.json'
    # quasart2nq(quasart_json, nq_json, nn_json)

    '''nq to quasart: webquestions'''
    # splt = 'test'
    # ifn = '/u/pandu/data/openQA/fb_wq/dpr_retrieval_sub/wq-'+splt+'.json'
    # ofn_json = '/u/pandu/data/openQA/data/datasets/wqsub/'+splt+'.json'
    # ofn_txt = '/u/pandu/data/openQA/data/datasets/wqsub/'+splt+'.txt'
    # nq2quasart(ifn, ofn_json, ofn_txt)

    print("Done!")

    '''debug set cmd'''
    # head -n 100 $ofn_json$ > $debug_ofn_json$