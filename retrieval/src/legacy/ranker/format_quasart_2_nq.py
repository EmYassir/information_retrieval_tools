import json
from tqdm import tqdm
import io


def quasart2nq(data_json, data_txt, ofn):
    tot_len = sum(1 for i in open(data_txt, 'r'))
    nq_qas = []
    new_nq_dat = []
    with io.open(data_txt, 'r', encoding='utf-8') as njr:
        for line in tqdm(njr, total=tot_len, desc="Loading"):
            nq_qas.append(json.loads(line.strip()))

    with io.open(data_json, 'r', encoding='utf-8') as qjr:
        for lid, line in enumerate(tqdm(qjr, total=tot_len, desc="Formatting")):
            que = json.loads(line)
            ctxs = []
            for pid, doc in enumerate(que):
                # if isinstance(doc["id"], int):
                #     cid = 'q'+str(doc["id"]) + 'p' + str(pid)
                # else:
                cid = doc["id"][0] + 'p' +str(doc["id"][1])
                ctxs.append({"id": cid,
                             "title": "",
                             "text": " ".join(doc["document"]),
                             "score": str(doc["score"]),
                             "has_answer": doc["has_answers"][0]})
            new_nq_dat.append({
                "question": nq_qas[lid]["question"],
                "answers": nq_qas[lid]["answers"],
                "ctxs": ctxs
            })
    print("Writing data in nq format to file", ofn)
    with open(ofn, 'w+') as ofw:
        ofw.write(json.dumps(new_nq_dat, indent=4)+'\n')


if __name__ == '__main__':

    # splt = 'debug'
    splt = 'test'
    # splt = 'dev'
    # splt = 'train'

    # dset = 'quasart'
    # dset = 'trec'
    # dset = 'webquestions'
    # dset = 'searchqa'
    # dset = 'unftriviaqa'
    # dset = 'nqsub'
    dset = 'trec100_full'
    data_dir = '/u/pandu/data/openQA/data/datasets/'
    data_txt = data_dir + dset + '/' + splt + '_utf8.norm.tok.txt'
    data_json = data_dir + dset + '/' + splt + '_utf8.norm_tok_rank.json'
    out_json = data_dir + dset + '/nq-' + splt + '-rank.json'
    quasart2nq(data_json, data_txt, out_json)
    print("Done!")

    '''debug set cmd'''
    # head -n 100 $ofn_json$ > $debug_ofn_json$