import torch

def transfer_generator(filename, trans_filename):
    print('=> loading checkpoint', filename)
    cp = torch.load(filename)
    state = {"g_epoch": cp["g_epoch"],
             "best_acc": cp["best_acc"],
             'dis_state': cp["dis_state"],
             'dis_optimizer': cp['dis_optimizer'],
             'gen_state': cp["gen_state"],
             'gen_optimizer': cp["gen_optimizer"],
             'word_dict': cp['word_dict'],
             'feature_dict': cp["feature_dict"]}

    print("=> checkpoint best EM %.2f", cp["best_acc"])
    print('-> saving checkpoint to', trans_filename)
    torch.save(state, trans_filename, _use_new_zipfile_serialization=False)
    print('Done!')

if __name__ == "__main__":
    # transfering models created by pytorch 1.6 to old versions, such as pytorch 1.2
    rankers = {
        "octal": {
            "quasart": "/u/pandu/pyspace/sigir/ranker/output/quasart/aranker/12-04-13-46-44-octal18/checkpoints/best_acc.pth.tar",
            "trec": "/u/pandu/pyspace/sigir/ranker/output/trec/aranker/12-25-00-32-12-Thu-407-X299/checkpoints/best_acc.pth.tar",
            "webquestions": "/u/pandu/pyspace/sigir/ranker/output/webquestions/aranker/12-23-00-54-09-Thu-407-X299/checkpoints/best_acc.pth.tar",
            "searchqa": "/u/pandu/pyspace/sigir/ranker/output/searchqa/aranker/12-10-14-41-34-cdr2496.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar",
            "unftriviaqa": "/u/pandu/pyspace/sigir/ranker/output/unftriviaqa/aranker/12-26-17-18-01-cdr2513.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar",
            "nqsub": "/u/pandu/pyspace/sigir/ranker/output/nqsub/aranker/cedar-11-03-22-44-02/best_acc.pth.tar"
        },
        "computecanada": {
            "quasart": "/home/mutux/projects/def-jynie/mutux/pyspace/aqa/aranker/output/quasart/aranker/cdr209.int.cedar.computecanada.ca-07-21-21-00-38/checkpoints/best_acc.pth.tar",
            "trec": "",
            "webquestions": "",
            "searchqa": "/home/mutux/projects/def-jynie/mutux/pyspace/sigir2021/ranker/output/searchqa/aranker/12-10-14-41-34-cdr2496.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar",
            "unftriviaqa": "/home/mutux/projects/def-jynie/mutux/pyspace/sigir2021/ranker/output/unftriviaqa/aranker/12-26-17-18-01-cdr2513.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar",
            "nqsub": "/home/mutux/projects/def-jynie/mutux/pyspace/aqa/aranker/output/nqsub/aranker/11-03-22-44-02-cdr2500.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar"
        },
        "thu": {
            "quasart": "",
            "trec": "",
            "webquestions": "",
            "searchqa": "/home/zlx/ranker/output/searchqa/aranker/12-10-14-41-34-cdr2496.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar",
            "unftriviaqa": "/home/zlx/ranker/output/unftriviaqa/aranker/12-26-17-18-01-cdr2513.int.cedar.computecanada.ca/checkpoints/best_acc.pth.tar",
            "nqsub": "/home/zlx/ranker/output/nqsub/aranker/cedar-11-03-22-44-02/best_acc.pth.tar"
        },
    }
    dset = 'unftriviaqa'
    # dset = 'nqsub'
    fn_v16 = rankers['octal'][dset]
    fn_old = '.'.join(fn_v16.split('.')[:-1]) + '.old.tar'
    transfer_generator(fn_v16, fn_old)