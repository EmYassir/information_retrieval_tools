import re
from datasets import load_from_disk, DatasetDict
from functools import partial
from tqdm import tqdm
import torch
from sklearn.metrics import ndcg_score
import torch
import math


def get_index_list(batch_field, indices):
    return torch.stack(
        [batch_field[i][j] for i, index_list in enumerate(indices) for j in index_list]
    )


def get_generator_sample_mask(batch, scores, num_samples=None):
    batch_size = batch["input_ids"].shape[0]
    mask = batch["answerability_labels"]

    out_mask = torch.zeros(mask.shape, dtype=torch.bool)

    num_to_sample = int(mask.sum())

    # Sample as many generator-chosen documents as there are answerable documents
    row_indices = torch.multinomial(
        torch.softmax(scores, 0), num_to_sample, replacement=False
    )

    # Set mask to 1 for entries selected by generator
    out_mask[row_indices] = 1
    return out_mask


def get_gen_oversample_indices(scores, num_to_sample=250):
    # Sample as many generator-chosen documents as there are answerable documents
    row_indices = torch.multinomial(scores, num_to_sample, replacement=True)

    return row_indices


def compute_ans_loss(ans_scores, ans_batch, ans_device):
    # Calculate weight for positive examples due to class imbalance
    batch_size = ans_batch["input_ids"].shape[0]
    num_docs = ans_batch["input_ids"].shape[1]
    # w0 = batch_size * num_docs / (2 * ans_batch["answerability_labels"].sum())

    ans_loss_fn = torch.nn.BCEWithLogitsLoss(
        # pos_weight=torch.tensor([w0]).to(ans_device)
    )

    loss = ans_loss_fn(ans_scores, ans_batch["answerability_labels"].float())
    return loss


def compute_gen_loss(
    gen_batch,
    gen_scores,
    dis_scores,
    ans_scores,
    gen_device,
    ans_dis_weight=0.25,
    reg_weight=1,
    num_gen_samples=20,
    lambda_val=0.5,
):
    answerability_mask = gen_batch["answerability_labels"]
    num_docs = len(answerability_mask)

    # Clamp scores since we will perform log(gen_scores)
    gen_scores = torch.softmax(gen_scores, dim=0).clamp(min=1e-8)

    # For each batch of questions, Find the uniform probability of picking a ground truth document
    # prob_true = P(d), d ~ p_true, for each question
    prob_true = 1.0 / torch.sum(answerability_mask)

    regularizer = torch.sum(
        prob_true
        * answerability_mask
        * (-torch.log(gen_scores) + torch.log(prob_true) * answerability_mask)
    )
    # gen_dat_mask = get_generator_sample_mask(gen_batch, gen_scores, num_gen_samples)

    # Importance sampling
    prob_IS = gen_scores * (1.0 - lambda_val)
    prob_IS = prob_IS + answerability_mask * lambda_val * prob_true
    choose_IS = gen_scores / prob_IS

    # Convert logits into scores
    # Place both scores in generator device
    ans_scores = torch.sigmoid(ans_scores).to(gen_device)
    dis_scores = torch.sigmoid(dis_scores).to(gen_device)
    gen_scores = gen_scores

    loss = -torch.mean(
        torch.log(gen_scores) * (dis_scores + ans_dis_weight * ans_scores) * choose_IS
    )
    return loss + reg_weight * regularizer


def compute_gen_ind_loss(
    gen_batch,
    gen_scores,
    dis_scores,
    ans_scores,
    gen_device,
    ans_dis_weight=0.25,
    reg_weight=1,
    num_gen_samples=20,
    lambda_val=0.5,
):
    answerability_mask = gen_batch["answerability_labels"]
    num_docs = len(answerability_mask)

    # Clamp scores since we will perform log(gen_scores)
    gen_scores = torch.softmax(gen_scores, dim=0).clamp(min=1e-8)

    # For each batch of questions, Find the uniform probability of picking a ground truth document
    # prob_true = P(d), d ~ p_true, for each question
    prob_true = 1.0 / torch.sum(answerability_mask)

    regularizer = torch.sum(
        prob_true
        * answerability_mask
        * (-torch.log(gen_scores) + torch.log(prob_true) * answerability_mask)
    )
    gen_dat_indices = get_gen_oversample_indices(gen_scores, num_to_sample=250)

    # Importance sampling
    prob_IS = gen_scores * (1.0 - lambda_val)
    prob_IS = prob_IS + answerability_mask * lambda_val * prob_true
    choose_IS = gen_scores / prob_IS
    choose_IS = choose_IS[gen_dat_indices]
    # Convert logits into scores
    # Place both scores in generator device
    ans_scores = torch.sigmoid(ans_scores).to(gen_device)[gen_dat_indices]
    dis_scores = torch.sigmoid(dis_scores).to(gen_device)[gen_dat_indices]
    gen_scores = gen_scores[gen_dat_indices]

    loss = -torch.mean(
        torch.log(gen_scores) * (dis_scores + ans_dis_weight * ans_scores) * choose_IS
    )
    return loss + reg_weight * regularizer


def compute_r_dis_loss(batch, dis_scores, gen_scores, gen_device):
    rank_loss_fn = torch.nn.BCEWithLogitsLoss()

    # Get discriminator scores for ground truth documents
    # Set the labels to 1
    answerability_labels = batch["answerability_labels"]
    pos_dis_scores = dis_scores[answerability_labels].to(gen_device)
    pos_dis_labels = torch.ones(pos_dis_scores.shape).to(gen_device)

    # Select documents from the generator
    # Set the labels to 0
    gen_dat_mask = get_generator_sample_mask(batch, gen_scores)
    neg_dis_scores = dis_scores[gen_dat_mask].to(gen_device)
    neg_dis_labels = torch.zeros(neg_dis_scores.shape).to(gen_device)

    # Compute loss of discriminator
    rank_loss = rank_loss_fn(neg_dis_scores, neg_dis_labels) + rank_loss_fn(
        pos_dis_scores, pos_dis_labels
    )
    return rank_loss


def compute_hits(
    all_scores, all_answerability_labels, hits_list=[1, 3, 5, 10, 20, 30, 50]
):
    # TODO: Add support for both ans_eval and non_ans_eval
    ndcg_gen_hits = [
        ndcg_score(all_answerability_labels, all_scores, k=hits) for hits in hits_list
    ]
    return ndcg_gen_hits


def get_hits_dict(all_scores, all_answerability_labels, hits_list, model_name):
    """Computes hits@n for each n in the hits list.

    Args:
        all_scores (list[tensor]): List of tensors, where each tensor contains the scores over num_docs
        all_answerability_labels (list[tensor]): list of tensors where each tensor is the answerability
            label of the documents
        hits_list (list[int]): list of n for each hits@n score to be computed
        model_name (str): Name of the model being evaluated

    Returns:
        dict: Dictionary of the {model_name}_hits@{n} for n in hits_list. Represents the NDCG score over
            all the given n values.
    """
    hits_scores_full = compute_hits(all_scores, all_answerability_labels, hits_list)

    ret = {
        model_name + "_full_hits@" + str(hit): h_score
        for hit, h_score in zip(hits_list, hits_scores_full)
    }
    all_scores_sub, all_answerability_labels_sub = zip(
        *filter(lambda x: any(x[1]), zip(all_scores, all_answerability_labels))
    )
    if len(all_answerability_labels_sub):
        hits_scores_sub = compute_hits(
            all_scores_sub, all_answerability_labels_sub, hits_list
        )
        ret.update(
            {
                model_name + "_sub_hits@" + str(hit): h_score
                for hit, h_score in zip(hits_list, hits_scores_sub)
            }
        )
    return ret


def get_collate_fn(hf_collator):
    return partial(collate_fn, hf_collator=hf_collator)


def collate_fn(features, hf_collator):
    # Input ids are currently in list form
    batch_size = len(features)
    first = features[0]
    num_docs = len(first["input_ids"])
    tokenizer_keys = first.keys()

    # Flatten the batch. ie, convert from list of dict to dict of list
    # and flatten the list so that lists of shape (BxD) become (B*D)
    flattened_batch = {
        key: [doc for datum in features for doc in datum[key]] for key in tokenizer_keys
    }

    # Apply collator and return shape to (BxD)
    padded_flat_batch = hf_collator(flattened_batch)
    reshaped_batch = {
        key: value.reshape(batch_size, num_docs, -1)
        for key, value in padded_flat_batch.items()
    }

    # Convert answerability labels to the correct shape+
    reshaped_batch["answerability_labels"] = reshaped_batch[
        "answerability_labels"
    ].reshape(batch_size, num_docs)
    return reshaped_batch


def find_whole_answer(w):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search


def label_answerability(example):
    # Get regular expression for all possible answers
    compile_func = find_whole_answer(
        "|".join([re.escape(answer) for answer in example["answer"]])
    )

    # Map the regex onto all the documents, convert to booleans, and return the list
    answerability_labels = list(
        map(float, map(bool, map(compile_func, example["documents"])))
    )
    return {"answerability_labels": answerability_labels}


def prepare_dataset(args, tokenizer):
    # Get the datasets: Loads a processed dataset from disk. Must be a dataset created by a DatasetProcessor
    raw_datasets = load_from_disk(args.dataset_path)

    # If test set doesn't contain answers, ignore it
    if "answer" not in raw_datasets["test"].column_names:
        raw_datasets = DatasetDict(
            {"train": raw_datasets["train"], "dev": raw_datasets["dev"]}
        )

    column_names = raw_datasets["train"].column_names
    answerable_datasets = raw_datasets.map(
        label_answerability,
        num_proc=args.preprocessing_num_workers,
        desc="Labeling each document as having an answer or not",
    )

    # Keep only questions that have at least one answerable document
    answerable_datasets["train"] = answerable_datasets["train"].filter(
        lambda example: any(example["answerability_labels"]),
        num_proc=args.preprocessing_num_workers,
    )

    # Tokenize the questions and documents
    def tokenize_function(examples):
        # Prepend each document with the question
        list_of_dicts = [
            tokenizer(
                examples["question"],
                document,
                max_length=args.max_seq_len,
                truncation=True,
            )
            for document in examples["documents"]
        ]
        first = list_of_dicts[0]
        return {key: [elem[key] for elem in list_of_dicts] for key in first.keys()}

    tokenized_datasets = answerable_datasets.map(
        tokenize_function,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on every text in dataset",
    )

    return tokenized_datasets


def move_batch_to_devices(batch, gen_device, dis_device, ans_device):
    gen_batch = {key: value.to(gen_device) for key, value in batch.items()}
    dis_batch = {key: value.to(dis_device) for key, value in batch.items()}
    ans_batch = {key: value.to(ans_device) for key, value in batch.items()}
    return gen_batch, dis_batch, ans_batch


def get_eval_metrics(
    args,
    generator,
    discriminator,
    ans_discriminator,
    eval_dataloader,
    gen_device,
    dis_device,
    ans_device,
):
    # Prepare models for evaluation
    generator = generator.to(gen_device)
    discriminator = discriminator.to(dis_device)
    ans_discriminator = ans_discriminator.to(ans_device)
    generator.eval()
    discriminator.eval()
    ans_discriminator.eval()

    # Initialize the score lists
    losses = []
    all_gen_scores = []
    all_dis_scores = []
    all_ans_scores = []
    all_answerability_labels = []
    # Run through the dataloader
    for batch in tqdm(eval_dataloader, desc="Performing Evaluation"):
        batch = {key: value[0] for key, value in batch.items()}
        gen_batch, dis_batch, ans_batch = move_batch_to_devices(
            batch, gen_device, dis_device, ans_device
        )
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                gen_scores = generator(
                    input_ids=gen_batch["input_ids"],
                    attention_mask=gen_batch["attention_mask"],
                    token_type_ids=gen_batch["token_type_ids"],
                ).logits.squeeze(-1)
                dis_scores = discriminator(
                    input_ids=dis_batch["input_ids"],
                    attention_mask=dis_batch["attention_mask"],
                    token_type_ids=dis_batch["token_type_ids"],
                ).logits.squeeze(-1)
                ans_scores = ans_discriminator(
                    input_ids=ans_batch["input_ids"],
                    attention_mask=ans_batch["attention_mask"],
                    token_type_ids=ans_batch["token_type_ids"],
                ).logits.squeeze(-1)
            loss = compute_gen_loss(
                gen_batch,
                gen_scores,
                dis_scores,
                ans_scores,
                gen_device,
                args.ans_discriminator_weight,
                args.regularizer_weight,
            )
        losses.append(loss.item())
        all_gen_scores.append(torch.softmax(gen_scores, 0).tolist())
        all_dis_scores.append(torch.sigmoid(dis_scores).tolist())
        all_ans_scores.append(torch.sigmoid(ans_scores).tolist())
        all_answerability_labels.append(batch["answerability_labels"].tolist())
    # For each model, calculate its hit score.
    # This is to monitor how each of the models performs in finding ground truth documents
    ret = {}
    gen_hits_scores = get_hits_dict(
        all_gen_scores, all_answerability_labels, args.hits_list, "generator"
    )
    dis_hits_scores = get_hits_dict(
        all_dis_scores, all_answerability_labels, args.hits_list, "discriminator"
    )
    ans_hits_scores = get_hits_dict(
        all_ans_scores,
        all_answerability_labels,
        args.hits_list,
        "ans_discriminator",
    )
    ret.update(gen_hits_scores)
    ret.update(dis_hits_scores)
    ret.update(ans_hits_scores)
    try:
        # Compute the mean loss
        eval_loss = sum(losses) / len(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    ret["eval_loss"] = eval_loss
    ret["perplexity"] = perplexity
    return ret
