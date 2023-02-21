from src.utilities.Retriever.retriever_utils import *
import torch
import torch.distributed as dist
from tqdm import tqdm
import pandas as pd
from transformers import DPRReaderTokenizerFast
from transformers import DPRReader
from transformers import DataCollatorWithPadding


def prepare(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: Optional[int] = 32,
    pin_memory: Optional[bool] = False,
    num_workers: Optional[int] = 2,
):
    """Creates dataloader for multi gpu perform

    Args:
        dataset (Dataset): Dataset to create Dataloader for
        tokenizer (PreTrainedTokenizerBase): Tokenizer used to tokenize the dataset
        batch_size (Optional[int], optional): Batch size to use. Defaults to 32.
        pin_memory (Optional[bool], optional): Defaults to False.
        num_workers (Optional[int], optional): Number of workers. Defaults to 2.

    Returns:
        _type_: _description_
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer, padding=True),
    )
    return dataloader


def perform_reader(rank, world_size, dataset, prefix, batch_size):
    # setup the process groups
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Setup Encoder
    tokenizer = DPRReaderTokenizerFast.from_pretrained(
        "facebook/dpr-reader-single-nq-base"
    )
    reader = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base").to(device)

    dataset = dataset.shard(num_shards=world_size, index=rank, contiguous=True)
    dataloader = prepare(dataset, tokenizer, batch_size=batch_size)

    # New dataset column lists
    input_ids = []
    attention_mask = []
    start_logits = []
    end_logits = []
    relevance_logits = []

    # Synchronizing the processes
    dist.barrier()

    # Encoding the data
    # Only show the progress bar once on each machine.
    iter_bar = tqdm(dataloader, desc="-Iter", disable=rank != 0)
    for batch in iter_bar:
        with torch.no_grad():
            b = batch["input_ids"].shape[0]
            k = batch["input_ids"].shape[1]
            reader_input = {
                key: value.reshape(b * k, -1).to(device) for key, value in batch.items()
            }
            out = reader(**reader_input)
            new_out = {
                key: value.to("cpu").detach().numpy().reshape(b, k, -1)
                for key, value in out.items()
            }
            input_ids.extend(batch["input_ids"].tolist())
            attention_mask.extend(batch["attention_mask"].tolist())
            start_logits.extend(new_out["start_logits"].tolist())
            end_logits.extend(new_out["end_logits"].tolist())
            relevance_logits.extend(new_out["relevance_logits"].tolist())
        iter_bar.update()
    iter_bar.close()

    df = pd.DataFrame(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_logits": start_logits,
            "relevance_logits": relevance_logits,
            "end_logits": end_logits,
        }
    )
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(prefix + str(rank))

    dist.barrier()

    cleanup()
