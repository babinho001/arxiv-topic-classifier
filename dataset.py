import torch

from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset, DatasetDict, load_dataset
from tokenizers import Tokenizer

from typing import Any, Dict, List

from config import CATEGORY_TO_ID


def primary_category(categories: List[str]) -> str:
    """
    Extracts the leaf (most specific) name of a paper's first-listed category.

    Args:
        categories (List[str]): Hierarchical category paths, e.g.
            ["Physics Archive->astro-ph->astro-ph.GA", "..."].

    Returns:
        str: Leaf category of the first entry, e.g. "astro-ph.GA".
    """
    return categories[0].split("->")[-1].strip()


def load_data(config) -> DatasetDict:
    """
    Loads the arXiv Categories dataset from Hugging Face, keeps only the rows
    whose primary (first-listed) category is one of the configured top
    categories, and attaches an integer `label` column.

    Args:
        config: A config file.

    Returns:
        DatasetDict: train/validation/test splits, each with (at least) the
            columns `abstract` and `label`.
    """
    dataset = load_dataset(config['dataset_name'])

    filtered = DatasetDict()
    for split_name, split in dataset.items():
        split = split.filter(lambda row: primary_category(row['categories']) in CATEGORY_TO_ID)
        split = split.map(lambda row: {'label': CATEGORY_TO_ID[primary_category(row['categories'])]})
        filtered[split_name] = split

    return filtered


class AbstractDataset(TorchDataset):
    """
    Wrapper class of Torch Dataset.
    Has to have methods __init__, __len__ and __getitem__ to function properly.
    """

    def __init__(
            self,
            dataset: HFDataset,
            tokenizer: Tokenizer,
            context_size: int
        ) -> None:
        """Initializing the AbstractDataset object.

        Args:
            dataset (HFDataset):
                HuggingFace dataset with (at least) columns 'abstract' and 'label'.
            tokenizer (Tokenizer): Tokenizer for the abstract text.
            context_size (int): Maximum allowed length of an abstract (in tokens).
        """
        super().__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.context_size = context_size

        # Start of sentence token signifies the beginning of the abstract.
        self.sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype = torch.int64)

        # End of sentence token signifies the end of the abstract.
        self.eos_token = torch.tensor([tokenizer.token_to_id('[EOS]')], dtype = torch.int64)

        # Padding token signifies the placeholder token for abstracts shorter than context size.
        self.pad_token = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype = torch.int64)

    def __len__(self) -> int:
        """
        Returns:
            int: Number of abstracts in the dataset.
        """
        return len(self.dataset)

    def __getitem__(
            self,
            index: int
        ) -> Dict[str, Any]:
        """Gets the row from the dataset at a specified index.

        Args:
            index (int): Index at which to return the element from the dataset.

        Returns:
            Dict[str, Any]: A dictionary with 4 fields:
                encoder_input:
                    Input to be fed to the encoder.
                    Tensor of dimension (context_size)
                encoder_mask:
                    Mask for the encoder, that will mask any padding tokens.
                    Tensor of dimension (1, 1, context_size)
                label:
                    Index of the true class.
                    Tensor of dimension ()
                abstract:
                    Original abstract text.
        """
        row = self.dataset[index]
        abstract = row['abstract']
        label = row['label']

        # Abstracts are frequently longer than context_size allows for; the
        # abstract body is truncated here rather than dropping the example,
        # since abstract length is unavoidably long-tailed and truncating
        # keeps more usable training data than rejecting overlong examples.
        max_body_tokens = self.context_size - 2 # room for [SOS] and [EOS]
        tokens = self.tokenizer.encode(abstract).ids[:max_body_tokens]

        num_padding_tokens = self.context_size - len(tokens) - 2

        # Encoder input is [SOS] token[1] token[2] ... token[K] [EOS] [PAD] [PAD] ... [PAD].
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        assert encoder_input.size(0) == self.context_size

        return {
            "encoder_input": encoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "label": torch.tensor(label, dtype = torch.int64),
            "abstract": abstract
        }
