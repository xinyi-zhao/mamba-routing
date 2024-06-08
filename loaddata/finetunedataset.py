import torch
import transformers
import json

from dataclasses import dataclass
from typing import Dict, Sequence
from tqdm import tqdm
from torch.utils.data import Dataset

class FinetuneDataset(Dataset):
    def __init__(self, tokenizer, prompts, labels, max_tokens = 600):
        super(FinetuneDataset, self).__init__()
        all_input_ids = []
        inputs = []
        for prompt, label in zip(prompts, labels):
            inputs.append(prompt + " " + label)
        prompt_tokenizer = tokenizer.batch_encode_plus(inputs, padding = True, return_tensors='pt', truncation = True, max_length = max_tokens)
        self.input_ids = prompt_tokenizer["input_ids"]
        self.labels = prompt_tokenizer["input_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollator(object):
    """
    Collate examples for supervised fine-tuning.
    """
    tokenizer: any
  
    def __call__(self, instances: Sequence[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and labels from the instances
        input_ids = [torch.tensor(instance['input_ids'], dtype=torch.long) for instance in instances]
        labels = [torch.tensor(instance['input_ids'], dtype=torch.long) for instance in instances]

        # Pad the sequences for input_ids
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # Pad the sequences for labels
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id) 
        # Create attention masks (1 for real tokens, 0 for padding)
        attention_mask = input_ids_padded.ne(self.tokenizer.pad_token_id)

        return {
            'input_ids': input_ids_padded,
            'labels': labels_padded,
            'attention_mask': attention_mask,
        }

class FinetuneDataModumn():
    def __init__(self, tokenizer, prompts, labels):
        self.dataset = FinetuneDataset(tokenizer, prompts, labels)
        self.data_collator = DataCollator(tokenizer=tokenizer)
        