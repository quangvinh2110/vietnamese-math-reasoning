from typing import List
import torch
from transformers import StoppingCriteria, AutoTokenizer


class StopSequenceCriteria(StoppingCriteria):
    def __init__(self, stop_sequences: List[str], tokenizer: "AutoTokenizer"):
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        

    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor, 
        **kwargs
    ) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0].cpu()[-10:])
        for stop_sequence in self.stop_sequences:
            if self.stop_sequence in generated_text:
                return True
        return False
    
    
    def __len__(self):
        return 1

    
    def __iter__(self):
        yield self
