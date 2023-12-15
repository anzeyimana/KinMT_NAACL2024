from typing import List

import torch
import torch.nn as nn
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
from torch.cuda.amp import custom_fwd


class BLEURTScore(nn.Module):
    def __init__(self, model_name='lucadiliello/BLEURT-20'):
        super(BLEURTScore, self).__init__()
        # config = BleurtConfig.from_pretrained(model_name)
        self.model = BleurtForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = BleurtTokenizer.from_pretrained(model_name)
        self.model.eval()
    @custom_fwd
    def forward(self, preds:List[str], targets:List[str]):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(targets, preds, padding='longest', return_tensors='pt')
            scores = self.model(**inputs).logits.flatten().tolist()
        return scores
