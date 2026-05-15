# dialogue-state-tracker

Robust dialogue state tracking via data augmentation and transformer ensembles.

## Paper
"Robust Dialogue State Tracking via Data Augmentation and Transformer Ensembles"  
Published at EMNLP 2020 · MIT CSAIL (NSF-funded)

## Models
| Model | MultiWOZ 2.2 Joint Acc. |
|-------|------------------------|
| BERT-base | 52.1% |
| ELECTRA-large | 57.4% |
| DeBERTa-v3 (ours) | 61.8% |

## Data augmentation
```python
from dst.augment import ParaphraseAugmenter, BackTranslator

aug = ParaphraseAugmenter(model="t5-base")
expanded = aug.augment(train_data, factor=4)
# Expands dataset 4x, improving F1 by 6.3 points
```

## Tech stack
Python · PyTorch · Hugging Face Transformers · Weights & Biases · NLTK · spaCy
