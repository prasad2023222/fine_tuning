# fine_tuning

# BERT Fine-tuning for Question Answering

## 📋 Overview

This repository demonstrates how to fine-tune a pre-trained BERT model (`bert-base-uncased`) for the Question Answering task. The implementation uses the Hugging Face Transformers library and follows best practices for fine-tuning transformer models.

## 🎯 Objective

Fine-tune BERT to answer questions based on given context passages from the SQuAD dataset, which contains questions and answers about Wikipedia articles.

## 📚 What You'll Learn

- Loading and exploring the SQuAD dataset
- Understanding BERT architecture for Question Answering
- Preprocessing data for QA tasks (tokenization, feature preparation)
- Fine-tuning BERT using Hugging Face Trainer API
- Performing inference with fine-tuned models

## 🛠️ Requirements

```bash
pip install transformers datasets torch
```

### Key Libraries:
- **transformers**: For pre-trained models and training utilities
- **datasets**: For loading and processing the SQuAD dataset
- **torch**: PyTorch deep learning framework

## 📖 Dataset

**SQuAD (Stanford Question Answering Dataset)**
- A reading comprehension dataset
- Contains questions posed by crowdworkers on a set of Wikipedia articles
- Format: Question + Context → Answer (span of text)

### Dataset Structure:
```python
{
    "question": "Which NFL team represented the AFC at Super Bowl 50?",
    "context": "Super Bowl 50 was an American football game...",
    "answers": {
        "text": ["Denver Broncos"],
        "answer_start": [177]
    }
}
```

## 🔧 Key Components

### 1. Model Loading

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')
```

- **BERT-base-uncased**: 110M parameters, case-insensitive
- Pre-trained on masked language modeling and next sentence prediction
- Fine-tuned for QA with a QA head (start/end logits)

### 2. Data Preprocessing

The `prepare_train_features()` function:
- Tokenizes questions and contexts
- Handles long contexts with sliding window (stride=128, max_length=384)
- Maps character positions to token positions
- Creates start and end position labels for answer spans

**Key Features:**
- **Truncation**: Only context is truncated (not questions)
- **Stride**: Overlapping windows for long contexts
- **Offset Mapping**: Maps tokens back to original text positions

### 3. Training Configuration

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="finetune-BERT-squad",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
)
```

**Hyperparameters:**
- Learning Rate: 2e-5 (standard for BERT fine-tuning)
- Batch Size: 8 per device
- Epochs: 1 (sufficient for QA fine-tuning)
- Weight Decay: 0.01 (L2 regularization)

### 4. Training

```python
from transformers import Trainer, DefaultDataCollator

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=tokenized_datasets["train"].select(range(1000)),
    eval_dataset=tokenized_datasets["validation"].select(range(100)),
    data_collator=DefaultDataCollator(),
)

trainer.train()
```

## 🔍 How Question Answering Works

### Model Output:
BERT QA model outputs two logits:
- **start_logits**: Probability distribution for answer start position
- **end_logits**: Probability distribution for answer end position

### Inference Process:
1. Tokenize question + context
2. Get start and end logits from model
3. Find positions with highest probabilities
4. Extract token span and decode to text

```python
start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits) + 1
answer_tokens = input_ids[0][start_index:end_index]
answer = tokenizer.decode(answer_tokens)
```

## 📊 Workflow

```
1. Install Dependencies
   ↓
2. Load SQuAD Dataset
   ↓
3. Load Pre-trained BERT Model & Tokenizer
   ↓
4. Explore Dataset Structure
   ↓
5. Test Inference (Pre-fine-tuning)
   ↓
6. Prepare Training Features
   ├─ Tokenize questions and contexts
   ├─ Handle long contexts with stride
   └─ Create start/end position labels
   ↓
7. Fine-tune Model
   ├─ Configure training arguments
   ├─ Create Trainer
   └─ Train model
   ↓
8. Evaluate Fine-tuned Model
```

## 🚀 Usage

### Running the Notebook:

1. **Open the notebook** in Jupyter/Colab
2. **Install dependencies** (Cell 0)
3. **Run cells sequentially** to:
   - Load dataset and model
   - Explore data structure
   - Prepare features
   - Fine-tune model
   - Perform inference

### Quick Start Example:

```python
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')

# Prepare input
question = "Which NFL team represented the AFC at Super Bowl 50?"
context = "Super Bowl 50 was an American football game..."
inputs = tokenizer(question, context, return_tensors="pt")

# Get predictions
outputs = model(**inputs)
start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits) + 1
answer = tokenizer.decode(inputs['input_ids'][0][start_idx:end_idx])
```

## 📝 Key Concepts

### BERT Architecture:
- **Encoder-only** transformer model
- **12 layers**, 768 hidden dimensions (base)
- Uses **attention mechanism** to understand context
- **Positional encoding** for sequence understanding

### Fine-tuning Strategy:
- Keep pre-trained weights
- Add task-specific head (QA head)
- Train on downstream task (SQuAD)
- **Transfer Learning**: Leverage pre-trained knowledge

### Data Processing Challenges:
- **Long contexts**: Use sliding window with stride
- **Answer alignment**: Map character positions to token positions
- **Multiple features**: One example can produce multiple features

## 🎓 Educational Notes

The notebook includes educational explanations about:
- Transformer architecture vs RNN/LSTM
- How BERT processes sequences (parallel vs sequential)
- Difference between uncased models
- Understanding model outputs (logits, probabilities)

## 📈 Expected Results

After fine-tuning:
- Model learns to identify answer spans in context
- Better accuracy on SQuAD validation set
- Improved understanding of question-context relationships

## 🔗 Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [BERT Model Card](https://huggingface.co/bert-base-uncased)


## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is an educational notebook demonstrating BERT fine-tuning. For production use, consider:
- Using larger datasets
- Training for more epochs
- Hyperparameter tuning
- Model evaluation metrics (F1 score, exact match)
- Model optimization and deployment strategies

