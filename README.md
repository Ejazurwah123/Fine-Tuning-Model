# Fine-Tuning-Model
Fine Tuning AI Models available on HuggingFace to acheive specific tasks

Fine-tuning a pre-trained model can be highly effective for achieving specific targets, such as generating relevant questions based on CV sections. Here is a step-by-step guide for fine-tuning a model using the HuggingFace Transformers library:

### Step-by-Step Guide for Fine-Tuning a Model

#### 1. **Set Up Environment**

Make sure you have Python and pip installed. Then, install the necessary libraries:

```bash
pip install transformers datasets torch
```

#### 2. **Prepare Your Data**

Create or collect a dataset suitable for your task. For question generation based on CVs, your dataset should include pairs of inputs (CV sections) and desired outputs (questions).

Example format:
```json
{
  "data": [
    {"input": "Section text here", "output": "Generated question here"},
    {"input": "Another section text", "output": "Another generated question"}
  ]
}
```

#### 3. **Load and Preprocess the Dataset**

Use the `datasets` library to load and preprocess your dataset.

```python
from datasets import load_dataset, Dataset
import pandas as pd

# Create a DataFrame from your data
data = [
    {"input": "Section text here", "output": "Generated question here"},
    {"input": "Another section text", "output": "Another generated question"}
]
df = pd.DataFrame(data)

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)

# Split dataset into train and validation sets
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
val_dataset = dataset['test']
```

#### 4. **Choose a Pre-trained Model**

Select a pre-trained model suitable for your task. For question generation, models like `t5-small` or `t5-base` are good starting points.

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
```

#### 5. **Tokenize the Data**

Tokenize your dataset to prepare it for the model.

```python
def tokenize_data(example):
    source = example['input']
    target = example['output']
    source_tokenized = tokenizer.encode(source, truncation=True, padding='max_length', max_length=512)
    target_tokenized = tokenizer.encode(target, truncation=True, padding='max_length', max_length=128)
    return {"input_ids": source_tokenized, "labels": target_tokenized}

train_dataset = train_dataset.map(tokenize_data)
val_dataset = val_dataset.map(tokenize_data)

train_dataset.set_format(type='torch', columns=['input_ids', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'labels'])
```

#### 6. **Set Up Training Arguments**

Define the training parameters.

```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True
)
```

#### 7. **Create a Trainer**

Create a `Seq2SeqTrainer` to handle training and evaluation.

```python
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)
```

#### 8. **Train the Model**

Start the training process.

```python
trainer.train()
```

#### 9. **Evaluate the Model**

Evaluate your model to check its performance.

```python
results = trainer.evaluate()
print(results)
```

#### 10. **Save the Model**

Save the fine-tuned model for future use.

```python
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')
```

### Additional Tips

- **Experiment with Hyperparameters:** Adjust learning rates, batch sizes, and the number of epochs to optimize performance.
- **Data Augmentation:** Use data augmentation techniques to increase the size and diversity of your dataset.
- **Early Stopping:** Implement early stopping to prevent overfitting.
- **Regular Evaluation:** Evaluate the model at regular intervals to monitor its progress.

This guide should help you fine-tune a model using the HuggingFace Transformers library. Feel free to adjust the steps and parameters according to your specific use case and requirements.
