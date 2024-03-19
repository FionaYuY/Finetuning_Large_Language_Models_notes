# Finetuning_Large_Language_Models_notes

## Why finetune

1. Comparison of General Purpose vs. Specialized Models

   | General Purpose | Specialized                |
   |-----------------|----------------------------|
   | GPT-3           | ChatGPT                    |
   | GPT-4           | GitHub Copilot             |
   | PCP (Analogy)   | Cardiologist, Dermatologist|

2. What does finetuning do for the model?

   - Lets you put more data into than what fits into the prompt
   - Gets the model to learn the data instead of just accessing it
   - Steers the model to more consistent output
   - Reduces hallucinations
   - Customizes the model to a specific use case
   - The process is similar to the model's earlier training

3. Prompt Engineering vs. Finetuning

   | Prompting                                | Finetuning                                |
   |------------------------------------------|-------------------------------------------|
   | **Pros**                                 | **Pros**                                  |
   | No data to get started                   | Nearly unlimited data fits                |
   | Smaller upfront cost                     | Learn new information                     |
   | No technical knowledge needed            | Correct incorrect information             |
   | Connect data through retrieval (RAG)     | Less cost afterwards if smaller model     |
   |                                          | Use RAG too                               |
   | **Cons**                                 | **Cons**                                  |
   | Much less data fits                      | More high-quality data                    |
   | Forgets data                             | Upfront compute cost                      |
   | Hallucinations                           | Needs some technical knowledge, esp. data |
   | RAG misses, or gets incorrect data       |                                           |

4. Benefits of finetuning your own LLM

   - **Performance:**
     - Stop hallucinations
     - Increase consistency
     - Reduce unwanted info

   - **Privacy:**
     - On-prem or VPC
     - Prevent leakage
     - No breaches

   - **Cost:**
     - Lower cost per request
     - Increased transparency
     - Greater control

   - **Reliability:**
     - Control uptime
     - Lower latency
     - Moderation

5. Tools for finetuning in this project: Pytorch(Meta), Huggingface, Llama library(Lamini)

## Where finetuning fits in

1. Pretraining
   - Model starts with zero knowledge about the world, unable to form English words
   - Trains on next token prediction with a giant corpus of text, often scraped from the internet
   - After training, the model learns language and knowledge

2. Data scraped from the internet
   - Often, pretraining data isn't publicized
   - Open-source pretraining data: 'The Pile'
   - Expensive and time-consuming to train

3. Limitations of pretrained base models
   - Example Input: 'What is the capital of Mexico?'
   - Example Output: 'What is the capital of Hungary?'
   - Not useful for a chatbot interface

4. Finetuning after pretraining
   - Base Model through Pretraining -> Finetuning -> Finetuned Model
   - Finetuning often refers to further training on a much smaller dataset

5. Finetuning outcomes
   - Changes behavior, teaching the model to respond more consistently
   - Improves knowledge on specific concepts
   - Increases model's overall capabilities

6. Tasks suitable for finetuning
   - Text-in, text-out tasks like extraction and expansion
   - Task clarity is a key indicator of success

7. First steps in finetuning
   - Identify tasks by prompt-engineering a large LLM
   - Curate about 1000 input-output pairs
   - Finetune a small LLM on this data

8. Read the file: `Where_finetuning_fits_in.ipynb`

## Instruction finetuning

1. Lab goal: Give chatting power to all models
2. Instruction finetuning aims to improve model behavior, making it a better chatbot interface

3. Datasets for instruction-following
   - FAQs, customer support conversations, Slack messages, and more can be used

4. LLM Data Generation
   - Convert non-Q&A data to Q&A using prompts or other LLMs

5. Generalization in Instruction Finetuning
   - Accesses pre-existing knowledge and generalizes learning to new data

6. Overview of the process
   - Data Prep -> Training -> Evaluation -> Repeat

7. Types of Finetuning and LoRA
   - Different methods for adapting the model to new tasks, including LoRA for low-rank adaptation

8. Read the file: `Instruction_tuning.ipynb`

## Data preparation

1. What kind of data?

   | Better        | Worse        |
   |---------------|--------------|
   | Higher Quality| Lower Quality|
   | Diversity     | Homogeneity  |
   | Real          | Generated    |
   | More          | Less         |

2. Steps to prepare your data:

   - Collect instruction-response pairs
   - Concatenate pairs (add prompt template, if applicable)
   - Tokenize: Pad, Truncate
   - Split into train/test

3. Tokenizing your data

   - There are multiple popular tokenizers
   - Use the tokenizer associated with your model

4. Read the file: `Data_preparation.ipynb`

## Training process

1. Training: same as other neural networks

2. What's going on?

   - Add training data
   - Calculate loss
   - Backpropagate through the model
   - Update weights

3. Hyperparameters

   - Learning rate
   - Learning rate scheduler
   - Optimizer hyperparameters

4. Training process code example

```
for epoch in range(num_epoch):
   for batch in train_dataloader:
      outputs = model(**batch)
      loss = output.loss
      loss.backward()
      optimizer.step()
```

5. Read the file: `training.ipynb`

## Evaluation and iteration

1. Evaluating generative models is notoriously difficult

- Human expert evaluation is most reliable
- Good test data is crucial
- Elo comparisons also popular (like chess)

2. LLM Benchmarks: Suite of Evaluation Methods

- Common LLM Benchmarks include:
  - ARC: A set of grade-school questions
  - HellaSwag: A test of common sense
  - MMLU: A multitask metric covering various subjects
  - TruthfulQA: Measures a model's propensity to reproduce falsehoods

3. Error Analysis

- Understand base model behavior before finetuning
- Categorize errors: iterate on data to fix these problems in data space

| Category    | Example with Problem                                         | Example Fixed                                       |
|-------------|--------------------------------------------------------------|-----------------------------------------------------|
| Misspelling | "Your kidney is healthy, but your lever is sick. Go get your lever checked." | "Your kidney is healthy, but your liver is sick."   |
| Too long    | "Diabetes is less likely when you eat a healthy diet because eating a healthy diet makes diabetes less likely, making..." | "Diabetes is less likely when you eat a healthy diet." |
| Repetitive  | "Medical LLMs can save healthcare workers time and money and time and money and time and money." | "Medical LLMs can save healthcare workers time and money." |

## Consideration on getting started now

1. Practical approach to finetuning

- Figure out your task
- Collect data related to the task's inputs/outputs
- Generate data if you don't have enough
- Finetune a small model (e.g., 400M-1B)
- Vary the amount of data you give the model
- Evaluate your LLM to know what's going well vs. not
- Collect more data to improve
- Increase task complexity
- Increase model size for performance

2. Tasks to finetune vs. model size

- Complexity: more tokens out is harder
  - Extract: 'reading' is easier
  - Expand: 'writing' is harder
- Combination of tasks is harder than one task
- Larger models are needed for harder or more general tasks

3. Model sizes x Compute

| AWS Instance  | GPUs    | GPU Memory  | Max inference size (# of params) | Max training size (# of tokens) |
|---------------|---------|-------------|----------------------------------|---------------------------------|
| p3.2xlarge    | 1 V100  | 16GB        | 7B                               | 1B                              |
| p3.8xlarge    | 4 V100  | 64GB        | 7B                               | 1B                              |
| p3.16xlarge   | 8 V100  | 128GB       | 7B                               | 1B                              |
| p3dn.24xlarge | 8 V100  | 256GB       | 14B                              | 2B                              |
| p4d.24xlarge  | 8 A100  | 320GB HBM2  | 18B                              | 2.5B                            |
| p4de.24xlarge | 8 A100  | 640GB HBM2e | 32B                              | 5B                              |

4. PEFT: Parameter-Efficient Finetuning Techniques
   - PEFT methods enable finetuning with fewer trainable parameters, which can be beneficial for smaller datasets or when computational resources are limited.

5. LoRA: Low-Rank Adaptation of LLMs
   - LoRA aims to reduce the number of trainable parameters during finetuning.
   - It uses rank decomposition to adapt certain layers of the model while keeping others frozen.
   - This can lead to less memory usage during training and may maintain similar performance levels to full model finetuning with less computational cost.

6. Getting started with finetuning now:
   - Starting with a clear task definition and data collection is crucial.
   - Consider using PEFT or LoRA based on your specific needs and constraints.

7. Model sizes and compute considerations:
   - Bigger models can handle more complex tasks and generalize better but require more computational resources.
   - Smaller models are cheaper and faster to train but may not perform as well on complex or general tasks.

