# Finetuning_Large_Language_Models_notes

## Why finetune
1.
| General Purpose | Specialized     | 
| --------------- | --------------- | 
| GPT-3           | ChatGPT         |
| GPT-4           | GitHub Copilot  |
| PCP (Analogy)   | Cardiologist, Dermatologist| 

2. WHat does finetuning do for the model?
   - Lets you put more data into than what fits into the prompt
   - Gets the model learn the data instead of just access to it
   - Steers the model to more consistent output
   - Reduces hallucinations
   - Customizes the model to a specific use case
   - Process is similar to the model's earlier training
3. Prompt Engineering vs. Finetuning
   
| Prompting                        | Finetuning                                              |
|----------------------------------|---------------------------------------------------------|
| **Pros**                         | **Pros**                                                |
| No data to get started           | Nearly unlimited data fits                              |
| Smaller upfront cost             | Learn new information                                   |
| No technical knowledge needed    | Correct incorrect information                           |
| Connect data through retrieval (RAG) | Less cost afterwards if smaller model                    |
|                                  | Use RAG too                                             |
| **Cons**                         | **Cons**                                                |
| Much less data fits              | More high-quality data                                  |
| Forgets data                     | Upfront compute cost                                    |
| Hallucinations                   | Needs some technical knowledge, esp. data               |
| RAG misses, or gets incorrect data |                                                         |

4. Benefits of finetuning your own LLM
   - Performance:
       * stop hallucinations
       * increase consistency
       * reduce unwanted info
   - Privacy:
       * on-perm or VPC
       * prevent leakage
       * no breahces
    - Cost:
       * Lower cost per request
       * increased transparency
       * greater control
    - Reliability:
       * control uptime
       * lower latency
       * moderation
5. What we'll be using to finetune in this project: Pytorch(Meta), Huggingface, Llama library(Lamini)
## Where finetuning fits in
1. Pretraining
   - Model at the start:
       * Zero knowledge about the world
       * Can't form English words
   - Next token prediction
   - Giant corpus of text data
   - Often scraped from the internet: 'unlabeled'
   - Self-supervised learning
   - After training:
       * Learns language
       * Learns knowledge
2. What is 'data scrpaed from the internet'
   - Often not publicized how to pretrain
   - Open-source pretraining data: 'The Pile'
   - Expensive & time-consuming to train
3. Limitations of pretrained base models
   Input : 'What is the capital of Mexico?'
   Output : 'What is the capital of Hungary?'
   Not useful from the sense of a chatbot interface
4. Finetuning after pretraining
   Pre-trainig -> Base Model -> Finetuning -> Finetuned Model
   - Finetuning usually refers to training further
     * Can also be self-supervised unlabeled data
     * Can be 'labeled' data you curated
     * Much less data needed
     * Tool in your toolbox
   - Finetuning for generative tasks is not well-defined:
     * Updates entire model, not just part of it
     * Same training objective as pretraining : Next token prediction
     * More advanced ways reduce how much to update
5. What is finetuning doing for you?
   - Behavior change
     * Learning to respond more consistently
     * Learning to focus, ex: moderation
     * Teasing out capability, ex: better at conversation
   - Gain knowledge
     * Increasing knowledge of new specific concepts
     * Correcting old incorrect information
   - Both
6. Tasks to finetune
   - Just text-in, text-out
     * Extraction: text in, less text out
       + 'Reading'
       + Keywords, topics, routing, agents(planning, reasoning, self-critic, tool use), etc
     * Expansion: text in, more text out
       + 'Writing'
       + Chat, write emails, write code
   - Task clarity is key indicator of success
   - Clarity means knowing what's bad vs. good vs. better
7. First time finetuning:
   Identify task(s) by prompt-engineering a large LLM ->
   Find tasks that you see an LLM doing ~OK at ->
   Pick one task ->
   Get ~1000 inputs and outputs for the task (Better than the ~OK from the LLM) ->
   Finetune a small LLM on this data
8. Read the file: Where_finetuning_fits_in.ipynb
## Instruction finetuning
1.  Goal for this lab: Give the chatting power to all of the models
2.  What is instruction finetuning?
    - Finetuning includes reasoning, routing, copilot, chat, agents. Instruction finetuning is one of it.
    - AKA 'instruction-tuned' or 'instruction-following' LLMs
    - Teaches model to behave more like a chatbot
    - Better user interface for model interaction
      * Turned GPT-3 into ChatGPT
      * Increase AI adoption, from thousands of researchers to millions of people
3. Instruction-following datasets
   - Some existing data is ready as-is, online:
     * FAQs (Frequently asked questions)
     * Customer support conversatons
     * Slack messages
4. LLM Data Generation
   - Non-Q&A data can also be converted to Q&A
     * Using a prompt template
     * Using another LLM
     * ChatGPT('Alpaca')
     * Open source model
5. Instruction Finetunig Generalization
   - Can access model's pre-existing knowledge
   - Generalize following instructions to other data, not in finetuning dataset
   - Questions not in finetuning data, only base data, model can also answer after finetuning.
6. Overview
   Data Prep -> Training -> Evaluation -> repeat...
7. Different Types of Finetuning: 'Data prep' is where differentiates the different types of finetunes.
8. Read the file : Instruction_tuning.ipynb
## Data preparation
1. What kind of data?

   |Better |Woser|
   |-------|--------|
   |Higher Quality| Lower Quality|
   |Diversity|Homogeneity|
   |Real|Generated|
   |More|Less|
2. Steps to prepare your data:
   Collect instruction-response pairs->
   Concatenate pairs(add prompt template, if applicable)->
   Tokenize: Pad, Truncate->
   Split into train/test
3. Tokenizing your data
   There are multiple popular tokenizers:
   - Use the tokenizer associated with your model
4. Read the file: Data_preparation.ipynb
## Training process
1. Training: same as other neural networks
2. What's goint on?
   - Add training data
   - Calculate loss
   - Backprop through model
   - Update weights
3. Hyperparameters
   - Learning rate
   - Learning rate scheduler
   - Optimier hyperparameters
4. Run through general chunks of training process in code
```
for epoch in range(num_epoch):
   for batch in train_dataloader:
      outputs = model(**batch)
      loss = output.loss
      loss.backward()
      optimizer.step()
```
5. Read the file: 
## Evaluation and iteration

## Consideration on getting started now

## Conclusion
