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

## Instruction finetuning

## Data preparation

## Training process

## Evaluation and iteration

## Consideration on getting started now

## Conclusion
