# AI Concepts

# Table of Contents

1. [Large Language Model](#1-large-language-model-llm)  
2. [Tokenization](#2-tokenization)  
3. [Vectorization](#3-vectorization)  
4. [Attention](#4-attention)  
5. [Self-Supervised Learning](#5-self-supervised-learning)  
6. [Transformer](#6-transformer)  
7. [Fine-Tuning](#7-fine-tuning)  
8. [Few-shot Prompting](#8-few-shot-prompting)  
9. [Retrieval Augmented Generation (RAG)](#9-retrieval-augmented-generation-rag)  
10. [Vector Database](#10-vector-database)  
11. [Model Context Protocol (MCP)](#11-model-context-protocol-mcp)  
12. [Context Engineering](#12-context-engineering)  
13. [Agents](#13-agents)  
14. [Reinforcement Learning](#14-reinforcement-learning)  
15. [Chain of Thought](#15-chain-of-thought)  
16. [Reasoning Models](#16-reasoning-models)  
17. [Multi-model Models](#17-multi-model-models)  
18. [Small Language Models](#18-small-language-models)  
19. [Distillation](#19-distillation)  
20. [Quantization](#20-quantization)  

---

## (1) Large Language Model (LLM)

A **nueral network trained** to predict the next term or input in a sequence.  

## (2) Tokenization

Processing input of llm into tokens. In the example below the 'ing' tells the model it the prefix to -ing is an action being performed. In this case the action of dance.

ex. "Lets Go Dancing" -> "lets", "go", "dance", "ing"

Tokenization is an essential part of how models understand human language.  

End Result: Input text broken into tokens

Dive Deeper -> [Coming-Soon]() Fundamental process of tokenizing an input.

## (3) Vectorization

Tokens tell what to focus on. Vectors is what meaning is derived.

Groups of tokens or words with similar meanings are clustered together in the blackbox with a coordinate.  
The coordinate in the blackbox is called a vector.  

By understanding smaller tokenized meanings of words model can create sentences.

End Result: Understand words in the english dictionary. through tokenization then vectorization. and the ability to take input text and process it.

*Dive Deeper* -> [Coming-Soon]() Process of vectorizing data

## (4) Attention

Mechanism.  
Use nearby words to give descriptive or identifying words context

operation = what + context   

ex.

`apple + revenue = company earnings`  
vs   
`apple + tasty = fruit flavor`  

This attention operation gives sentences meaning. (2017)   
This also allows LLMs to create coherent sentences. (2022)  

## (5) Self-Supervised Learning

2017 concept  
looking at text already existing in the world and created multiple challenges for yourself without human intervention

- makes test data cheaper. 
- incorrect means loss in training. 
- establishes an inherent structure of input of output

## (6) Transformer

> cleanup wording, strucutre, and overall presentation of information

NOT LLM  
LLM -> predicts next token given an input sequence
Transformer -> same thing but specific algorithm or method by which you predict the next token

Transformer = 
Input Tokens ->  
Attention Block(disambiguate terms) -> Feed Forward Neural Network -> [v for v in output-vectors] -> Attention(complex relationships, sarcasm, implications) -> Feed Forward Neural Network -> and so on... ->  
till confident enough to generate output

each loop is a layer  
think GPT archtectures are hundreds of layers

gets meaning and manipulate again and again to predict next token 

Performance: `O(n^2)` this is the engine of the models car

## (7) Fine-Tuning
> cleanup wording

Train Base Model in self-supervised fashion.  
Then after series of Q&A called Fine-Tuning.  
This forces models to take questions and answer as expected.  

Ex. The same model can have two separate fine-tune Q&As one to create a financial fine tuned model and a separate medical fine tuned model

*Dive Deeper* -> [Fine Tuning Models](/AI_ML_Knowledge/Fine_Tuning.md)  

## (8) Few-shot Prompting

Before sending vanilla query to LLM you augment the query.  

Add examples in prompts for LLM to go through.  

Improves model response quality.  

Also known as Example Prompting.

*Dive Deeper* -> [Coming-Soon]()  

## (9) Retrieval Augmented Generation (RAG)

Gives Input + Example Prompting + Relevant Documents to LLM

Direct User Input + How LLM Should Output + Context

This tends to give models high quality responses

Database type DOES NOT matter data is still fetched vectorized stored and fed as context to the model [In-Memory, GraphDB, VectorDB]

Retrieve Context -> Augment The Query -> Generate A Response

*Dive Deeper* -> [RAG](/AI_ML_Knowledge/RAG.md)  

## (10) Vector Database

Used to find relevant documents for incoming query
It helps to make connections in sentences for example if it sees "upset" and "refund" it can assume if given refund feeling of upset will go away  

Vectors can encapsulate semantic meaning, so documents related to upset will be close to drop-off and low-ratings which when retrieved add context to the model

So take user query and search in vectorDB for document closet to query and add as context to LLM

So query + vectordb context + (optional)system prompt ---> LLM

Context Documents are stored in vectorDB to perform similarity searches efficiently because it is stored with hierarchical algorthims.

VectorDB is a black box you store documents and can quickly retrieve them

*Dive Deeper* -> [Coming-Soon]()  

## (11) Model Context Protocol (MCP)

VectorDB handles INSIDE system context. MCP handles OUTSIDE system context.  
Way to communicate to transfer context into a model.  

Before receiving query from user, model has mcp client, which now can give context to something like American Airlines MCP gives information on american airlines flights etc. 

Its an external database wrapper for its context to LLM to make better decisions.  

*Dive Deeper* -> [Coming-Soon]()  

## (12) Context Engineering

Combination of:
- Few Short Prompting(examples) 
- RAG (relevant docs from vectordb for context)
- MCP (external servers for outside context and actions as needed)

User preferences & 
Context summarization
- example: use sliding window where last 100 chats are sent to llm and the rest are summarized into 5 sentences. To limit max amount of chats. 
- some do 1 chat and the rest summary. 
- some focus on keywords.

With new document summarize with cheaper small language model or distilled model and generate the context then send to LLM

Context vs. Prompt Engineering
- Context: evolves as user declares preferences, previous chat history, new data, etc.
- Prompt: one single prompt, stateless, system prompts stay the same

*Dive Deeper* -> [Coming-Very-Soon]()  

## (13) Agents

Its a server getting an API call.  
Long Running Process that can call LLM, query external systems, and other agents.  

Agent can have access to an LLM and MCP Clients
Ex. A booking agent has serparte MCP clients for Flights and Hotels

*Dive Deeper* -> [Agents](/AI_ML_Knowledge/Agents_.md)  

## (14) Reinforcement Learning

Train models to behave in particular way.  

If you give a query to a model it can generate two responses. Chosen response gets a +1, other gets -1

Took user query as vector and then go to the coordinate and generate more tokens. think points on a graph progressively moving through 1 by 1. Thats the path, when a response is given +1 each point on the coordinate graph gets a plus 1 to indicate correct paths. worse response has the same happen but minus 1 to discorage that search path.

Creates neutral, positive, and negative paths. Helps to guide the model to make better decisions the more positive, the more the model wants to go there.

*Dive Deeper* -> [Reinforcement Learning](/AI_ML_Knowledge/Reinforcement_Learning.md)  

## (15) Chain of Thought

When training a model, clearly define the thought process of how you expect the model to perform a problem.
Because its trained to reason step by step we call this Chain of Thought

Think of it as a series of deductions or inferences the model must make to improve the response quality.

In the step by step breakdown the model can add new steps as it sees fit, based on training data as problems gets more difficult. This has been observed with DeepSeek Models. It will give less steps to easy problems and more steps for more difficult problems.

This is how the model is trained on complex math.

There are more ways to develop thought -> [Giving Models Thought]()

## (16) Reasoning Models 

A model that can figure out how to solve a problem step by step is a reasoning model.
Or LRMs -> Deepseek, OpenAIo1,o2,o3...

## (17) Multi-modal Models

most operate on text. some can accept read and create images, videos. 
just means multiple was of understanding (text, images, video) with more training on images, videos, etc.

## (18) Small Language Models

Companies want specific closed data models.  
Fewer parameters than large (3-300 Million params with fewer widths)(LLM is 3-300 Billion)

Trained usually on company or task specific data. Like a model just for sales could sell anything but is terrible at weather analysis.

*Dive Deeper* -> [SLM]()  

## (19) Distillation

The process of building small language models.  

Send input to LLM and to SLM and both give predicted output.  
The Student SLM tried to match the Teacher LLM.  
If the two outputs match, then the SLM is doing well, else it changes the internal weights.  
Goal is to condense information from complex neural network into the most reasonable representation of performance but with significantly reduced costs.  
SLM will be much faster in inference time during production and is alot easier to host.

*Dive Deeper* -> [Distillation of SLM]()  

## (20) Quantization

After training you take the weights(branches between neurons) are reduced from 32-bit to 8-bit. This reduces inference cost during production since 75% of memory from weights are saved.

#### Sources

- [20 AI Concepts Explained in 40 Minutes](https://www.youtube.com/watch?v=OYvlznJ4IZQ)
- [ADK + MCP + RAG + Ollama](https://gaodalie.substack.com/p/google-adk-mcp-rag-ollama-the-key)
- [ToFind-fine-tuning-models-source]()
- To Find: prompt crafting, context crafting, reinforcement learning, chain-of-thought