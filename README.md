# Cortado AI Agent Challenge

## Introduction


This repository is for a technical test for the Cortado AI ML/AI engineer position, focusing on developing an agent capable of accessing PDF and JSON documents to respond to evaluation questions. The project implements Retrieval-Augmented Generation (RAG) and utilizes various agent tools to enhance data processing and response accuracy.

The Cortado AI Agent is designed to assist customers in managing rentals and hospitality services efficiently by leveraging advanced AI capabilities. The agent processes and responds to user queries, analyzes documents, and manages data. The integration of large language models (LLMs) and sophisticated data processing tools achieve high relevance and accuracy in responses, enhancing user experience and operational efficiency.

For a comprehensive understanding of the project and the role, please refer to the following resources:
- Cortado AI ML/AI Engineer Job Description: [link](https://cortado-ai.notion.site/Cortado-AI-ML-Engineer-971a3a3b3b154582bf7cdf1f091788ab)
- Cortado AI Hiring Challenge Documentation: [link](https://cortado-ai.notion.site/Cortado-Hiring-Challenge-Lead-AI-ML-8a0646cb2b3e48508fc7f2ac85afc2b1?pvs=4)
- Two demonstration videos:
  - One resolving the proposed questions from Cortado using GPT-4 in the following [link](https://drive.google.com/file/d/1PHgz0zGtGlsvKsYbfztbhWvXRprofdMk/view?usp=sharing)
  - Another showcasing the agent's memory capabilities and handling a two-step question for a challenging edge case, with GPT-3.5-turbo, in this [link](https://drive.google.com/file/d/19W4nm_B7rlVnNkKC4pBFYTl53EGXWUMo/view?usp=sharing)
- PDF for the introduction and extended details in the challenge development: [link](https://drive.google.com/file/d/1W7It2YxMXHwLMX8UwDiRdfpoi5AS3mHt/view?usp=share_link)

### Technical Background

The core of the Cortado AI Agent is built upon the integration of LlamaIndex, a versatile framework designed to augment LLMs with custom data sources, agents and tools. This enables the system to ingest, structure, and access domain-specific data efficiently. The system uses OpenAI's LLMs for natural language understanding and HuggingFace embeddings for semantic search and retrieval.

### Implementation Goals

- **Efficient Data Management**: Streamline rental and hospitality management by processing and retrieving data from JSON and PDF files.
- **Advanced Query Handling**: Use vector tools and JSON analysis tools to provide accurate and comprehensive responses to user queries.
- **Modular Architecture for RAG Capabilities**: Design a backend framework that supports the integration of diverse tools and APIs, enabling flexible implementation of Retrieval-Augmented Generation (RAG) functionalities for enhanced data processing and response accuracy.

## Installation and Setup

### Requirements

- Python 3.8+

### Install Required Packages

It is recommended to use a virtual environment to manage dependencies. To set up a virtual environment and install the required packages, run:

```bash
python3 -m virtualenv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```

### Configuration

#### Export API Key

Before running the scripts, make sure to export your OpenAI API key:

```bash
export OPENAI_API_KEY='your_openai_api_key'
```

#### Configure Parameters

Ensure that all parameters in the configuration file are correctly set according to your environment.

## Running the Evaluation

To run the evaluation, execute the following command:

```bash
python evaluation.py
```

## Code Structure and Modules

### `agent_backend.py`

This is the main file that initializes and runs the agent backend.

#### Classes and Methods

- **AgentBackend**: Initializes configurations, components, tools, memory, and the agent runner.
  - `__init__()`: Initializes the backend with configurations.
  - `initialize_components() -> Tuple[OpenAI, HuggingFaceEmbedding]`: Initializes the LLM and embedding model.
  - `setup_settings()`: Sets up global settings for LLM and embedding model.
  - `initialize_vector_tool() -> VectorTool`: Initializes the vector tool.
  - `initialize_jsonalyze_tools() -> List[JSONalyzeTool]`: Initializes JSONalyze tools.
  - `initialize_memory() -> SimpleComposableMemory`: Initializes memory.
  - `create_agent_runner() -> AgentRunner`: Creates the agent runner.
  - `process_question(question: str) -> str`: Processes a question and returns the response.

### `vector_tool.py`

Handles the vector tool which provides detailed welcome documentation from PDF documents.

#### Classes and Methods

- **VectorTool**: Initializes the vector tool with configurations, LLM, and embedding model.
  - `__init__(config: Dict, llm: OpenAI, embed_model: HuggingFaceEmbedding)`: Initializes the vector tool.
  - `ensure_index_construction()`: Ensures the vector index is constructed and stored.
  - `load_memory_index() -> VectorStoreIndex`: Loads the memory index from storage.
  - `initialize_vector_tool() -> QueryEngineTool`: Initializes the vector tool with a query engine and metadata.

### `json_tool.py`

Handles JSONalyze tools which provide structured information about rental properties from JSON files.

#### Classes and Methods

- **JSONalyzeTool**: Initializes the JSONalyze tool with configurations, LLM, and JSON file.
  - `__init__(config: Dict, llm: OpenAI, json_file: str)`: Initializes the JSONalyze tool.
  - `initialize_jsonalyze_tool() -> QueryEngineTool`: Initializes the JSONalyze tool with a query engine and metadata.

### `config.py`

This module provides a comprehensive configuration dictionary that is essential for the entire system's operation. It allows for highly parametrized modules and components to be easily configured through a centralized config file. Key elements include parameters for various models (LLMs), prompts for guiding the agent's responses, and metadata for tools used in processing. This flexibility ensures that the system can be tuned to specific requirements and easily adjusted without modifying the underlying codebase.

#### Methods

- `get_config() -> dict`: Returns the configuration dictionary.

### `evaluation.py`

The evaluation script is designed to test the agent's functionalities and correctness. It uses the DeepEval library to assess the performance of the agent's responses. The GEval metrics from DeepEval are customizable and parametrizable, allowing for tailored evaluations based on specific requirements which are also tracked in JSON files, facilitating analysis and tracking of performance over time.

#### Classes and Methods

- **AgentEvaluator**: Manages the evaluation process.
  - `__init__(config: Dict[str, Any])`: Initializes the evaluator with configuration and agent backend.
  - `create_geval_metric(metric_info: Dict[str, Any]) -> GEval`: Creates a GEval metric.
  - `evaluate_answers(questions: List[str], expected_answers: List[str], responses: List[str]) -> List[Dict[str, Any]]`: Evaluates answers using relevancy and custom GEval metrics.
  - `serialize_tool_output(output: Any) -> Dict[str, Any]`: Serializes the tool output.
  - `serialize_agent_response(response: Any) -> Dict[str, Any]`: Serializes the agent response.
  - `process_agent_responses(responses: List[Any]) -> Dict[str, Any]`: Processes agent responses for serialization.
  - `save_to_json(data: Dict[str, Any], filename: str)`: Saves data to a JSON file.
  - `run_evaluation()`: Runs the evaluation process.

## Running the Backend

To start the backend and process a question, execute the following:

```bash
python agent_backend.py
```

The user can input a question to the agent from the keyboard. Process runs until user types "exit".

Example question:

```bash
What are the house rules?
```

## Data Sources

### Directory Structure

```
data/
│
├── json/
│   ├── listing_object.json
│   ├── prior_conversations.json
│   ├── listing_object.pkl
│   └── prior_conversations.pkl
│
├── pdf/
│   └── welcome_packet (12).pdf
│
├── evaluation/
│   └── (stores evaluation run outputs)
│
├── processed_responses/
│    └── (stores processed agent execution outputs)
│
├── questions.json
```

### Data Files

- **JSON Files**:
  - `json/listing_object.json`: Contains detailed information about rental properties.
  - `json/prior_conversations.json`: Stores prior conversation data related to the rental properties.
  - `json/listing_object.pkl` and `json/prior_conversations.pkl`: Used for storing lists of dictionaries for the JSON query engine. These files use SQLite to build a database that can be queried.

- **PDF Files**:
  - `pdf/welcome_packet (12).pdf`: Contains the welcome packet and house manual for the rental properties.

- **Evaluation questions**: 
  - `questions.json`: Contains evaluation questions for the agent.


## Functionalities

- **Efficient Rental and Hospitality Management**: The system is designed to streamline the management of rental properties and hospitality services, enabling property owners and managers to efficiently handle inquiries, bookings, and guest interactions. It automates various tasks, reducing manual effort and enhancing operational efficiency.

- **Comprehensive Responses Based on PDF Documents and JSON Files**: The agent leverages a combination of RAG tools to retrieve structured data from JSON files and unstructured information from PDF documents to provide detailed and accurate responses to user inquiries. 

- **Memory Buffer for Chat Context**: The implementation includes a sophisticated memory buffer that retains chat history and context, allowing the agent to provide more personalized and contextually relevant responses. Memory buffer is optional for a more conversational agent, so also can be disabled if the task 
does not require a memory buffer.

## Implementation

- **OpenAI LLMs**: The core of the agent's functionality is powered by OpenAI's advanced language models which enables it to understand and generate human-like text. The models can interact with different RAG tools for agentic capabilities in conversational tasks, making it adept at handling a wide range of inquiries and retrieve the correct documentation.

- **Employs HuggingFace Embeddings for Semantic Search and Retrieval**: The system incorporates HuggingFace's embedding models to facilitate semantic search capabilities, with the ability of implementing different embedding models. This allows the agent to retrieve relevant information based on the meaning of the queries rather than just keyword matching, improving the accuracy of responses.

- **Configurable Settings for Chunking, Vector Tools, and JSONalyze Tools**: The architecture supports customizable settings that allow users to define parameters for data chunking, vector tool configurations, and JSON analysis. This flexibility ensures that the system can be adapted to meet specific operational needs and optimize performance based on the context of use.

## Code

- `agent_backend.py`: Main entry point.
- `vector_tool.py`: Handles vector-based document retrieval.
- `json_tool.py`: Handles JSON-based data retrieval.
- `config.py`: Provides configuration settings.
- `evaluation.py`: Manages the evaluation of agent responses.

## Modules

- **LLM and Embeddings**: OpenAI GPTs through API and HuggingFace embeddings.
- **Memory**: SimpleComposableMemory for managing chat history.
- **Tools**: VectorTool and JSONalyzeTool for handling specific data sources.

## Architecture

The Cortado AI  Agent is a sophisticated function-calling agent that intelligently queries and routes requests to various tools, functioning as dynamic query engines. This architecture not only supports modular integration of diverse data sources but also enhances the agent's capabilities to perform automated search and retrieval over unstructured, semi-structured, and structured data. 

By leveraging a reasoning loop, the agent can determine which tools to utilize based on the input task, the sequence of tool calls, and the parameters required for each tool. The Cortado AI Agent is designed to extend its functionality with an ad hoc toolset, allowing for specific developments tailored to user needs. It can process responses from various tools, query engines and APIs and store relevant information for future use, making it a powerful knowledge worker within the LlamaIndex framework.

Incorporating conversational memory, the agent retains chat history and context, which enhances its ability to provide personalized and contextually relevant responses. This memory buffer can be configured to be optional, allowing for a more conversational experience when required.

The agent's architecture supports various types of agents, including Function Calling Agents, ReAct agents and Custom Agent developmenets, each designed to optimize the interaction with tools and data sources for RAG applications. With proper tool abstractions, users can define a set of tools that serve as APIs for the agent, facilitating seamless integration and interaction with external services (like Airbnb or other rental platforms).

The Cortado AI Agent exemplifies the capabilities of LLM-powered data agents, enabling efficient management of rental properties and hospitality services through intelligent data handling and response generation.

### Useful Links to LlamaIndex Functions

- **JSONalyze Query Engine**: [JSONalyze Query Engine Documentation](https://docs.llamaindex.ai/en/stable/examples/query_engine/JSONalyze_query_engine/)
- **Vector Stores**: [Using Vector Stores](https://docs.llamaindex.ai/en/stable/examples/vector_stores/)
- **Agent Functions**: [LlamaIndex Agent Functions](https://docs.llamaindex.ai/en/stable/agents/)

### Evaluation Information

For evaluation metrics and detailed documentation on DeepEval, refer to the DeepEval documentation:

- **DeepEval Documentation**: [DeepEval Metrics and Evaluations](https://docs.confident-ai.com/docs/metrics-llm-evals)

### Sources

- [LlamaIndex Documentation](https://docs.llamaindex.ai)
- [LlamaIndex GitHub Repository](https://github.com/run-llama/llama_index)
- [LlamaIndex Overview](https://www.llamaindex.ai)