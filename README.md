# Cortado AI Agent Test

## Introduction

The Cortado AI Agent is designed to assist customers in managing rentals and hospitality services efficiently. By leveraging advanced AI capabilities, the agent processes and responds to user queries, analyzes documents, and manages data. The integration of large language models (LLMs) and sophisticated data processing tools ensures high relevance and accuracy in responses, enhancing user experience and operational efficiency.

### Technical Background

The core of the Cortado AI Backend Agent is built upon the integration of LlamaIndex, a versatile framework designed to augment LLMs with custom data sources, agents and tools. This enables the system to ingest, structure, and access domain-specific data efficiently. The system uses OpenAI's LLMs for natural language understanding and HuggingFace embeddings for semantic search and retrieval.

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
  - `get_tool_metadata() -> ToolMetadata`: Gets the metadata for the tool.

### `config.py`

This module provides a comprehensive configuration dictionary that is essential for the entire system's operation. It allows for highly parametrized modules and components to be easily configured through a centralized config file. Key elements include parameters for various models (LLMs), prompts for guiding the agent's responses, and metadata for tools used in processing. This flexibility ensures that the system can be tuned to specific requirements and easily adjusted without modifying the underlying codebase.

#### Methods

- `get_config() -> dict`: Returns the configuration dictionary.

### `evaluation.py`

The evaluation script is designed to test the agent's functionalities and correctness. It uses the DeepEval library to assess the performance of the agent's responses.

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

- **OpenAI LLMs**: The core of the agent's functionality is powered by OpenAI's advanced language models which enables it to understand and generate human-like text. The models can interact with
different RAG tools for agentic capabilities in conversational tasks, making it adept at handling a wide range of inquiries and retrieve the correct documentation.

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

The Cortado AI Backend Agent is a function-calling agent designed to query and route to different tools as query engines. The architecture supports modular integration of various data sources and query engines to provide comprehensive and contextually relevant responses.

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