from typing import Tuple, List
from llama_index.core import Settings
from llama_index.core.memory import (
    SimpleComposableMemory,
    ChatSummaryMemoryBuffer,
    VectorMemory,
)
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import tiktoken

from config import get_config
from tools.vector_tool import VectorTool
from tools.json_tool import JSONalyzeTool


class CortadoAgent:
    def __init__(self):
        """
        Initializes the CortadoAgent with configurations, components,
        tools, memory, and the agent runner.

        This class is responsible for setting up the entire backend system
        for the Cortado AI Agent. It loads configurations, initializes the
        necessary components such as the LLM and embedding model,
        sets up global settings, and prepares initializes tools and memory modules
        required for the agent's operation.

        The agent runner is also created to handle the processing of user queries.
        """
        self.config = get_config()
        self.llm, self.embed_model = self._initialize_components()
        self._setup_settings()
        self.vector_tool = self._initialize_vector_tool()
        self.jsonalyze_tools = self._initialize_jsonalyze_tools()
        self.tools = [self.vector_tool] + self.jsonalyze_tools
        self.composable_memory = self._initialize_memory()
        self.system_prompt = self.config["prompts"]["system_prompt"]
        self.agent_runner = self._create_agent_runner()

    def _setup_settings(self):
        """
        Set up global settings for the LLM, embedding model, and other components
        in the Settings class from llama_index.
        """
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = self.config["chunking"]["chunk_size"]
        Settings.chunk_overlap = self.config["chunking"]["chunk_overlap"]

    def _initialize_components(self) -> Tuple[OpenAI, HuggingFaceEmbedding]:
        """
        Initialize the components including the LLM and embedding model.

        This method creates instances of the OpenAI language model and the HuggingFace
        embedding model based on the configuration settings.

        Returns:
            Tuple[OpenAI, HuggingFaceEmbedding]: The initialized LLM and embedding model.
        """
        llm = OpenAI(
            model=self.config["models"]["llm"],
            temperature=self.config["llm_params"]["temperature"],
        )
        embed_model = HuggingFaceEmbedding(
            model_name=self.config["models"]["embedding"],
            trust_remote_code=self.config["embedding_params"]["trust_remote_code"],
        )
        return llm, embed_model

    def _initialize_vector_tool(self) -> VectorTool:
        """
        Initialize the vector tool with a retriever query engine and metadata.

        This method sets up the VectorTool, which is responsible for handling vectorized
        data sources. It leverages the language model and embedding model to enhance the
        accuracy and relevance of responses generated from the underlying data.

        Returns:
            VectorTool: The initialized vector tool.
        """
        vector_tool = VectorTool(
            config=self.config, llm=self.llm, embed_model=self.embed_model
        )
        return vector_tool

    def _initialize_jsonalyze_tools(self) -> List[JSONalyzeTool]:
        """
        Initialize the JSONalyze tools based on the configuration.

        This method creates instances of the JSONalyzeTool for each JSON file
        specified in the configuration.

        Returns:
            List[JSONalyzeTool]: The list of initialized JSONalyze tools.
        """
        return [
            JSONalyzeTool(config=self.config, llm=self.llm, json_file=json_file)
            for json_file in self.config["json_tool"]["files"]
        ]

    def _initialize_memory(self) -> SimpleComposableMemory:
        """
        Initialize the composable memory with a chat summary memory buffer.

        This method sets up the memory components required for the agent's operation.
        It initializes a chat summary memory buffer to store and summarize chat history,
        and a vector memory to handle vectorized data.

        Returns:
            SimpleComposableMemory: The initialized composable memory.
        """
        chat_history = [
            # Add initial chat history if needed
        ]

        summarizer_llm = OpenAI(
            model_name=self.config["memory"]["summarizer_llm"],
            max_tokens=self.config["memory"]["summarizer_max_tokens"],
        )

        chat_summary_memory_buffer = ChatSummaryMemoryBuffer.from_defaults(
            chat_history=chat_history,
            llm=summarizer_llm,
            token_limit=self.config["memory"]["token_limit"],
        )

        vector_memory = VectorMemory.from_defaults(
            vector_store=None,
            embed_model=self.embed_model,
            retriever_kwargs=self.config["memory"]["retriever_kwargs"],
        )

        return SimpleComposableMemory.from_defaults(
            primary_memory=chat_summary_memory_buffer,
            secondary_memory_sources=[vector_memory],
        )

    def _create_agent_runner(self) -> AgentRunner:
        """
        Create the agent runner with the LLM, tools, system prompt and composable memory.

        This method sets up the agent runner, which is responsible for handling user queries
        and generating responses.

        It configures the FunctionCallingAgentWorker with the LLM model, tools,
        system prompt, and memory components.

        Returns:
            FunctionCallingAgentWorker: The created agent runner.
        """
        agent_worker = FunctionCallingAgentWorker.from_tools(
            llm=self.llm,
            tools=[tool.tool for tool in self.tools],
            system_prompt=self.system_prompt,
            max_function_calls=self.config["agent"]["max_function_calls"],
            verbose=self.config["agent"]["verbose"],
            allow_parallel_tool_calls=False,
        )

        memory_param = (
            self.composable_memory
            if self.config["memory"].get("use_memory", False)
            else None
        )
        return agent_worker.as_agent(memory=memory_param)

    def process_question(self, question: str) -> str:
        """
        Process a question by passing it to the agent runner and returning the response.

        This method takes a user question as input, processes it using the agent runner,
        and returns the generated response.

        It also updates the memory components if memory usage is enabled in the configuration.

        Args:
            question (str): The question to process.

        Returns:
            str: The response from the agent runner.
        """
        try:
            response = self.agent_runner.chat(
                message=question, tool_choice=self.config["agent"]["tool_choice"]
            )
            if self.config["memory"].get("use_memory", False):
                history = self.composable_memory.get()
                self.composable_memory.put(history[-1])
            return response
        except Exception as e:
            return f"Error processing question: {e}"


if __name__ == "__main__":
    agent_backend = CortadoAgent()
    while True:
        question = input("Enter a question for the agent (type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Exiting the agent. Goodbye!")
            break
        response = agent_backend.process_question(question)
        print(response)
