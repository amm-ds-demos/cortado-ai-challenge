import os
import pickle
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import JSONalyzeQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.prompts import PromptTemplate
from typing import Dict


class JSONalyzeTool:
    def __init__(self, config: Dict, llm: OpenAI, json_file: str):
        """
        Initializes the JSONalyzeTool with configurations, LLM, and JSON file.
        Args:
            config (Dict): The configuration dictionary.
            llm (OpenAI): The LLM instance.
            json_file (str): The JSON file name.
        """
        self.config = config
        self.llm = llm
        self.json_file = json_file
        self.tool = self._initialize_jsonalyze_tool()

    def _initialize_jsonalyze_tool(self) -> QueryEngineTool:
        """
        Initialize the JSONalyze tool with a query engine and metadata.
        Returns:
            QueryEngineTool: The initialized JSONalyze tool.
        """
        prompt_template_str = self.config["prompts"][f"json_tool_{self.json_file}"][
            "prompt"
        ]
        prompt_template = PromptTemplate(prompt_template_str)
        pickle_path = os.path.join(
            self.config["directories"]["json_dir"], f"{self.json_file}.pkl"
        )

        with open(pickle_path, "rb") as file:
            json_data = pickle.load(file)

        jsonalyze_engine = JSONalyzeQueryEngine(
            list_of_dict=json_data,
            llm=self.llm,
            verbose=self.config["json_tool"]["verbose"],
            table_name=self.json_file,
            jsonalyze_prompt=prompt_template,
        )
        return QueryEngineTool(
            query_engine=jsonalyze_engine, metadata=self._get_tool_metadata()
        )

    def _get_tool_metadata(self) -> ToolMetadata:
        """
        Get the metadata for the tool.
        Returns:
            ToolMetadata: The metadata for the tool.
        """
        return ToolMetadata(
            name=self.config["prompts"][f"json_tool_{self.json_file}"]["name"],
            description=self.config["prompts"][f"json_tool_{self.json_file}"][
                "description"
            ],
        )
