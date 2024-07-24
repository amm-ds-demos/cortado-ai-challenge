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
        Initializes the JSONalyzeTool, which is responsible for processing and analyzing JSON data
        using a specified language model (LLM). This tool leverages the configuration settings to
        set up the environment and prepare the necessary components for querying JSON data.

        Args:
            config (Dict): A dictionary containing configuration settings, including prompts and
                           directory paths for JSON files.
            llm (OpenAI): An instance of the OpenAI language model used for generating responses
                          based on the JSON data.
            json_file (str): The name of the JSON file to be analyzed, which contains structured
                             data relevant to rental properties or prior conversations.
        """
        self.config = config
        self.llm = llm
        self.json_file = json_file
        self.tool = self._initialize_jsonalyze_tool()

    def _initialize_jsonalyze_tool(self) -> QueryEngineTool:
        """
        Initializes the JSONalyze tool by creating a query engine that can process the specified
        JSON data. This method retrieves the appropriate prompt template, loads the JSON data from
        a pickle file with a list of dicts, and sets up the query engine with the necessary parameters.

        Returns:
            QueryEngineTool: An instance of QueryEngineTool that encapsulates the JSON query engine
                             and its associated metadata, enabling structured queries against the
                             loaded JSON data.
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
            query_engine=jsonalyze_engine,
            metadata=ToolMetadata(
                name=self.config["prompts"][f"json_tool_{self.json_file}"]["name"],
                description=self.config["prompts"][f"json_tool_{self.json_file}"][
                    "description"
                ],
            ),
        )
