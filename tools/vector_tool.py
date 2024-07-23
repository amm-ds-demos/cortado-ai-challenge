import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from typing import Dict


class VectorTool:
    def __init__(self, config: Dict, llm: OpenAI, embed_model: HuggingFaceEmbedding):
        """
        Initializes the VectorTool with configurations, LLM, and embedding model.
        Args:
            config (Dict): The configuration dictionary.
            llm (OpenAI): The LLM instance.
            embed_model (HuggingFaceEmbedding): The embedding model instance.
        """
        self.config = config
        self.llm = llm
        self.embed_model = embed_model
        self._ensure_index_construction()
        self.tool = self._initialize_vector_tool()

    def _ensure_index_construction(self):
        """
        Ensure the vector index is constructed and stored.
        """
        if not os.path.exists(self.config["directories"]["index_name"]):
            os.makedirs(self.config["directories"]["index_name"])
            transformations = Settings.transformations

            documents = SimpleDirectoryReader(
                self.config["directories"]["pdf_dir"]
            ).load_data()
            VectorStoreIndex.from_documents(
                documents, transformations=transformations
            ).storage_context.persist(self.config["directories"]["index_name"])

    def _load_memory_index(self) -> VectorStoreIndex:
        """
        Load the memory index from storage.
        Returns:
            VectorStoreIndex: The loaded memory index.
        """
        storage_context = StorageContext.from_defaults(
            persist_dir=self.config["directories"]["index_name"]
        )
        return load_index_from_storage(storage_context)

    def _initialize_vector_tool(self) -> QueryEngineTool:
        """
        Initialize the vector tool with a retriever query engine and metadata.
        Returns:
            QueryEngineTool: The initialized vector tool.
        """
        vector_store_info = VectorStoreInfo(
            content_info=self.config["vector_tool"]["content_info"],
            metadata_info=[
                MetadataInfo(
                    name=info["name"],
                    type=info["type"],
                    description=info["description"],
                )
                for info in self.config["vector_tool"]["metadata_info"]
            ],
        )
        vector_index = self._load_memory_index()
        vector_auto_retriever = VectorIndexAutoRetriever(
            vector_index, vector_store_info
        )
        rerank = SentenceTransformerRerank(
            model=self.config["models"]["embedding"],
            top_n=self.config["vector_tool"]["rerank_top_n"],
        )
        retriever_query_engine = RetrieverQueryEngine.from_args(
            vector_auto_retriever, llm=self.llm, node_postprocessors=[rerank]
        )
        return QueryEngineTool(
            query_engine=retriever_query_engine,
            metadata=ToolMetadata(
                name=self.config["prompts"]["vector_tool"]["name"],
                description=self.config["prompts"]["vector_tool"]["description"],
            ),
        )
