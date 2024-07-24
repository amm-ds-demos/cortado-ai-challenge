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
from llama_index.core.node_parser import SentenceSplitter

from typing import Dict


class VectorTool:
    def __init__(self, config: Dict, llm: OpenAI, embed_model: HuggingFaceEmbedding):
        """
        Initializes the VectorTool, a component designed to facilitate the retrieval and processing of
        information from vectorized data sources.

        Args:
            config (Dict): A comprehensive configuration dictionary that contains settings for
                           directories, model parameters, and tool-specific options.
            llm (OpenAI): An instance of the OpenAI language model, which is utilized for generating
                          natural language responses based on the retrieved data.
            embed_model (HuggingFaceEmbedding): An instance of the HuggingFace embedding model,
                                                 responsible for transforming textual data into
                                                 vector representations for efficient similarity
                                                 searches.

        The VectorTool is responsible for ensuring that the vector index is constructed and
        maintained, allowing for efficient querying of data.
        """
        self.config = config
        self.llm = llm
        self.embed_model = embed_model
        self._ensure_index_construction()
        self.tool = self._initialize_vector_tool()

    def _ensure_index_construction(self):
        """
        Ensures that the vector index is constructed and stored in the specified directory. If the
        index does not exist, it creates the necessary directory structure and loads documents from
        the specified PDF directory. The documents are then processed and stored in a vector index
        for future retrieval.

        This method utilizes the SimpleDirectoryReader to load data from PDF files and the
        VectorStoreIndex to create a persistent storage context for the vectorized documents.
        """
        if not os.path.exists(self.config["directories"]["index_name"]):
            os.makedirs(self.config["directories"]["index_name"])
            # Pipeline of different transformations can be added here
            transformations = [
                SentenceSplitter(
                    chunk_size=self.config["chunking"]["chunk_size"],
                    chunk_overlap=self.config["chunking"]["chunk_overlap"],
                ),
                self.embed_model,
            ]

            documents = SimpleDirectoryReader(
                self.config["directories"]["pdf_dir"]
            ).load_data()
            VectorStoreIndex.from_documents(
                documents, transformations=transformations
            ).storage_context.persist(self.config["directories"]["index_name"])

    def _load_memory_index(self) -> VectorStoreIndex:
        """
        Loads the memory index from the specified storage directory. This index contains the
        vectorized representations of documents that have been previously stored, allowing for
        efficient retrieval during query processing.

        Returns:
            VectorStoreIndex: The loaded memory index, which can be used for querying and
                              retrieving relevant information based on user input.
        """
        storage_context = StorageContext.from_defaults(
            persist_dir=self.config["directories"]["index_name"]
        )
        return load_index_from_storage(storage_context)

    def _initialize_vector_tool(self) -> QueryEngineTool:
        """
        Initializes the vector tool by setting up a retriever query engine and associated metadata.
        This process involves creating a VectorStoreInfo object that encapsulates information about
        the content and metadata of the vectorized documents. It also configures a retriever query
        engine that utilizes the vector index and applies a reranking model to enhance the quality
        of the retrieved results.

        Returns:
            QueryEngineTool: The initialized vector tool, which includes the query engine and
                             metadata necessary for processing user queries and generating
                             responses.
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
