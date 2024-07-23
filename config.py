def get_config() -> dict:
    """
    Get the configuration dictionary.
    Returns:
        dict: The configuration dictionary.
    """
    cfg = {}

    # Models
    cfg["models"] = {"llm": "gpt-3.5-turbo", "embedding": "BAAI/bge-large-en-v1.5"}

    cfg["llm_params"] = {"temperature": 0.6}
    cfg["embedding_params"] = {"trust_remote_code": False}

    # Directories
    cfg["directories"] = {
        "pdf_dir": "data/pdf",
        "json_dir": "data/json",
        "index_name": "collections/storage_gpt3.5_0.6_96_4_10_bge_run2",
    }

    # Prompts
    cfg["prompts"] = {}
    cfg["prompts"][
        "system_prompt"
    ] = """
        You are an assistant for Cortado AI and one single rental property, helping customers manage rentals and hospitality services efficiently. 
        Provide informative, kind, and comprehensive responses to all inquiries regarding rental properties.

        Guidelines:
        - Refer to ALL tools for comprehensive information.
        - When calculating prices, consider base price, number of nights, and other applicable charges ACCORDING TO EACH CASE AND INQUIRY, do not make assumptions. Double-check math.
        - Reconcile conflicting information from different sources and provide the most accurate answer based on available data.
        - In your final response you should NEVER mention "The document provides/does not mention", "check the documentation/contact the host" or similar responses. 
        - If information is not available, elaborate answer with the retrieved information only.
        - AGGREGATE AND ANALYZE ALL FUNCTION OUTPUT FOR ELABORATING FINAL ANSWER. 
        - PARAPHRASE AND COPY FUNCTION OUTPUT FOR BETTER RETRIEVAL. IF SPECIFIC NAMES OR DETAILS APPEAR, ADD THEM TO FINAL ANSWER.

        Tool Descriptions:
            vector_tool:
                Provides detailed welcome documentation from PDF document, including:
                - parking
                - public transport
                - kitchen equipment: coffee, garbage, dishwasher
                - laundry
                - restaurants and bars
                - groceries
                - nearby attractions
                - check-out procedures

            listing_object_tool:
                Retrieve structured information about rental properties from JSON files, including:
                - rental information: name, description, address, capacity, smart-lock code, wifi and internet details
                - pricing details: nightly rates and applicable fees
                - house rules: cleanliness, smoking policies, pet policies, party and noise restrictions
                - cancellation policies: policies for cancellation
                - contact details: host name, phone, email
                - check-in and check-out times

            prior_conversations_tool:
                Access records of previous customer interactions from JSON files, including:
                - conversation histories
                - message body 
                - communication types (email, SMS)
                - timestamps of interactions
                - customer details: names, specific requirements, dates of stay
        
        The way you use the tools is by specifying a json blob.
        Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

        The only values that should be in the "action" field are: vector_tool, listing_object_tool, prior_conversations_tool

        The $JSON_BLOB should only contain a SINGLE action. Here is an example of a valid $JSON_BLOB:

        {
        "action": $TOOL_NAME,
        "action_input": $INPUT
        }
        
        $INPUT CONTAINS QUERY TOOL REQUEST. $INPUT SHOULD ALWAYS BE RELATED TO THE USER INPUT.
        """

    cfg["prompts"]["vector_tool"] = {
        "name": "vector_tool",
        "description": "Comprehensive guide for welcoming guests to the rental, covering property details and local attractions to ensure a pleasant stay.",
    }

    cfg["prompts"]["json_tool_prior_conversations"] = {
        "name": "prior_conversations_tool",
        "description": "Provides insights from prior conversations with customers. CHECK TABLE SCHEMA AND TOOL PROMPT",
        "prompt": (
            "You are given a table named: '{table_name}' with schema, "
            "generate SQLite SQL query to answer the given question.\n"
            "Table schema:\n"
            "{table_schema}\n"
            "Here is the structure and description of the fields in the DB:\n"
            "id: (int) Unique identifier for the communication entry. Ex: 160610667\n"
            "listingMapId: (int) Identifier linking the communication to a specific property listing. Ex: 243759\n"
            "reservationId: (int) Identifier for the reservation related to the communication. Ex: 30540343\n"
            "conversationId: (int) Identifier for the conversation thread. Ex: 20149952\n"
            "body: (str) The content of the communication message. Contains direct messages with customer. Ex: 'Hey there!'\n"
            "imagesUrls: (str or None) URLs to images included in the communication. Ex: None\n"
            "isIncoming: (int) (1) or outgoing (0). Ex: 1\n"
            "listingTimeZoneName: (str) The time zone of the property listing. Ex: 'America/New_York'\n"
            "status: (str) The status of the communication (e.g., sent, received). Ex: 'sent'\n"
            "sentChannelDate: (str) The date and time when the communication was sent. Ex: '2024-07-16 21:19:02'\n"
            "communicationType: (str) The type of communication (e.g., email, SMS). Ex: 'email'\n"
            "INSTRUCTION #1: NEVER USE AND STATEMENTS, USE OR FOR BROAD OUTPUTS. MAKE BIG SELECT QUERIES FOR BIGGER OUTPUTS.\n"
            "INSTRUCTION #2: LIKE STATEMENTS SHOULD ONLY BE A SINGLE WORD-> WHERE BODY LIKE '%WORD', NEVER WITH UNDERSCORES\n"
            "INSTRUCTION #3: body FIELD IS THE TEXT MESSAGES WITH CUSTOMER, ANALYZE ALWAYS THE CONTENT.\n"
            "INSTRUCTION #4: QUERY AS THE EXAMPLE DATES AND WEEKLY FOR BETTER CONVERSATION CONTEXT\n"
            "Question: {question}\n\n"
            "SQLQuery: "
        ),
    }

    cfg["prompts"]["json_tool_listing_object"] = {
        "name": "listing_object_tool",
        "description": "Provides detailed information about the property listings. CHECK TABLE SCHEMA AND TOOL PROMPT",
        "prompt": (
            "You are given a table named: '{table_name}' with schema, "
            "generate SQLite SQL query to answer the given question.\n"
            "Table schema:\n"
            "{table_schema}\n"
            "Here is the structure and description of the fields in the DB:\n"
            "id: (int) Unique identifier for the property listing. \n"
            "propertyTypeId: (int) Identifier for the type of property. \n"
            "name: (str) The name of the property. \n"
            "externalListingName: (str) The external name used for listing the property. \n"
            "internalListingName: (str) The internal name used within the system for the property. \n"
            "description: (str) A detailed description of the property. \n"
            "thumbnailUrl: (str) URL to the thumbnail image of the property. \n"
            "houseRules: (str) Rules that guests must follow when staying at the property.\n"
            "doorSecurityCode: (str) Security code for accessing the property. \n"
            "country: (str) The country where the property is located. \n"
            "countryCode: (str) The country code where the property is located. \n"
            "state: (str) The state where the property is located. \n"
            "city: (str) The city where the property is located. \n"
            "street: (str) The street address of the property. \n"
            "address: (str) The full address of the property. \n"
            "publicAddress: (str) The public address of the property. \n"
            "zipcode: (str) The postal code of the property. \n"
            "nightlyPrice: (int) The price per night to stay at the property.\n"
            "starRating: (float or None) The star rating of the property. \n"
            "weeklyDiscount: (float) The discount applied for weekly stays. Ex: 0.85\n"
            "monthlyDiscount: (float) The discount applied for monthly stays. Ex: 0.75\n"
            "personCapacity: (int) The maximum number of people the property can accommodate. Ex: 8\n"
            "checkInTimeStart: (int) The earliest check-in time. \n"
            "checkInTimeEnd: (int) The latest check-in time. \n"
            "checkOutTime: (int) The check-out time. \n"
            "cancellationPolicy: (str) The policy for canceling a reservation.\n"
            "squareMeters: (int) The size of the property in square meters. \n"
            "roomType: (str) The type of room available.\n"
            "bathroomType: (str) The type of bathroom available.\n"
            "bedroomsNumber: (int) The number of bedrooms. \n"
            "bedsNumber: (int) The number of beds. \n"
            "bathroomsNumber: (int) The number of bathrooms. \n"
            "minNights: (int) The minimum number of nights required for a stay. \n"
            "maxNights: (int) The maximum number of nights allowed for a stay. \n"
            "guestsIncluded: (int) The number of guests included in the price.\n"
            "cleaningFee: (int) The fee for cleaning the property. \n"
            "priceForExtraPerson: (int) The price for each additional person beyond the included number. \n"
            "petFee: (int) The fee for bringing a pet. ONE-TIME-FEE. \n"
            "contactName: (str) The first name of the contact person.\n"
            "contactSurName: (str) The surname of the contact person.'\n"
            "contactPhone1: (str) The phone number of the contact person. Ex: '+1 (555) 123-4567'\n"
            "contactLanguage: (str) The language spoken by the contact person. Ex: 'English'\n"
            "contactEmail: (str) The email address of the contact person. Ex: 'john.doe@example.com'\n"
            "contactAddress: (str) The address of the contact person. Ex: '456 Fictional Rd'\n"
            "language: (str) The language of the listing. Ex: 'en'\n"
            "currencyCode: (str) The currency code for pricing. Ex: 'USD'\n"
            "timeZoneName: (str) The time zone of the property. \n"
            "wifiUsername: (str) The WiFi username for the property. \n"
            "wifiPassword: (str) The WiFi password for the property. \n"
            "cleaningInstruction: (str) Instructions for cleaning the property. \n"
            "latestActivityOn: (str) The date and time of the latest activity related to the listing. Ex: '2023-12-01 12:00:00'\n"
            "Question: {question}\n\n"
            "SQLQuery: "
        ),
    }

    # Chunking settings
    cfg["chunking"] = {"chunk_size": 128, "chunk_overlap": 4}

    # Vector tool settings
    cfg["vector_tool"] = {
        "rerank_top_n": 7,
        "content_info": "Detailed information about the rental property, including its features, amenities, and surrounding area, extracted from PDF documents.",
        "metadata_info": [
            {
                "name": "property_context",
                "type": "pdf",
                "description": "Detailed information about the rental property, including its features, amenities, and surrounding area, extracted from PDF documents.",
            }
        ],
    }

    # JSON tool settings
    cfg["json_tool"] = {
        "verbose": True,
        "files": ["prior_conversations", "listing_object"],
    }

    cfg["memory"] = {
        "use_memory": False,
        "tokenizer_llm": "gpt-3.5-turbo",
        "summarizer_llm": "gpt-3.5-turbo",
        "retriever_kwargs": {"similarity_top_k": 3},
        "summarizer_max_tokens": 32,
        "token_limit": 32,
    }
    # Agent settings
    cfg["agent"] = {
        "tool_choice": [
            "vector_tool",
            "prior_conversations_tool",
            "listing_object_tool",
        ],
        "max_function_calls": 7,
        "verbose": True,
    }

    # Add these parameters to the existing config dictionary
    cfg["evaluation"] = {
        "questions_file": "data/questions.json",
        "evaluations_output_file": "evaluations/evaluation_gpt-3.5-0.6_96_4_10_bge_run2.json",
        "processed_agent_responses_file": "processed_responses/processed_agent_responses_gpt-3.5-0.6_96_4_10_bge_run2.json",
        "relevancy_threshold": 0,
        "correctness_threshold": 0,
        "evaluation_model": "gpt-3.5-turbo",
        "geval_metrics": [
            {
                "name": "Correctness",
                "criteria": "Correctness - determine if the actual output is correct according to the expected output.",
                "evaluation_params": ["ACTUAL_OUTPUT", "EXPECTED_OUTPUT"],
            },
            {
                "name": "Truthfulness",
                "criteria": "Truthfulness - determine if the actual output is true and informative according to the expected output.",
                "evaluation_params": ["ACTUAL_OUTPUT", "EXPECTED_OUTPUT"],
            },
        ],
    }

    return cfg
