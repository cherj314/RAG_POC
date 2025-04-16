def build_prompt(context_chunks, user_prompt):
    """
    Builds a single-line prompt for the LLM that includes:
    1. A system prompt defining the assistant's role
    2. Retrieved context from the RAG system
    3. The user's specific request
    
    Returns a well-formatted prompt as a single string.
    """
    # Format the context chunks with clear separators and numerical indicators
    formatted_chunks = []
    for i, (doc, metadata, score) in enumerate(context_chunks, 1):
        # Clean the document text by removing excessive newlines and normalizing whitespace
        clean_doc = ' '.join(doc.split())
        formatted_chunks.append(f"EXAMPLE {i}: {clean_doc}")
    
    # Join the formatted chunks with a separator
    context = " || ".join(formatted_chunks)
    
    # Create the complete prompt with system instructions, context, and user request
    prompt = (
        "SYSTEM: You are an expert software proposal writer for a professional software development company. "
        "Create a comprehensive, persuasive proposal based on the user's request and the provided examples. "
        f"RETRIEVED CONTEXT: {context} "
        f"USER REQUEST: {user_prompt} "
        "TASK: Generate a complete software proposal that includes: 1) Executive Summary, "
        "2) Project Understanding, 3) Technical Approach, 4) Deliverables, 5) Timeline, "
        "6) Team Structure, 7) Pricing, and 8) Next Steps. Make the proposal compelling, "
        "technically sound, and tailored to the client's specific needs."
    )
    
    return prompt