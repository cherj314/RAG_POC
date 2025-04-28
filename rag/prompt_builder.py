def build_prompt(context_chunks, user_prompt):
    """
    Builds a prompt that combines retrieved context with the user request.
    
    Args:
        context_chunks (list): List of tuples containing (document_text, metadata, similarity_score)
        user_prompt (str): The user's original request
        
    Returns:
        str: A formatted prompt for the LLM
    """
    # Format each context chunk
    formatted_chunks = []
    
    for i, (doc, metadata, score) in enumerate(context_chunks, 1):
        # Clean and normalize the document text (limit to 800 chars per chunk)
        clean_doc = ' '.join(doc.split())[:800]
        
        # Include metadata if available
        source = metadata.get('file_name', '') if metadata and isinstance(metadata, dict) else ''
        meta_str = f" [Source: {source}]" if source else ""
        
        formatted_chunks.append(f"EXAMPLE {i} (RELEVANCE: {score:.2f}){meta_str}: {clean_doc}")
    
    # Join the formatted chunks
    context = " || ".join(formatted_chunks)
    
    # Create the complete prompt
    prompt = (
        "SYSTEM: You are an expert software proposal writer for a professional software development company. "
        "Create a comprehensive, persuasive proposal based on the user's request and the provided examples. "
        f"RETRIEVED CONTEXT: {context} "
        f"USER REQUEST: {user_prompt} "
        "TASK: Generate a complete software proposal that includes: "
        "1) Executive Summary, "
        "2) Project Understanding, "
        "3) Technical Approach, "
        "4) Deliverables, "
        "5) Timeline, "
        "6) Team Structure, "
        "7) Pricing, and "
        "8) Next Steps. "
        "Make the proposal compelling, technically sound, and tailored to the client's specific needs."
    )
    
    return prompt