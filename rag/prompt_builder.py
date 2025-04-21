def build_prompt(context_chunks, user_prompt):
    """
    Builds a prompt for the LLM that combines retrieved context with the user request.
    
    Args:
        context_chunks (list): List of tuples containing (document_text, metadata, similarity_score)
        user_prompt (str): The user's original request
        
    Returns:
        str: A formatted prompt ready to be sent to the LLM
    """
    # Format each context chunk with a clear identifier and relevance score
    formatted_chunks = []
    
    for i, (doc, metadata, score) in enumerate(context_chunks, 1):
        # Clean and normalize the document text (limit length to avoid giant prompts)
        clean_doc = ' '.join(doc.split())[:800]  # Limit to 800 chars per chunk
        
        # Include metadata if available
        meta_str = ""
        if metadata and isinstance(metadata, dict):
            if 'file_name' in metadata:
                meta_str = f" [Source: {metadata['file_name']}]"
        
        formatted_chunks.append(f"EXAMPLE {i} (RELEVANCE: {score:.2f}){meta_str}: {clean_doc}")
    
    # Join the formatted chunks with a separator (using single-line format)
    context = " || ".join(formatted_chunks)
    
    # Create the complete prompt with clear sections (maintain single-line format)
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