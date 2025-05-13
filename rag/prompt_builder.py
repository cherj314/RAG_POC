# Builds a prompt that combines retrieved context with the user request.
def build_prompt(context_chunks, user_prompt):
    # Format each context chunk with more detailed source information
    formatted_chunks = []
    
    for i, (doc, metadata, score) in enumerate(context_chunks, 1):
        # Clean and normalize the document text
        clean_doc = doc.strip()
        
        # Include detailed metadata if available
        source = metadata.get('file_name', '') if metadata and isinstance(metadata, dict) else ''
        page_info = f", Page {metadata.get('page_num', '')}" if metadata and metadata.get('page_num') else ''
        section_info = f", Section: {metadata.get('current_section', '')}" if metadata and metadata.get('current_section') else ''
        
        # Combine metadata into a single string
        meta_str = f"[Source: {source}{page_info}{section_info}]" if source else ""
        
        formatted_chunks.append(f"PASSAGE {i} (RELEVANCE: {score:.2f}) {meta_str}:\n{clean_doc}")
    
    # Join the formatted chunks with clear separation
    context = "\n\n---\n\n".join(formatted_chunks)
    
    # Create the optimized Harry Potter prompt
    prompt = (
        f"RETRIEVED PASSAGES:\n\n{context}\n\n"
        f"QUESTION: {user_prompt}\n\n"
        "ANSWER:"
    )
    
    return prompt