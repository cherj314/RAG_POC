def build_prompt(context_chunks, user_prompt):
    context = "\n\n---\n\n".join(chunk[0] for chunk in context_chunks)
    return f"""You are an expert software proposal writer.
Use the following past examples to help write a new proposal.

Context:
{context}

User Request:
{user_prompt}
"""
