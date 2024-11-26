from Functions.text import clear_text

def generate_response(user_messages, retrieve_from_vector_store_function, generate_text_function):

    # Filter messages to get only those from the user
    user_messages = [msg['content'] for msg in user_messages if msg['role'] == 'user']
    
    # Check if there are any user messages
    if user_messages:
        last_user_message = user_messages[-1]
    else:
        last_user_message = None
        
    # Step 1: Obter entrada do usuário
    user_input = last_user_message

    response_vector_store = retrieve_from_vector_store_function(user_input)
    context = "\n".join([x.text for x in response_vector_store.source_nodes])
    context = clear_text(context)


    prompt = f"""
    COMANDOS GERAIS:
    Você um chatBot prestativo que ajuda estudantes a entender as normas da faculdade e tira dúvidas acadêmicas.
    
    INSTRUÇÃO:
    Responda à(s) pergunta(s) abaixo baseado no contexto, que é proveniente de documentos da universidade
    
    CONTEXTO:
    {context}
    
    PERGUNTA DO USUÁRIO: {user_input}
    
    SUA RESPOSTA:
    """

    text = generate_text_function(prompt)

    return text