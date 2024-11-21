def generate_response(user_messages, retrieve_from_vector_store_function, generate_text_function) -> string:

    # Filter messages to get only those from the user
    user_messages = [msg['content'] for msg in user_messages if msg['role'] == 'user']
    
    # Check if there are any user messages
    if user_messages:
        last_user_message = user_messages[-1]
    else:
        last_user_message = None
        
    # Step 1: Obter entrada do usuário
    user_input = last_user_message

    context = retrieve_from_vector_store_function(user_input)

    
    prompt = f"""
    Você um chatBot prestativo que ajuda estudantes a entender normas da faculdade.
    Responda as perguntas baseadas no contexto:{context}.
    
    Minha pergunta é: {user_input}
    """

    text = generate_text_function(prompt)

    return text