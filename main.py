from Functions.documents import load_documents
from Functions.text import clear_text, remove_stopwords, split_text_into_chunks

document_paths = ["./Documents/a (1).pdf", "./Documents/FAQ (1).pdf"]

textos = load_documents(document_paths)

textos_limpos = [clear_text(texto) for texto in textos]

textos_sem_stopwords = [remove_stopwords(texto) for texto in textos_limpos]

texto_unido = " ".join(textos_sem_stopwords)

for x in split_text_into_chunks(texto_unido):
    print(x)