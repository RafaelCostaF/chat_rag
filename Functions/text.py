import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


def clear_text(text):
    """
    Cleans up the given text by removing unwanted Unicode characters such as
    zero-width spaces and other whitespace artifacts.

    Parameters:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    # Remove zero-width spaces and other similar characters
    cleaned_text = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', text)
    
    # Optionally remove excessive whitespace, line breaks, etc.
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text



def remove_stopwords(text, language='portuguese'):
    """
    Removes stopwords from the given text.

    Parameters:
        text (str): The input text to be processed.
        language (str): The language of the stopwords (default is 'portuguese').

    Returns:
        str: The text with stopwords removed.
    """
    # Tokenize and clean text (splitting words, removing punctuation, etc.)
    words = re.findall(r'\w+', text)
    
    # Load the stopwords for the specified language
    stop_words = set(stopwords.words(language))
    
    # Remove stopwords
    filtered_text = ' '.join(word for word in words if word.lower() not in stop_words)
    
    return filtered_text

def split_text_into_chunks(text, chunk_size=500):
    """
    Splits the input text into chunks of approximately equal size.

    Parameters:
        text (str): The input text to be split.
        chunk_size (int): The number of words per chunk (default is 500).

    Returns:
        list of str: A list containing the text chunks.
    """
    # Tokenize the text into words
    words = text.split()
    
    # Split words into chunks of specified size
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    
    return chunks