import streamlit as st
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Menginisialisasi Sastrawi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Mengunduh kamus stopword Bahasa Indonesia dari NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# Membuat fungsi untuk membersihkan teks
def clean_text(text):
    # Mengubah ke lowercase
    text = text.lower()

    # Menghapus karakter non-alfanumerik
    text = re.sub(r'\W+', ' ', text)

    # Menghapus angka
    text = re.sub(r'\d+', '', text)

    return text

def tokenizing(text):
    tokens = word_tokenize(str(text))
    return tokens

def stopword_removal(tokens):
    stopword_tokens = [token for token in tokens if token not in stop_words]
    return stopword_tokens

def stem_tokens(stopword_tokens):
    stemmed_tokens = [stemmer.stem(token) for token in stopword_tokens]
    return stemmed_tokens


def load_sentiment_dictionary():
    sentiment_dict = {
        'positif': set(),
        'negatif': set()
    }

    # Membaca file kamus positif
    with open('positive.txt', 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            sentiment_dict['positif'].add(word)

    # Membaca file kamus negatif
    with open('negative.txt', 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            sentiment_dict['negatif'].add(word)

    return sentiment_dict


def analyze_sentiment(text, sentiment_dict):
    tokens = tokenizing(text)
    stopword = stopword_removal(tokens)
    stemmed = stem_tokens(stopword)
    
    positive_count = sum(token in sentiment_dict['positif'] for token in stemmed)
    negative_count = sum(token in sentiment_dict['negatif'] for token in stemmed)

    if positive_count > negative_count:
        sentiment = 'Positif'
    elif positive_count < negative_count:
        sentiment = 'Negatif'
    else:
        sentiment = 'Netral'
    
    return sentiment

# Mengatur tampilan Streamlit

sentiment_dict = load_sentiment_dictionary()

def main():
    st.title("Analisis Sentimen Bahasa Indonesia")
    st.write("Masukkan teks di bawah untuk menganalisis sentimen:")
    
    text = st.text_area("Teks")
    
    if st.button("Analisis"):    
        clean = clean_text(text)
        st.text_area("Clean Text:", clean)

        tokens = tokenizing(clean)
        st.text_area("Tokenizing:", tokens)
        
        filtered_tokens = stopword_removal(tokens)
        st.text_area("Stopword Removal:", filtered_tokens)
        
        stemmed_tokens = stem_tokens(filtered_tokens)
        st.text_area("Stemming:", stemmed_tokens)

        sentiment = analyze_sentiment(text, sentiment_dict)
        st.write("Sentimen:", sentiment)


# Menjalankan aplikasi
if __name__ == "__main__":
    main()
