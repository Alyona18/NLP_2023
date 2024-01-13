#python index.py --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 
#--data_path "D:\Nlp'23\nlp-2023\data\news.txt.gz" 
#--index_path "D:\Nlp'23\nlp-2023\tasks\task 3\data\multilingual-MiniLM-L12-v2_index.index"
#or  
#--model_name sentence-transformers/distiluse-base-multilingual-cased-v2
#--index_path  "D:\Nlp'23\nlp-2023\tasks\task 3\data\distiluse-base-multilingual-cased-v2_index.index"

import gzip
from sentence_transformers import SentenceTransformer
import faiss
import argparse
from gensim.utils import simple_preprocess

class FaissVectorStorage:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)

    def initialize(self):
        pass

    def add_vectors(self, embeddings, documents):
        self.index.add(embeddings)

    def save_index(self, index_path):
        faiss.write_index(self.index, index_path)

def index_documents(model, documents, index_storage):
    embeddings = model.encode(documents)
    index_storage.add_vectors(embeddings, documents)

def load_texts_from_gzip(file_path):
    texts = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
        for line in gz_file:
            _, _, text = line.strip().split('\t')
            texts.append(simple_preprocess(text, deacc=True))  
    return texts

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Индексация текстовых документов с использованием FAISS.")
    parser.add_argument("--model_name", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Имя предварительно обученной модели SentenceTransformer")
    parser.add_argument("--data_path", required=True, help="Путь к файлу с текстовыми документами")
    parser.add_argument("--index_path", required=True, help="Путь для сохранения индекса")

    args = parser.parse_args()

    model = SentenceTransformer(args.model_name)
 
    texts = load_texts_from_gzip(args.data_path)

    vector_storage = FaissVectorStorage(dimension=model.get_sentence_embedding_dimension())
    index_documents(model, texts, vector_storage)
    vector_storage.save_index(args.index_path)

    print(f"Индекс сохранен по пути: {args.index_path}")
