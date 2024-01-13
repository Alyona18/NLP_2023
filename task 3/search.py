import gzip
import argparse
from sentence_transformers import SentenceTransformer
import faiss

class FaissVectorStorage:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = None

    def load_index(self, index_path):
        self.index = faiss.read_index(index_path)

    def search(self, query_embedding, k=5):
        _, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return indices[0]
   
def load_texts_from_gzip(file_path):
    texts = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
        for line in gz_file:
            _, _, text = line.strip().split('\t')  
            texts.append(text)  
    return texts
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск текстовых документов с использованием FAISS.")
    parser.add_argument("--model_name", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--query", required=True, help="Текст запроса")
    parser.add_argument("--index_path", required=True, help="Путь к сохраненному индексу")
    parser.add_argument("--top_k", type=int, default=5, help="Количество возвращаемых результатов (по умолчанию 5)")
    parser.add_argument("--data_path", required=True, help="Путь к файлу с текстовыми документами")

    args = parser.parse_args()

    model = SentenceTransformer(args.model_name)

    query_embedding = model.encode(args.query)

    vector_storage = FaissVectorStorage(dimension=model.get_sentence_embedding_dimension())
    vector_storage.load_index(args.index_path)

    search_results = vector_storage.search(query_embedding, k=args.top_k)
    
    documents = load_texts_from_gzip(args.data_path)

    print(f"Результаты поиска для запроса '{args.query}':")
    documents = load_texts_from_gzip(args.data_path)  # Загрузка текстовых документов

    for result_index in search_results:
        if result_index < len(documents):
            print(result_index)
            print(documents[result_index])
            print()
        else:
            print(f"Index {result_index} is out of range for the loaded documents.")
