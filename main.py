"""compute sililarity between two documents using cosine similarity"""
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(corpus):
    """compute cosine similarity"""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(corpus)
    return cosine_similarity(tfidf[0, :], tfidf[1, :])

def read_file(filename):
    """read file and return content"""
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def main():
    """main function"""
    if len(sys.argv) != 3:
        print('Usage: python compute_similarity.py <document1> <document2>')
        sys.exit(1)

    document1 = read_file(sys.argv[1])
    document2 = read_file(sys.argv[2])
    similarity = compute_cosine_similarity([document1, document2])
    print (similarity)

if __name__ == '__main__':
    main()
