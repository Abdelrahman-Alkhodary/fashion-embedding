from sentence_transformers import SentenceTransformer


class SentenceTransformerModel:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences):
        return self.model.encode(sentences)