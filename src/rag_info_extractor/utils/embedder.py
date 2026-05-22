from typing import Any

from langchain_core.embeddings import Embeddings

from rag_info_extractor.utils.apis_connector import call_embedder_service


class HFEmbedder(Embeddings):

    def __init__(self, **encode_kwargs) -> None:
        self.encode_kwargs = encode_kwargs

    def _embed(
        self, texts: list[str], encode_kwargs: dict[str, Any]
    ) -> list[list[float]]:
        """Embed a text using the HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.
            encode_kwargs: Keyword arguments to pass when calling the
                `encode` method for the documents of the SentenceTransformer
                encode method.

        Returns:
            List of embeddings, one for each text.

        """

        texts = [x.replace("\n", " ") for x in texts]
        embeddings = call_embedder_service(texts, **encode_kwargs)

        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.

        """
        return self._embed(texts, self.encode_kwargs)

    def embed_query(self, text: str) -> list[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.

        """
        return self._embed([text], self.encode_kwargs)[0]


if __name__ == "__main__":
    obj = HFEmbedder(normalize_embeddings=True)

    embeddings = obj.embed_documents(
        [
            "BILANCIO E UTILIArt.23.- Gli utili netti, in base a delibera assembleare, sono ripartiti comesegue:- il cinque per cento (5%) sarà destinato alla riserva legale fino alraggiungimento dell'importo pari al venti per cento del capitale sociale;- la rimanenza è ripartita fra i soci in proporzione delle rispettive quote dicapitale, salvo che essi non decidano diversamente.",
            "Art.3.- La durata della società è fissata fino al trentuno dicembreduemilasessanta.Con delibera dell'Assemblea dei soci, potrà essere sciolta anticipatamenteo prorogata.",
            "Art.19. Agli Amministratori spetta, oltre al rimborso delle spesesostenute in ragione del loro ufficio, un compenso eventuale determinato daisoci.",
        ]
    )

    print(embeddings)
