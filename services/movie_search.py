import json
from typing import cast

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from models.messages import MovieCandidate, ParsedPreference

GENRE_NAME_TO_ID = {
    "Action": 28,
    "Adventure": 12,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Family": 10751,
    "Fantasy": 14,
    "History": 36,
    "Horror": 27,
    "Music": 10402,
    "Mystery": 9648,
    "Romance": 10749,
    "Science Fiction": 878,
    "TV Movie": 10770,
    "Thriller": 53,
    "War": 10752,
    "Western": 37,
}


class MovieSearchService:
    def __init__(self, chroma_path: str = "./chroma-data"):
        client = chromadb.PersistentClient(path=chroma_path)
        embedding_fn = cast(
            EmbeddingFunction,
            OpenAIEmbeddingFunction(model_name="text-embedding-3-small"),
        )

        self.collection = client.get_or_create_collection(
            name="movies",
            embedding_function=embedding_fn,
        )

    def search(
        self, preference: ParsedPreference, top_k: int = 10
    ) -> list[MovieCandidate]:

        where = self._build_where(preference)

        results = self.collection.query(
            query_texts=[preference.semantic_query],
            where=where if where else None,
            n_results=top_k,
        )

        ids = (results.get("ids") or [[]])[0]
        if not ids and where:
            results = self.collection.query(
                query_texts=[preference.semantic_query],
                n_results=top_k,
            )
            ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        movies = []

        for movie_id, document, metadata in zip(ids, documents, metadatas):
            doc = json.loads(document) if isinstance(document, str) else document
            metadata = metadata or {}

            raw_genres = metadata.get("genres")
            genres: list[str] = (
                [str(g) for g in raw_genres] if isinstance(raw_genres, list) else []
            )

            raw_score = metadata.get("score")
            rating = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0

            poster_path = doc.get("poster_path")
            if poster_path is not None:
                poster_path = str(poster_path) or None

            movies.append(
                MovieCandidate(
                    movie_id=str(movie_id),
                    title=doc.get("title", ""),
                    genres=genres,
                    rating=rating,
                    overview=doc.get("overview", ""),
                    cast=[],
                    poster_path=poster_path,
                )
            )

        return movies

    def _build_where(self, preference: ParsedPreference) -> dict | None:
        filters = []

        for genre_name in preference.genres:
            if genre_name in GENRE_NAME_TO_ID:
                filters.append({"genres": {"$contains": genre_name}})

        for genre_name in preference.excluded_genres:
            if genre_name in GENRE_NAME_TO_ID:
                filters.append({"genres": {"$not_contains": genre_name}})

        if preference.min_rating is not None:
            filters.append({"score": {"$gte": preference.min_rating}})

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}
