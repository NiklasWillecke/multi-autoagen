from models.messages import MovieCandidate
from services.vote_scoring import (
    build_vote_response_format,
    missing_score_ids,
    normalize_scores,
    normalize_vetos,
    trim_candidate_pool,
)


def _movie(movie_id: str, rating: float) -> MovieCandidate:
    return MovieCandidate(
        movie_id=movie_id,
        title=movie_id,
        genres=["Drama"],
        rating=rating,
        overview="",
    )


def test_normalize_scores_backfills_missing():
    pool_ids = {"1", "2", "3"}
    scores = normalize_scores({"1": 8, "2": 5}, pool_ids)
    assert scores == {"1": 8.0, "2": 5.0, "3": 0.0}


def test_trim_candidate_pool_prefers_overlap_and_rating():
    nominations = [
        _movie("a", 6.0),
        _movie("b", 7.0),
        _movie("a", 6.0),
        _movie("c", 9.0),
    ]
    trimmed = trim_candidate_pool(nominations, max_size=2)
    ids = {movie.movie_id for movie in trimmed}
    assert ids == {"a", "c"}


def test_build_vote_response_format_requires_all_ids():
    schema = build_vote_response_format(["10", "20"])
    required = schema["json_schema"]["schema"]["properties"]["scores"]["required"]
    assert required == ["10", "20"]


def test_missing_score_ids():
    assert missing_score_ids({"1": 1.0}, {"1", "2"}) == {"2"}


def test_normalize_vetos_limits_and_requires_low_score():
    scores = {"1": 0.0, "2": 8.0, "3": 1.0}
    pool_ids = {"1", "2", "3"}
    assert normalize_vetos(["1", "2", "3"], scores, pool_ids) == ["1"]
    assert normalize_vetos(["2", "3"], scores, pool_ids) == ["3"]
