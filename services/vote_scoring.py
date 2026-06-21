"""Helpers for complete per-movie agent vote scores."""

MAX_SCORE_RETRIES = 1
MAX_VETOS_PER_AGENT = 1
VETO_MAX_SCORE = 2.0


def trim_candidate_pool(
    nominations: list, *, max_size: int = 12
) -> list:
    """Keep the most relevant candidates (overlap + rating)."""
    from collections import Counter

    if len(nominations) <= max_size:
        return list({c.movie_id: c for c in nominations}.values())

    counts = Counter(c.movie_id for c in nominations)
    by_id = {c.movie_id: c for c in nominations}
    ranked = sorted(
        by_id.values(),
        key=lambda movie: (-counts[movie.movie_id], -movie.rating),
    )
    return ranked[:max_size]


def build_vote_response_format(pool_ids: list[str]) -> dict:
    """OpenAI json_schema response_format requiring every movie_id in scores."""
    score_properties = {
        movie_id: {"type": "number"}
        for movie_id in pool_ids
    }
    movie_id_schema = {"type": "string", "enum": pool_ids}

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_vote",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "ranked_movie_ids": {
                        "type": "array",
                        "items": movie_id_schema,
                    },
                    "scores": {
                        "type": "object",
                        "properties": score_properties,
                        "required": pool_ids,
                        "additionalProperties": False,
                    },
                    "veto_movie_ids": {
                        "type": "array",
                        "items": movie_id_schema,
                        "maxItems": MAX_VETOS_PER_AGENT,
                    },
                    "reasoning": {"type": "string"},
                    "responding_to": {"type": "string"},
                },
                "required": [
                    "ranked_movie_ids",
                    "scores",
                    "veto_movie_ids",
                    "reasoning",
                    "responding_to",
                ],
                "additionalProperties": False,
            },
        },
    }


def normalize_scores(raw_scores: dict, pool_ids: set[str]) -> dict[str, float]:
    scores = {
        movie_id: max(0.0, min(10.0, float(score)))
        for movie_id, score in raw_scores.items()
        if movie_id in pool_ids
    }
    for movie_id in pool_ids:
        scores.setdefault(movie_id, 0.0)
    return scores


def missing_score_ids(scores: dict[str, float], pool_ids: set[str]) -> set[str]:
    return pool_ids - set(scores.keys())


def normalize_vetos(
    veto_movie_ids: list[str],
    scores: dict[str, float],
    pool_ids: set[str],
    *,
    max_vetos: int = MAX_VETOS_PER_AGENT,
    max_score: float = VETO_MAX_SCORE,
) -> list[str]:
    """Keep at most one veto per agent, only for films scored <= max_score."""
    valid: list[str] = []
    for movie_id in veto_movie_ids:
        if movie_id not in pool_ids:
            continue
        if scores.get(movie_id, 10.0) > max_score:
            continue
        if movie_id in valid:
            continue
        valid.append(movie_id)
        if len(valid) >= max_vetos:
            break
    return valid


def build_vote_prompt_suffix(pool_ids: set[str], candidate_count: int) -> str:
    id_list = ", ".join(sorted(pool_ids))
    return (
        f"You MUST score all {candidate_count} candidate movie_ids from 0 "
        "(would hate it) to 10 (perfect fit).\n"
        f"Your scores object must contain exactly these {candidate_count} keys: "
        f"{id_list}\n"
        "Do not omit any movie_id. Use 0 for films you would not watch.\n"
        f"veto_movie_ids: at most {MAX_VETOS_PER_AGENT} film you absolutely "
        "cannot watch (score 0-2). Use low scores instead of veto for mild "
        "dislikes. Leave veto_movie_ids empty if no hard veto.\n"
        'Return JSON: {"ranked_movie_ids": [str], '
        '"scores": {"movie_id": number}, "veto_movie_ids": [str], '
        '"reasoning": str, "responding_to": str}'
    )


def build_missing_scores_retry_prompt(missing_ids: set[str]) -> str:
    return (
        "Your previous response was incomplete. "
        f"Add scores (0-10) for these missing movie_ids: "
        f"{', '.join(sorted(missing_ids))}.\n"
        "Return the full JSON again with every movie_id scored."
    )
