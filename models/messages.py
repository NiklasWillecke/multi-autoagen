# messages.py
# messages.py
from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input / Request
# ---------------------------------------------------------------------------
class UserPreferenceInput(BaseModel):
    user_id: str
    user_name: str
    raw_text: str


class StartRecommendation(BaseModel):
    users: List[UserPreferenceInput]
    max_rounds: int = Field(default=3, ge=1, le=5)
    enable_discussion: bool = False
    discussion_turns: int = Field(default=3, ge=1, le=6)


# ---------------------------------------------------------------------------
# Parsed preferences
# ---------------------------------------------------------------------------
class ParsedPreference(BaseModel):
    user_id: str
    user_name: str

    genres: List[str] = Field(default_factory=list)
    excluded_genres: List[str] = Field(default_factory=list)
    min_rating: Optional[float] = None

    # Hauptfeld für Vector Search
    semantic_query: str = ""


# ---------------------------------------------------------------------------
# Movie search
# ---------------------------------------------------------------------------
class MovieCandidate(BaseModel):
    movie_id: str
    title: str
    genres: List[str]
    rating: float
    overview: str
    cast: List[str] = Field(default_factory=list)
    poster_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Discussion protocol (NEU)
# ---------------------------------------------------------------------------
class NominationRequest(BaseModel):
    """Moderator -> User-Agent: schlag deine Top-Filme vor."""

    pass


class Nomination(BaseModel):
    """User-Agent -> Moderator: meine Kandidaten."""

    user_id: str
    user_name: str
    candidates: List[MovieCandidate]


class DiscussionEntry(BaseModel):
    """Was ein Agent in der letzten Runde gesagt hat."""

    user_name: str
    reasoning: str
    top_choice_title: str
    veto_titles: List[str] = Field(default_factory=list)


class ConversationEntry(BaseModel):
    """Live-Nachricht in der Agenten-Diskussion."""

    user_name: str
    message: str
    round: int = 0
    turn: int = 0
    responding_to: str = ""


class DiscussionPrompt(BaseModel):
    """Moderator -> ein Agent: du bist dran in der Gruppendiskussion."""

    round: int
    turn: int
    candidates: List[MovieCandidate]
    conversation: List[ConversationEntry] = Field(default_factory=list)
    current_leader_title: Optional[str] = None


class DiscussionReply(BaseModel):
    """User-Agent -> Moderator: Beitrag in der Live-Diskussion."""

    user_id: str
    user_name: str
    round: int
    turn: int
    message: str
    responding_to: str = ""


class MoviePool(BaseModel):
    """Moderator -> alle Agenten: vollständiger Kandidaten-Pool."""

    candidates: List[MovieCandidate]
    round: int = 0
    # Diskussionsprotokoll der Vorrunde (leer in Runde 0)
    discussion: List[DiscussionEntry] = Field(default_factory=list)
    # Live-Diskussion der aktuellen Runde (vor der Abstimmung)
    conversation: List[ConversationEntry] = Field(default_factory=list)
    # aktueller Frontrunner (damit Agenten gezielt reagieren können)
    current_leader_title: Optional[str] = None


class AgentVote(BaseModel):
    """User-Agent -> Moderator: Ranking + Begründung."""

    user_id: str
    user_name: str
    ranked_movie_ids: List[str]
    scores: Dict[str, float]
    veto_movie_ids: List[str] = Field(default_factory=list)
    reasoning: str = ""
    round: int = 0
    # Reaktion auf andere (für Transparenz / Frontend)
    responding_to: str = ""


# ---------------------------------------------------------------------------
# Evaluation / Result
# ---------------------------------------------------------------------------
class MovieEvaluation(BaseModel):
    """(Legacy – nur falls noch irgendwo gebraucht)"""

    user_id: str
    user_name: str
    movie_id: str
    movie_title: str
    score: float  # 0-100
    reasoning: str


@dataclass
class GroupRecommendation:
    movie: MovieCandidate
    group_score: float
    individual_scores: Dict[str, float]
    fairness_score: float
