# agents/moderator.py
from collections import defaultdict
from statistics import mean, stdev

from autogen_core import MessageContext, RoutedAgent, message_handler

from models.messages import (
    AgentVote,
    ConversationEntry,
    DiscussionPrompt,
    DiscussionReply,
    GroupRecommendation,
    MovieCandidate,
    MoviePool,
    Nomination,
)
from services.vote_scoring import trim_candidate_pool

MAX_SCORE_SPREAD = 5.8
CONSENSUS_SPREAD = 2.5
MIN_INDIVIDUAL_SCORE = 3.0

# Wie viele Filme sind am ende im Vote Pool
MAX_POOL_SIZE = 12


def compute_movie_score(scores: list[float]) -> tuple[float, float, float, float]:
    """Return avg (0-10), spread, group_score (0-100), fairness (0-100)."""
    if not scores:
        return 0.0, 0.0, 0.0, 0.0
    avg = mean(scores)
    spread = stdev(scores) if len(scores) > 1 else 0.0
    avg_100 = avg * 10
    fairness = max(0.0, 100.0 - (spread / MAX_SCORE_SPREAD) * 100.0)
    group_score = 0.7 * avg_100 + 0.3 * fairness
    return avg, spread, group_score, fairness


class ModeratorAgent(RoutedAgent):
    def __init__(
        self,
        expected_users,
        user_agent_ids,
        max_rounds=3,
        enable_discussion=False,
        discussion_turns=3,
    ):
        super().__init__("Coordinates cooperative discussion")
        self._expected = expected_users
        self._user_agents = user_agent_ids
        self._max_rounds = max_rounds
        self._enable_discussion = enable_discussion
        self._discussion_turns = discussion_turns
        self._round = 0

        self._nominations: list[MovieCandidate] = []
        self._nomination_count = 0
        self._pool: list[MovieCandidate] = []
        self._pool_by_id: dict[str, MovieCandidate] = {}

        self._votes: dict[int, list[AgentVote]] = defaultdict(list)
        self._final: GroupRecommendation | None = None
        self._candidates_summary: list[dict] = []
        self._round_history: list[dict] = []
        self.transcript: list[dict] = []

        self._conversation_thread: list[ConversationEntry] = []
        self._discussion_turn = 0
        self._discussion_speaker_idx = 0
        self._pending_discussion: list = []
        self._pending_leader_title: str | None = None

    #
    @message_handler
    async def on_nomination(self, message: Nomination, ctx: MessageContext) -> None:
        print(f"received message from user agent {ctx.sender}")
        print(message)
        self._nominations.extend(message.candidates)
        self._nomination_count += 1
        if self._nomination_count == self._expected:
            trimmed = trim_candidate_pool(self._nominations, max_size=MAX_POOL_SIZE)
            self._pool_by_id = {c.movie_id: c for c in trimmed}
            self._pool = trimmed
            if len(self._nominations) > len(self._pool):
                print(
                    f"ℹ️ Pool auf {MAX_POOL_SIZE} Kandidaten reduziert "
                    f"(von {len(self._nominations)} Nominierungen)."
                )
            if not self._pool:
                print("⚠️ No movie candidates nominated - cannot start voting.")
                return
            await self._begin_round(discussion=[], leader_title=None)

    async def _begin_round(self, discussion, leader_title) -> None:
        self._pending_discussion = discussion
        self._pending_leader_title = leader_title
        if self._enable_discussion:
            await self._start_discussion()
        else:
            await self._broadcast(
                discussion=discussion,
                leader_title=leader_title,
                conversation=[],
            )

    async def _start_discussion(self) -> None:
        self._conversation_thread = []
        self._discussion_turn = 0
        self._discussion_speaker_idx = 0
        await self._prompt_next_speaker()

    async def _prompt_next_speaker(self) -> None:
        if self._discussion_turn >= self._discussion_turns:
            await self._broadcast(
                discussion=self._pending_discussion,
                leader_title=self._pending_leader_title,
                conversation=list(self._conversation_thread),
            )
            return

        agent_id = self._user_agents[self._discussion_speaker_idx]
        await self.send_message(
            DiscussionPrompt(
                round=self._round,
                turn=self._discussion_turn,
                candidates=self._pool,
                conversation=list(self._conversation_thread),
                current_leader_title=self._pending_leader_title,
            ),
            agent_id,
        )

    @message_handler
    async def on_discussion_reply(
        self, message: DiscussionReply, ctx: MessageContext
    ) -> None:
        if message.round != self._round:
            return

        entry = ConversationEntry(
            user_name=message.user_name,
            message=message.message,
            round=message.round,
            turn=message.turn,
            responding_to=message.responding_to,
        )
        self._conversation_thread.append(entry)
        self.transcript.append(
            {
                "type": "discussion",
                "round": message.round,
                "turn": message.turn,
                "user_name": message.user_name,
                "message": message.message,
                "responding_to": message.responding_to,
            }
        )

        self._discussion_speaker_idx += 1
        if self._discussion_speaker_idx >= self._expected:
            self._discussion_speaker_idx = 0
            self._discussion_turn += 1
        await self._prompt_next_speaker()

    async def _broadcast(self, discussion, leader_title, conversation) -> None:
        pool_msg = MoviePool(
            candidates=self._pool,
            round=self._round,
            discussion=discussion,
            conversation=conversation,
            current_leader_title=leader_title,
        )
        for aid in self._user_agents:
            await self.send_message(pool_msg, aid)

    @message_handler
    async def on_vote(self, message: AgentVote, ctx: MessageContext) -> None:
        if message.round != self._round:
            return
        self._votes[self._round].append(message)

        self.transcript.append(
            {
                "type": "vote",
                "round": message.round,
                "user_name": message.user_name,
                "reasoning": message.reasoning,
                "responding_to": message.responding_to,
                "top_choice_title": (
                    self._pool_by_id[message.ranked_movie_ids[0]].title
                    if message.ranked_movie_ids
                    and message.ranked_movie_ids[0] in self._pool_by_id
                    else None
                ),
                "scores": {
                    self._pool_by_id[movie_id].title: score
                    for movie_id, score in message.scores.items()
                    if movie_id in self._pool_by_id
                },
                "veto_titles": [
                    self._pool_by_id[movie_id].title
                    for movie_id in message.veto_movie_ids
                    if movie_id in self._pool_by_id
                ],
            }
        )

        if len(self._votes[self._round]) == self._expected:
            await self._evaluate_round()

    def _collect_movie_scores(
        self, votes: list[AgentVote], movie_id: str
    ) -> list[float] | None:
        scores: list[float] = []
        for vote in votes:
            if movie_id not in vote.scores:
                return None
            scores.append(vote.scores[movie_id])
        return scores

    def _score_eligible_movies(
        self, votes: list[AgentVote], *, require_min_score: bool
    ) -> list[tuple[float, float, float, MovieCandidate, list[float]]]:
        vetoed = {mid for v in votes for mid in v.veto_movie_ids}
        out: list[tuple[float, float, float, MovieCandidate, list[float]]] = []

        for movie in self._pool:
            if movie.movie_id in vetoed:
                continue

            scores = self._collect_movie_scores(votes, movie.movie_id)
            if scores is None:
                continue

            min_score = min(scores)
            if require_min_score and min_score < MIN_INDIVIDUAL_SCORE:
                continue

            _, spread, group_score, _ = compute_movie_score(scores)
            out.append((group_score, spread, min_score, movie, scores))

        return out

    async def _evaluate_round(self) -> None:
        votes = self._votes[self._round]
        last_round = self._round >= self._max_rounds - 1
        scored = self._score_eligible_movies(votes, require_min_score=True)

        consensus = False
        leader_title: str | None = None
        leader_group_score: float | None = None
        is_final = last_round

        if scored:
            scored.sort(key=lambda x: (x[0], x[2]), reverse=True)
            best_score, best_spread, _, best_movie, _ = scored[0]
            consensus = best_spread <= CONSENSUS_SPREAD
            leader_title = best_movie.title
            leader_group_score = round(best_score, 1)
            is_final = last_round

        self._round_history.append(
            self._build_round_summary(
                votes,
                round_num=self._round,
                consensus=consensus,
                is_final=is_final,
                leader_title=leader_title,
                leader_group_score=leader_group_score,
            )
        )

        if not scored:
            print(
                "⚠️ No eligible movies this round "
                "(vetoes, missing scores, or min-score rule)."
            )
            self._candidates_summary = self._build_candidates_summary(votes)
            if last_round:
                return
            self._round += 1
            await self._begin_round(
                discussion=self._build_discussion(votes),
                leader_title=None,
            )
            return

        best_score, best_spread, _, best_movie, _ = scored[0]

        if last_round:
            self._finalize(best_movie, best_score, best_spread, votes)
            return

        self._round += 1
        await self._begin_round(
            discussion=self._build_discussion(votes),
            leader_title=best_movie.title,
        )

    def _build_discussion(self, votes) -> list:
        from models.messages import DiscussionEntry

        entries = []
        for v in votes:
            top_id = v.ranked_movie_ids[0] if v.ranked_movie_ids else None
            top_title = (
                self._pool_by_id[top_id].title if top_id in self._pool_by_id else "?"
            )
            veto_titles = [
                self._pool_by_id[m].title
                for m in v.veto_movie_ids
                if m in self._pool_by_id
            ]
            entries.append(
                DiscussionEntry(
                    user_name=v.user_name,
                    reasoning=v.reasoning,
                    top_choice_title=top_title,
                    veto_titles=veto_titles,
                )
            )
        return entries

    def _build_standings(self, votes: list[AgentVote]) -> list[dict]:
        vetoed = {mid for v in votes for mid in v.veto_movie_ids}
        summaries = []

        for movie in self._pool:
            individual = {
                v.user_name: v.scores[movie.movie_id]
                for v in votes
                if movie.movie_id in v.scores
            }
            scores_list = list(individual.values())
            complete = len(individual) == len(votes)
            min_score = min(scores_list) if scores_list else None

            if scores_list and complete:
                avg, _, group_score, _ = compute_movie_score(scores_list)
            elif scores_list:
                avg = mean(scores_list)
                group_score = 0.0
            else:
                avg, group_score = 0.0, 0.0

            eligible = (
                movie.movie_id not in vetoed
                and complete
                and min_score is not None
                and min_score >= MIN_INDIVIDUAL_SCORE
            )

            summaries.append(
                {
                    "movie": movie.model_dump(),
                    "average_score": round(avg, 1),
                    "group_score": round(group_score, 1),
                    "individual_scores": individual,
                    "vetoed": movie.movie_id in vetoed,
                    "eligible": eligible,
                    "complete_scores": complete,
                }
            )

        summaries.sort(key=lambda x: x["group_score"], reverse=True)
        return summaries

    def _build_round_summary(
        self,
        votes: list[AgentVote],
        *,
        round_num: int,
        consensus: bool,
        is_final: bool,
        leader_title: str | None,
        leader_group_score: float | None,
    ) -> dict:
        standings = self._build_standings(votes)
        eligible = [entry for entry in standings if entry["eligible"]]

        vote_entries = []
        for vote in votes:
            top_id = vote.ranked_movie_ids[0] if vote.ranked_movie_ids else None
            top_title = (
                self._pool_by_id[top_id].title
                if top_id and top_id in self._pool_by_id
                else None
            )
            vote_entries.append(
                {
                    "user_name": vote.user_name,
                    "top_choice_title": top_title,
                    "scores": {
                        self._pool_by_id[movie_id].title: score
                        for movie_id, score in vote.scores.items()
                        if movie_id in self._pool_by_id
                    },
                    "veto_titles": [
                        self._pool_by_id[movie_id].title
                        for movie_id in vote.veto_movie_ids
                        if movie_id in self._pool_by_id
                    ],
                    "reasoning": vote.reasoning,
                    "responding_to": vote.responding_to,
                }
            )

        return {
            "round": round_num,
            "display_round": round_num + 1,
            "leader_title": leader_title,
            "leader_group_score": leader_group_score,
            "consensus": consensus,
            "is_final": is_final,
            "eligible_count": len(eligible),
            "votes": vote_entries,
            "standings": standings,
        }

    def _build_candidates_summary(self, votes: list[AgentVote]) -> list[dict]:
        return self._build_standings(votes)

    def _finalize(self, movie, score, spread, votes) -> None:
        self._candidates_summary = self._build_candidates_summary(votes)
        self._final = GroupRecommendation(
            movie=movie,
            group_score=score,
            individual_scores={
                v.user_name: v.scores[movie.movie_id]
                for v in votes
                if movie.movie_id in v.scores
            },
            fairness_score=max(0.0, 100.0 - (spread / MAX_SCORE_SPREAD) * 100.0),
        )
        print(f"\n🎬 Consensus after {self._round + 1} round(s): {movie.title}")
        print(f"Group Score: {score:.1f} | Spread: {spread:.1f}")

    @property
    def result(self) -> GroupRecommendation | None:
        return self._final

    @property
    def candidates_summary(self) -> list[dict]:
        return self._candidates_summary

    @property
    def round_history(self) -> list[dict]:
        return self._round_history
