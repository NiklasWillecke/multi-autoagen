import json

from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from models.messages import (
    AgentVote,
    DiscussionPrompt,
    DiscussionReply,
    MoviePool,
    Nomination,
    NominationRequest,
)
from services.movie_search import MovieSearchService
from services.usage_tracker import TrackingChatClient
from services.vote_scoring import (
    MAX_SCORE_RETRIES,
    build_missing_scores_retry_prompt,
    build_vote_prompt_suffix,
    build_vote_response_format,
    missing_score_ids,
    normalize_scores,
    normalize_vetos,
)


class UserMovieAgent(RoutedAgent):
    def __init__(self, user_name, model_client, moderator_id, search, preference):
        super().__init__(f"Movie agent for {user_name}")
        self._user_name = user_name
        self._model_client: OpenAIChatCompletionClient | TrackingChatClient = (
            model_client
        )
        self._moderator_id = moderator_id
        self._preference = preference
        self._movie_service: MovieSearchService = search

    # Jeder Agent Nomeniert sein Filme
    @message_handler
    async def on_nomination_request(
        self, message: NominationRequest, ctx: MessageContext
    ) -> None:
        candidates = self._movie_service.search(preference=self._preference)
        await self.send_message(
            Nomination(
                user_id=self._preference.user_id,
                user_name=self._user_name,
                candidates=candidates[:5],
            ),
            self._moderator_id,
        )

    #
    @message_handler
    async def on_discussion_prompt(
        self, message: DiscussionPrompt, ctx: MessageContext
    ) -> None:
        reply = await self._generate_discussion_reply(message)
        print(f"Discussion {self.id}: {reply}")
        await self.send_message(reply, self._moderator_id)

    #
    @message_handler
    async def on_movie_pool(self, message: MoviePool, ctx: MessageContext) -> None:
        vote = await self._discuss_and_vote(message)
        print(f"Voting {self.id}: {vote}")
        await self.send_message(vote, self._moderator_id)

    async def _generate_discussion_reply(
        self, prompt: DiscussionPrompt
    ) -> DiscussionReply:
        movies_text = "\n".join(
            f"- [{c.movie_id}] {c.title}: {c.overview[:120]}" for c in prompt.candidates
        )

        if prompt.conversation:
            conversation_text = "\n".join(
                f'{e.user_name}: "{e.message}"'
                + (f" (→ @{e.responding_to})" if e.responding_to else "")
                for e in prompt.conversation
            )
            conversation_block = (
                f"\n--- Group chat so far ---\n{conversation_text}\n"
                "Reply directly to others by name. Pick up a specific point "
                "someone made, compare at least two candidates, explain "
                "trade-offs for the group, or ask a clarifying question "
                "before suggesting compromises.\n"
            )
        else:
            conversation_block = (
                "\nYou speak first in this round. Present your top pick with "
                "concrete reasons (mood, genre, who it suits) and one "
                "alternative from the list that could also work.\n"
            )

        turn_block = ""
        if prompt.turn == 0:
            turn_block = (
                "\nThis is the opening of the round — set out your position "
                "clearly; don't rush to a group verdict yet.\n"
            )
        elif prompt.turn >= 1:
            turn_block = (
                "\nThe discussion is ongoing — dig deeper: challenge weak "
                "arguments, name what you'd give up, and what you need from "
                "others before you'd switch picks.\n"
            )

        leader_block = ""
        if prompt.current_leader_title:
            leader_block = (
                f"\nCurrent group frontrunner from last vote: "
                f"{prompt.current_leader_title}\n"
            )

        prompt_text = (
            f"You are '{self._user_name}', discussing a group movie choice.\n"
            f"Your taste: {self._preference.semantic_query}\n"
            f"Liked: {self._preference.genres}, "
            f"avoid: {self._preference.excluded_genres}\n\n"
            f"Candidates (ONLY these films may be discussed):\n{movies_text}\n"
            f"{leader_block}"
            f"{conversation_block}"
            f"{turn_block}\n"
            "Only mention movies from the candidate list above, using their "
            "exact titles. Do not suggest or reference films that are not "
            "on the list.\n"
            "Write a substantive chat message (about 5-8 sentences). Be "
            "friendly but advocate for your own taste — compromise is fine, "
            "but don't pretend you love films outside your preferences.\n"
            'Return JSON: {"message": str, "responding_to": str}'
        )

        result = await self._model_client.create(
            [
                SystemMessage(
                    content="You are a friend in a lively group chat picking "
                    "a movie together. You respond by name, explain your "
                    "reasoning in detail, compare options, and negotiate "
                    "without ending the debate too quickly."
                ),
                UserMessage(content=prompt_text, source="user"),
            ],
            json_output=True,
        )

        if not isinstance(result.content, str):
            raise TypeError(
                f"Expected JSON string response, got {type(result.content)}"
            )

        data = json.loads(result.content)
        return DiscussionReply(
            user_id=self._preference.user_id,
            user_name=self._user_name,
            round=prompt.round,
            turn=prompt.turn,
            message=data.get("message", ""),
            responding_to=data.get("responding_to", ""),
        )

    async def _discuss_and_vote(self, pool: MoviePool) -> AgentVote:
        pool_ids = {candidate.movie_id for candidate in pool.candidates}
        pool_id_list = sorted(pool_ids)
        movies_text = "\n".join(
            f"- [{c.movie_id}] {c.title}: {c.overview[:120]}" for c in pool.candidates
        )

        blocks = []

        if pool.conversation:
            chat = "\n".join(
                f'{e.user_name}: "{e.message}"'
                + (f" (→ @{e.responding_to})" if e.responding_to else "")
                for e in pool.conversation
            )
            blocks.append(f"--- Live group discussion ---\n{chat}")

        if pool.discussion:
            others = [d for d in pool.discussion if d.user_name != self._user_name]
            discussion_text = "\n".join(
                f'{d.user_name} voted for "{d.top_choice_title}": "{d.reasoning}"'
                + (f" (vetoed: {', '.join(d.veto_titles)})" if d.veto_titles else "")
                for d in others
            )
            blocks.append(
                f"--- Previous round votes ---\n{discussion_text}\n"
                f"Current group frontrunner: {pool.current_leader_title}"
            )

        context_block = "\n\n".join(blocks)
        if context_block:
            context_block += (
                "\n\nUse the discussion when scoring, but stay honest about "
                "your own taste. Mention whose argument influenced you.\n"
            )

        prompt = (
            f"You are '{self._user_name}', picking a movie WITH friends.\n"
            f"Your taste: {self._preference.semantic_query}\n"
            f"Liked: {self._preference.genres}, "
            f"avoid: {self._preference.excluded_genres}\n\n"
            f"Candidates:\n{movies_text}\n\n"
            f"{context_block}\n"
            "Score each film 0-10 for how much YOU would enjoy it given your "
            "taste above — not how happy others would be. Films outside your "
            "preferred genres or in your avoid list should rarely score above "
            "5 unless you genuinely changed your mind. A workable group pick "
            "often lands around 6-8 for everyone, not 10/10 from all.\n"
            f"{build_vote_prompt_suffix(pool_ids, len(pool.candidates))}"
        )

        messages = [
            SystemMessage(
                content="You are picking a group movie. Score honestly from "
                "your own preferences; compromise in rankings, not by giving "
                "everyone's favorite a 10."
            ),
            UserMessage(content=prompt, source="user"),
        ]

        data = await self._request_vote_json(messages, pool_id_list)
        raw_scores = data.get("scores", {})

        for attempt in range(MAX_SCORE_RETRIES):
            missing = missing_score_ids(raw_scores, pool_ids)
            if not missing:
                break
            messages.append(
                UserMessage(
                    content=build_missing_scores_retry_prompt(missing),
                    source="user",
                )
            )
            data = await self._request_vote_json(messages, pool_id_list)
            raw_scores = data.get("scores", {})

        scores = normalize_scores(raw_scores, pool_ids)
        veto_movie_ids = normalize_vetos(
            data.get("veto_movie_ids", []), scores, pool_ids
        )
        ranked_movie_ids = [
            movie_id
            for movie_id in data.get("ranked_movie_ids", [])
            if movie_id in pool_ids
        ]
        if not ranked_movie_ids:
            ranked_movie_ids = sorted(
                pool_ids, key=lambda movie_id: scores[movie_id], reverse=True
            )

        return AgentVote(
            user_id=self._preference.user_id,
            user_name=self._user_name,
            round=pool.round,
            ranked_movie_ids=ranked_movie_ids,
            scores=scores,
            veto_movie_ids=veto_movie_ids,
            reasoning=data.get("reasoning", ""),
            responding_to=data.get("responding_to", ""),
        )

    async def _request_vote_json(self, messages: list, pool_id_list: list[str]) -> dict:
        try:
            result = await self._model_client.create(
                messages,
                extra_create_args={
                    "response_format": build_vote_response_format(pool_id_list),
                },
            )
        except Exception as exc:
            print(
                f"Vote schema request failed for {self._user_name}, "
                f"falling back to json_output: {exc}"
            )
            result = await self._model_client.create(messages, json_output=True)

        if not isinstance(result.content, str):
            raise TypeError(
                f"Expected JSON string response, got {type(result.content)}"
            )

        return json.loads(result.content)
