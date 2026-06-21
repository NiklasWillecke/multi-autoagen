import json

from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from models.messages import ParsedPreference
from services.usage_tracker import TrackingChatClient


class PreferenceParser:
    def __init__(self, model_client: OpenAIChatCompletionClient | TrackingChatClient) -> None:
        self.model_client = model_client
        self.system_message = SystemMessage(
            content=(
                "Extract structured movie preferences from the user input. "
                "Return valid JSON with exactly these fields: "
                "genres (list of strings), "
                "excluded_genres (list of strings), "
                "min_rating (float or null), "
                "semantic_query (string). "
                "The semantic_query should be a short natural-language search "
                "text suitable for vector database search. "
                "Include themes, atmosphere, setting, and desired movie type "
                "in semantic_query. "
                "Do not include any fields other than these four."
            )
        )

    async def parse_preference(
        self,
        user_id: str,
        user_name: str,
        raw_text: str,
    ) -> ParsedPreference:
        result = await self.model_client.create(
            [self.system_message, UserMessage(content=raw_text, source="user")],
            json_output=True,
        )

        if not isinstance(result.content, str):
            raise TypeError("Expected JSON string response, got function calls")

        data = json.loads(result.content)

        return ParsedPreference(
            user_id=user_id,
            user_name=user_name,
            **data,
        )
