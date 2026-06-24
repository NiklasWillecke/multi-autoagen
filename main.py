import asyncio
import os

import uvicorn
from autogen_core import AgentId, SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agents.moderator_agent import ModeratorAgent
from agents.user_movie_agent import UserMovieAgent
from models.messages import NominationRequest, StartRecommendation
from services.access_guard import AccessGuardMiddleware, access_protection_enabled
from services.download_db import DownloadDB
from services.movie_search import MovieSearchService
from services.preference_parser import PreferenceParser
from services.usage_tracker import TrackingChatClient, UsageSummary

load_dotenv()

test = DownloadDB()
print("Hallo")
test.download()
print("Hallo")

app = FastAPI(title="Group Movie Recommendation System", version="1.0.0")
app.add_middleware(AccessGuardMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Singletons (einmal erstellen, wiederverwenden)
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# chroma DB search
search_service = MovieSearchService()


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/recommend")
async def recommend(request: StartRecommendation):
    if len(request.users) < 2 or len(request.users) > 6:
        raise HTTPException(400, "2-6 Nutzer erforderlich.")
    for user in request.users:
        if not user.raw_text.strip():
            raise HTTPException(
                400, f"Präferenz für {user.user_name} darf nicht leer sein."
            )

    # 1. Parse preferences (Funktionsaufrufe, parallel möglich)
    usage = UsageSummary(model=getattr(model_client, "model", "gpt-4o-mini"))
    tracking_client = TrackingChatClient(model_client, usage)
    parser = PreferenceParser(tracking_client)
    parsed_prefs = await asyncio.gather(
        *[
            parser.parse_preference(
                user_id=user.user_id,
                user_name=user.user_name,
                raw_text=user.raw_text,
            )
            for user in request.users
        ]
    )

    # 2. AutoGen Runtime nur für die echten Agents
    runtime = SingleThreadedAgentRuntime()

    user_agent_ids = [AgentId(f"user_{p.user_id}", "default") for p in parsed_prefs]

    await ModeratorAgent.register(
        runtime,
        "moderator",
        lambda: ModeratorAgent(
            expected_users=len(request.users),
            user_agent_ids=user_agent_ids,
            max_rounds=request.max_rounds,
            enable_discussion=request.enable_discussion,
            discussion_turns=request.discussion_turns,
        ),
    )
    mod_id = AgentId("moderator", "default")

    for pref in parsed_prefs:
        await UserMovieAgent.register(
            runtime,
            f"user_{pref.user_id}",
            lambda p=pref: UserMovieAgent(
                p.user_name, tracking_client, mod_id, search_service, p
            ),
        )

    runtime.start()

    # Kickoff
    for aid in user_agent_ids:
        await runtime.send_message(NominationRequest(), aid)

    await runtime.stop_when_idle()

    moderator = await runtime.try_get_underlying_agent_instance(mod_id, ModeratorAgent)
    result = moderator.result
    if result is None:
        raise HTTPException(
            500,
            "Keine Einigung erzielt. Mögliche Gründe: alle Kandidaten "
            "haben ein Veto, fehlende Agenten-Scores oder niemand erreicht "
            "den Mindest-Score von 3/10 für alle. Prüfe Chroma-Datenbank "
            "(./chroma-data) und OPENAI_API_KEY.",
        )

    return {
        "movie": result.movie.title,
        "movie_details": result.movie.model_dump(),
        "group_score": result.group_score,
        "individual_scores": result.individual_scores,
        "fairness_score": result.fairness_score,
        "transcript": moderator.transcript,
        "round_history": moderator.round_history,
        "candidates": moderator.candidates_summary,
        "usage": usage.to_dict(),
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "access_protected": access_protection_enabled(),
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
