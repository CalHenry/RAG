from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from rag.config import OPENROUTER_API_KEY, USE_OPENROUTER
from rag.data_models import RAGDeps, RAGResponse
from rag.query.helpers import retrieve

# from pydantic_ai.profiles import ModelProfile # uncomment for reasoning models

# Observability - debugging -------------------------------------------------
# import logfire
# logfire.configure()
# logfire.instrument_pydantic_ai()

# AI agent set up -----------------------------------------------------------
if USE_OPENROUTER:
    rag_agent_model = OpenAIChatModel(
        model_name="mistralai/ministral-3b-2512",
        provider=OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        ),
    )
else:
    rag_agent_model = OpenAIChatModel(
        model_name="ministral-3-3b-instruct-2512",
        provider=OpenAIProvider(
            base_url="http://127.0.0.1:1234/v1",
        ),
        # profile=ModelProfile(thinking_tags=("<think>", "</think>")), # uncomment for reasoning models
    )

rag_agent = Agent(
    rag_agent_model,
    deps_type=RAGDeps,
    output_type=RAGResponse,
    system_prompt="Tu es un assistant d'analyse documentaire. Réponds uniquement à partir du contexte fourni. Si la réponse n'y figure pas, dis-le clairement.",
    retries=2,
    model_settings={"temperature": 0.0},
)


@rag_agent.system_prompt
async def inject_context(ctx: RunContext[RAGDeps]) -> str:
    """
    1. retrieve() gather a list[dict] - the retrieved chunks from the vector db (see `data_models.ChunkResult`)
    2. format the dict into a string
    3. Add it to the system prompt of the AI agent (see the function decorator)
    """
    chunks = await retrieve(ctx.deps, ctx.deps.retrieval_query, doc_id=ctx.deps.doc_id)

    formatted = "\n\n".join(
        f"[{i + 1}] (score: {c['_distance']:.3f})\n{c['chunk_text']}"
        for i, c in enumerate(chunks)
    )
    return f"""
    Ce document traite-t-il de {ctx.deps.retrieval_query} ? Réponds d'abord par Oui ou Non. Si oui, liste chaque argument en une phrase, pas de résumé général.
    Contexte :
    ---
    {formatted}
    ---
    """
