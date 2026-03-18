import logfire
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from src.rag.data_models import RAGDeps, RAGResponse
from src.rag.query_pipeline_helpers import retrieve

logfire.configure()
logfire.instrument_pydantic_ai()

rag_agent_model = OpenAIChatModel(
    model_name="ministral-3-3b-instruct-2512",
    provider=OpenAIProvider(
        base_url="http://127.0.0.1:1234/v1",
    ),
    # profile=ModelProfile(thinking_tags=("<think>", "</think>")), # needed for reasoning models
)

rag_agent = Agent(
    rag_agent_model,
    deps_type=RAGDeps,
    output_type=RAGResponse,
    system_prompt="Tu es un assistant d'analyse documentaire. Réponds uniquement à partir du contexte fourni. Si la réponse n'y figure pas, dis-le clairement.",
    retries=2,
)


@rag_agent.system_prompt
async def inject_context(ctx: RunContext[RAGDeps]) -> str:
    """
    1.
    """
    chunks = await retrieve(ctx.deps, ctx.deps.retrieval_query, doc_id=ctx.deps.doc_id)

    formatted = "\n\n".join(
        f"[{i + 1}] (score: {c['_distance']:.3f} - date: {c['publish_date']})\n{c['chunk_text']}"
        for i, c in enumerate(chunks)
    )
    return f"""
    Ce document traite-t-il de {ctx.deps.retrieval_query} ? Réponds d'abord par Oui ou Non. Si oui, liste chaque argument en une phrase, pas de résumé général.
    Contexte :
    ---
    {formatted}
    ---
    """
