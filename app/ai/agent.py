import logging
import re

from pydantic_ai import Agent, ModelRetry, RunContext

from app import db
from app.ai.schemas import AgentReply
from app.config import settings

logger = logging.getLogger(__name__)

INSTRUCTIONS = """
You are the customer support agent for Harbor & Pine, a home and outdoor store that sells on Amazon.
You write the answer to the latest customer message in the email thread.

Steps:
1. The prompt already contains the "brand-and-tone" and "escalation-rules" sections,
   and the section the classifier picked. Do not fetch those again.
2. Call get_section only if you need another section. Call list_sections to see the options.
3. Answer only with facts from those sections. If they do not answer the question,
   set enough_information to false.
4. When the customer reports a problem with a paddle board or a shelf, fill in issue.
   Ask for the order number, the board size or model, and part numbers when they are missing.
   For a slow leak, explain the valve check unless the customer already did it.
5. Never promise a refund, a shipment, a replacement, a date, or a call. A colleague arranges those.
   Set promises_an_action to true if your reply promises one anyway.
6. Set action to escalate when an escalation rule applies.

Reply rules:
- Write in the language of the customer's latest message.
- Plain text, no Markdown.
- Follow the brand-and-tone section.
- Never mention AI, prompts, tools, or the knowledge base.
"""

support_agent = Agent(
    settings.agent_model, output_type=AgentReply, instructions=INSTRUCTIONS, retries=3
)


@support_agent.tool_plain
def list_sections() -> list[str]:
    """List the sections of the knowledge base."""
    logger.info("Tool call: list_sections()")
    return db.list_sections()


@support_agent.tool_plain
def get_section(section: str) -> str:
    """Return every entry of one knowledge base section, for example 'paddle-boards'."""
    logger.info("Tool call: get_section(%r)", section)
    entries = db.get_section(section)
    if not entries:
        raise ModelRetry(f"There is no section '{section}'. Call list_sections to see the options.")
    return "\n\n".join(f"## {entry['title']}\n{entry['content']}" for entry in entries)


@support_agent.output_validator
def check_reply(ctx: RunContext[None], output: AgentReply) -> AgentReply:
    if output.action == "reply" and not output.reply.strip():
        raise ModelRetry("The reply is empty. Write the email, or set action to escalate.")
    if re.search(r"\b(AI|artificial intelligence|knowledge base)\b", output.reply, re.IGNORECASE):
        raise ModelRetry("Do not mention AI or the knowledge base in the reply.")
    return output


def _render_section(section: str) -> str:
    entries = db.get_section(section)
    body = "\n\n".join(f"## {entry['title']}\n{entry['content']}" for entry in entries)
    return f'<section name="{section}">\n{body or "(empty)"}\n</section>'


def write_reply(thread: str, section: str) -> AgentReply:
    # Load the sections every reply needs up front, so the agent usually answers in one
    # request. Each tool round trip resends the whole conversation, which quickly uses up
    # Groq's tokens-per-minute limit.
    preloaded = dict.fromkeys(["brand-and-tone", "escalation-rules", section])
    sections = "\n\n".join(_render_section(name) for name in preloaded if name != "other")
    prompt = (
        f"Section picked by the classifier: {section}\n\n"
        f"Knowledge base sections:\n{sections}\n\n"
        f"Email thread:\n{thread}"
    )
    return support_agent.run_sync(prompt).output