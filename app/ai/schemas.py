from typing import Literal

from pydantic import BaseModel, Field, field_validator

Section = Literal["paddle-boards", "shelving", "returns-and-warranty", "shipping", "other"]


class Classification(BaseModel):
    section: Section
    language: str = Field(description="Language of the latest customer message, e.g. English")
    needs_human: bool = Field(description="True when an always-escalate rule applies")
    reason: str = Field(description="One sentence explaining the classification")
    acknowledgement: str = Field(
        description="Short note to the customer, in their language, that a colleague replies"
    )


class Issue(BaseModel):
    product: Literal["paddle-board", "shelving"]
    problem: Literal[
        "missing-accessory", "slow-leak", "seam-split",
        "missing-hardware", "damaged-panels", "damaged-frame", "wrong-finish",
    ] = Field(
        description=(
            "For paddle-board: missing-accessory (fin, pump, paddle, leash, wrench, or bag), "
            "slow-leak, or seam-split. For shelving: missing-hardware (screws, cam locks, "
            "dowels, or other parts of the hardware kit), damaged-panels, damaged-frame, "
            "or wrong-finish."
        )
    )
    order_number: str | None = None
    days_since_delivery: int | None = None
    valve_checked: bool = Field(default=False, description="True when the valve check was done")
    part_numbers: list[str] = Field(
        default_factory=list, description="Part numbers of damaged panels, e.g. HP-S3-04"
    )

    # Models often send null for "nothing". Rejecting it makes the model retry, and after a
    # failed validation gpt-oss tends to answer with empty messages until the run gives up.
    @field_validator("valve_checked", mode="before")
    @classmethod
    def _null_is_false(cls, value: object) -> object:
        return False if value is None else value

    @field_validator("part_numbers", mode="before")
    @classmethod
    def _null_is_empty(cls, value: object) -> object:
        return [] if value is None else value


class AgentReply(BaseModel):
    action: Literal["reply", "escalate"]
    reply: str = Field(description="The email to the customer, plain text")
    reasoning: str = Field(description="Which knowledge base entries you used, and why")
    knowledge_base_covers_it: bool = Field(
        description=(
            "True when the knowledge base tells you how to answer, including when the answer is "
            "to ask the customer for missing details such as the order number. False only when "
            "the knowledge base does not cover the question."
        )
    )
    promises_an_action: bool = Field(
        description="True when the reply promises a refund, shipment, replacement, date, or call"
    )
    issue: Issue | None = Field(
        default=None, description="Fill in when the customer reports a problem with a product"
    )


class Decision(BaseModel):
    action: Literal["reply", "escalate"]
    section: str
    reply: str
    reason: str
    proposed_action: str | None = None