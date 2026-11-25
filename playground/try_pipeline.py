import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ai.decide import decide

question = "Which paddle board should I get? I weigh 90 kg."
decision = decide(question)
print(decision.model_dump_json(indent=2))