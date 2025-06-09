import pytest
from pressure_agi.engine.field import Field
from pressure_agi.engine.memory import EpisodicMemory
from pressure_agi.engine.critic import Critic
from pressure_agi.demos.repl import step_once

@pytest.mark.asyncio
async def test_loop_once():
    from pressure_agi.demos.repl import step_once
    mem = EpisodicMemory()
    field = Field(n=0, device='cpu')
    critic = Critic()
    action = await step_once(
        "hello",
        field,
        critic,
        mem,
        loop_count=1,
        settle_steps=10,
        pos_threshold=0.1,
        neg_threshold=-0.1,
        verbose=False
    )
    assert action in {"say_positive","say_neutral","say_negative"}
    assert len(mem.G) == 1
    snapshot = mem.retrieve_last(k=1)[0]
    assert snapshot['t'] == 1
    assert snapshot['decision'] == action 