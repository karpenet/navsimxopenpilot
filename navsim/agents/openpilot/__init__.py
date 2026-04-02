"""NAVSIM agent wrapping the openpilot vision + policy driving model (scaffold)."""

from navsim.agents.openpilot.op_agent import OpenpilotAgent

__all__ = ["OpenpilotAgent", "OpenpilotNavsimAgent"]


def __getattr__(name: str):
    """Lazy import so ONNX ``OpenpilotAgent`` works without comma ``openpilot`` on PYTHONPATH."""
    if name == "OpenpilotNavsimAgent":
        from navsim.agents.openpilot.openpilot_agent import OpenpilotNavsimAgent

        return OpenpilotNavsimAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
