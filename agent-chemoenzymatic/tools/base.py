"""Tool base class and registry for the retrosynthesis agent.

Three-tier tool system:
- Tier 1 (Perception): Pure RDKit tools for understanding molecules
- Tier 2 (Action): Pure RDKit tools for manipulating molecules
- Tier 3 (Knowledge): External-dependency tools (models, databases, APIs)

Tools support `detail_level` parameter:
- "brief" (default): minimal output for LLM decision-making
- "full": complete output when LLM needs deeper info
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Base class for all tools."""
    name: str = ""
    description: str = ""
    parameters: dict = {}  # JSON Schema for parameters
    tier: int = 1           # 1=perception, 2=action, 3=knowledge

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool and return a JSON string result."""
        ...

    def to_openai_tool(self) -> dict:
        """Convert to OpenAI function-calling tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_anthropic_tool(self) -> dict:
        """Convert to Anthropic Messages-API tool format.

        Anthropic uses {name, description, input_schema} at the top level
        rather than wrapping in {type:function, function:{...}}.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


def _add_detail_level_param(params: dict) -> dict:
    """Inject detail_level parameter into a tool's JSON schema if not present."""
    props = params.get("properties", {})
    if "detail_level" not in props:
        props["detail_level"] = {
            "type": "string",
            "enum": ["brief", "full"],
            "description": "Output verbosity. 'brief' (default) returns minimal info; 'full' returns everything.",
            "default": "brief",
        }
        params["properties"] = props
    return params


class ToolRegistry:
    """Registry for tools, provides lookup and OpenAI schema generation."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        logger.info("Registered tool [tier %d]: %s", tool.tier, tool.name)

    def register_optional(self, tool_factory, *args, **kwargs) -> bool:
        """Try to register a tool; swallow errors if deps are missing."""
        try:
            tool = tool_factory(*args, **kwargs)
            self.register(tool)
            return True
        except Exception as e:
            logger.warning("Optional tool registration failed: %s", e)
            return False

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, arguments: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return tool.execute(**arguments)
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return json.dumps({"error": str(e)})

    def get_openai_tools(self, exclude_internal: bool = True) -> list[dict]:
        """Get tools as OpenAI function-calling schema.
        By default, excludes internal tools (name starts with '_') from the LLM-facing list.
        """
        tools = self._tools.values()
        if exclude_internal:
            tools = [t for t in tools if not t.name.startswith("_")]
        return [t.to_openai_tool() for t in tools]

    def get_anthropic_tools(self, exclude_internal: bool = True) -> list[dict]:
        """Get tools as Anthropic Messages-API tool schema."""
        tools = self._tools.values()
        if exclude_internal:
            tools = [t for t in tools if not t.name.startswith("_")]
        return [t.to_anthropic_tool() for t in tools]

    def get_tools_by_tier(self, tier: int) -> list[Tool]:
        return [t for t in self._tools.values() if t.tier == tier]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())
