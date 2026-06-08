"""Tests for tools, memory, and the ReAct agent."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.agents.tools import CalculatorTool, CurrentDateTool, ToolRegistry


class TestCalculatorTool:
    def setup_method(self):
        self.tool = CalculatorTool()

    def test_basic_arithmetic(self):
        assert self.tool("2 + 2").output == "4.0"
        assert self.tool("10 * 5").output == "50.0"
        assert self.tool("100 / 4").output == "25.0"

    def test_power(self):
        result = self.tool("2 ** 10")
        assert result.output == "1024.0"

    def test_sqrt(self):
        result = self.tool("sqrt(144)")
        assert result.output == "12.0"

    def test_complex_expression(self):
        result = self.tool("(10 + 5) * 2 - 3")
        assert result.output == "27.0"

    def test_invalid_expression_returns_error(self):
        result = self.tool("import os; os.system('rm -rf /')")
        assert result.error is not None

    def test_division_by_zero(self):
        result = self.tool("1 / 0")
        assert result.error is not None


class TestCurrentDateTool:
    def test_returns_date_string(self):
        tool = CurrentDateTool()
        result = tool()
        assert result.success
        assert "UTC" in result.output


class TestToolRegistry:
    def test_register_and_retrieve(self):
        registry = ToolRegistry()
        calc = CalculatorTool()
        registry.register(calc)
        assert registry.get("calculator") is calc

    def test_execute_known_tool(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        result = registry.execute("calculator", {"expression": "3 + 3"})
        assert result.success
        assert result.output == "6.0"

    def test_execute_unknown_tool_returns_error(self):
        registry = ToolRegistry()
        result = registry.execute("nonexistent", {})
        assert result.error is not None
        assert "Unknown tool" in result.error

    def test_openai_specs_format(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(CurrentDateTool())
        specs = registry.openai_specs()
        assert len(specs) == 2
        for spec in specs:
            assert "type" in spec
            assert "function" in spec
            assert "name" in spec["function"]


class TestConversationMemory:
    def test_add_and_retrieve_messages(self):
        from src.agents.memory import ConversationMemory
        with patch("src.agents.memory.OpenAI"):
            mem = ConversationMemory(max_tokens=5000)
            mem.set_system("You are a helpful assistant.")
            mem.add_user("Hello")
            mem.add_assistant("Hi there!")

            msgs = mem.get_messages()
            assert msgs[0]["role"] == "system"
            assert any(m["content"] == "Hello" for m in msgs)
            assert any(m["content"] == "Hi there!" for m in msgs)

    def test_window_size_limits_messages(self):
        from src.agents.memory import ConversationMemory
        with patch("src.agents.memory.OpenAI"):
            mem = ConversationMemory(max_tokens=50000, window_size=2)
            for i in range(10):
                mem.add_user(f"User message {i}")
                mem.add_assistant(f"Assistant reply {i}")

            msgs = [m for m in mem.get_messages() if m["role"] != "system"]
            # window_size=2 → 4 messages max
            assert len(msgs) <= 4

    def test_clear_resets_history(self):
        from src.agents.memory import ConversationMemory
        with patch("src.agents.memory.OpenAI"):
            mem = ConversationMemory()
            mem.set_system("System prompt")
            mem.add_user("test")
            mem.clear()
            mem.set_system("System prompt")
            msgs = mem.get_messages()
            assert len(msgs) == 1  # only system prompt
