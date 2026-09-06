"""Tests for AIAgent.steer() — mid-run user message injection.

/steer lets the user add a note to the agent's next tool result without
interrupting the current tool call. The agent sees the note inline with
tool output on its next iteration, preserving message-role alternation
and prompt-cache integrity.
"""
from __future__ import annotations

import threading

import pytest

from agent.prompt_builder import STEER_MARKER_OPEN, format_steer_marker
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    """Build an AIAgent without running __init__, then install the steer
    state manually — matches the existing object.__new__ stub pattern
    used elsewhere in the test suite.
    """
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._model_request_active = threading.Event()
    agent._executing_tools = False
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent._tool_worker_threads = None
    agent._tool_worker_threads_lock = None
    agent._current_streamed_assistant_text = ""
    agent._stream_needs_break = False
    agent._strip_think_blocks = lambda content: content
    agent.quiet_mode = True
    agent.api_mode = "chat_completions"
    return agent


class TestSteerAcceptance:
    def test_accepts_non_empty_text(self):
        agent = _bare_agent()
        assert agent.steer("go ahead and check the logs") is True
        assert agent._pending_steer == "go ahead and check the logs"







class TestSteerDrain:
    def test_drain_returns_and_clears(self):
        agent = _bare_agent()
        agent.steer("hello")
        assert agent._drain_pending_steer() == "hello"
        assert agent._pending_steer is None



class TestActiveTurnRedirect:
    def test_rejects_when_no_turn_is_active(self):
        agent = _bare_agent()
        assert agent.redirect("change course") is False
        assert agent._pending_redirect is None

    def test_cancels_only_an_active_model_request(self):
        agent = _bare_agent()
        agent._model_request_active.set()

        assert agent.redirect("use Postgres") is True
        assert agent._pending_redirect == "use Postgres"
        assert agent._interrupt_requested is True
        assert agent._interrupt_message is None

    def test_multiple_redirects_preserve_message_boundaries(self):
        agent = _bare_agent()
        agent._model_request_active.set()

        assert agent.redirect("first correction") is True
        assert agent.redirect("second correction") is True
        assert agent._pending_redirect == (
            "first correction\n\n"
            "[Additional user correction]\n"
            "second correction"
        )

    def test_hard_interrupt_wins_over_new_redirect(self):
        agent = _bare_agent()
        agent._model_request_active.set()
        agent._interrupt_requested = True

        assert agent.redirect("too late") is False
        assert agent._pending_redirect is None

    def test_reasoning_deltas_are_display_only(self):
        """Streamed reasoning must never accumulate into replayable transcript
        state — an assistant checkpoint that inlines chain-of-thought trips
        Anthropic's output classifier and permanently bricks the session
        (deterministic empty-response storms on every replay)."""
        agent = _bare_agent()
        seen = []
        agent.reasoning_callback = seen.append

        agent._fire_reasoning_delta("visible provider thinking")

        # Displayed to the surface, but never checkpointed anywhere.
        assert seen == ["visible provider thinking"]
        assert not getattr(agent, "_current_streamed_reasoning_text", "")

    def test_response_completion_before_redirect_lock_rejects_correction(self):
        agent = _bare_agent()
        agent._model_request_active.set()
        started = threading.Event()
        outcome = {}

        def redirect():
            started.set()
            outcome["accepted"] = agent.redirect("late correction")

        with agent._pending_redirect_lock:
            worker = threading.Thread(target=redirect)
            worker.start()
            assert started.wait(timeout=1)
            # Mirrors conversation_loop clearing the request-active marker
            # under this same lock before redirect can commit its slot.
            agent._model_request_active.clear()
        worker.join(timeout=1)

        assert outcome["accepted"] is False
        assert agent._pending_redirect is None

    def test_hard_stop_wins_concurrent_redirect(self):
        agent = _bare_agent()
        agent._model_request_active.set()
        start = threading.Barrier(3)
        outcome = {}

        def redirect():
            start.wait()
            outcome["redirect"] = agent.redirect("change course")

        def hard_stop():
            start.wait()
            agent.interrupt("stop requested")

        redirect_thread = threading.Thread(target=redirect)
        stop_thread = threading.Thread(target=hard_stop)
        redirect_thread.start()
        stop_thread.start()
        start.wait()
        redirect_thread.join(timeout=1)
        stop_thread.join(timeout=1)

        assert redirect_thread.is_alive() is False
        assert stop_thread.is_alive() is False
        assert agent._interrupt_requested is True
        assert agent._interrupt_message == "stop requested"
        assert agent._pending_redirect is None

    def test_codex_app_server_hard_stop_reaches_native_session(self):
        agent = _bare_agent()
        calls = []
        agent.api_mode = "codex_app_server"
        agent._codex_session = type(
            "_CodexSession",
            (),
            {"request_interrupt": lambda self: calls.append("interrupt")},
        )()

        agent.interrupt()

        assert calls == ["interrupt"]


    def test_redirect_during_tool_execution_uses_safe_steer_boundary(self):
        agent = _bare_agent()
        agent._executing_tools = True

        assert agent.redirect("also check migrations") is True
        assert agent._pending_redirect is None
        assert agent._pending_steer == "also check migrations"
        assert agent._interrupt_requested is False


class TestActiveTurnRedirectCheckpoint:
    def test_assistant_tail_puts_correction_last(self):
        from agent.conversation_loop import _apply_active_turn_redirect

        agent = _bare_agent()
        agent._current_streamed_assistant_text = "Visible draft."
        messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "committed assistant item"},
        ]

        _apply_active_turn_redirect(agent, messages, "Use Postgres instead.")

        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Use Postgres instead."
        assert sum(1 for m in messages if m["role"] == "assistant") == 1
        # Scaffolding is provider-replay text, carried in the sidecar so the
        # model still sees the interrupted context — never in the transcript.
        replayed = messages[-1]["api_content"]
        assert "Visible draft." in replayed
        assert "Context from the interrupted assistant response" in replayed
        assert replayed.endswith("Use Postgres instead.")

    def test_scaffolding_never_lands_in_transcript_content(self):
        """The checkpoint machinery is for the MODEL, not the transcript.

        Persisting ``[This response was interrupted by a user correction.]``
        into ``content`` painted raw scaffolding as an assistant bubble on
        every reload. It must ride in ``api_content`` (replayed to the
        provider) while ``content`` stays clean, or be marked
        ``display_kind="hidden"`` when there is no clean form at all.
        """
        from agent.conversation_loop import _apply_active_turn_redirect

        scaffolding = (
            "[This response was interrupted by a user correction.]",
            "Visible response before the interruption:",
            "[Context from the interrupted assistant response]",
        )

        for tail_role in ("tool", "assistant"):
            for streamed in ("Partial reply on screen.", ""):
                agent = _bare_agent()
                agent._current_streamed_assistant_text = streamed
                messages = [{"role": "user", "content": "start"}]
                if tail_role == "assistant":
                    messages.append({"role": "assistant", "content": "committed"})
                else:
                    messages.append(
                        {"role": "assistant", "tool_calls": [{"id": "a"}]}
                    )
                    messages.append(
                        {"role": "tool", "content": "out", "tool_call_id": "a"}
                    )

                _apply_active_turn_redirect(agent, messages, "New direction.")

                for msg in messages:
                    if msg.get("display_kind") == "hidden":
                        continue  # dropped by every transcript surface
                    content = str(msg.get("content", ""))
                    for marker in scaffolding:
                        assert marker not in content, (
                            f"scaffolding leaked into visible content "
                            f"(tail={tail_role}, streamed={bool(streamed)}): {content!r}"
                        )

                # The user's correction is always shown verbatim.
                assert messages[-1]["content"] == "New direction."
                # ...and the model still receives the interrupted context.
                replayed = "".join(
                    str(m.get("api_content") or m.get("content", "")) for m in messages
                )
                assert "[This response was interrupted by a user correction.]" in replayed
                if streamed:
                    assert streamed in replayed

    def test_checkpoint_never_replays_chain_of_thought(self):
        """Raw CoT serialized into checkpoint content reads to Anthropic's
        output classifier as reasoning-injection; because the checkpoint is
        persisted and replayed on every later call, one redirect during a
        thinking phase permanently bricked sessions with deterministic
        empty-response storms (July 2026). Reasoning must never appear in
        replayable content — in either the assistant-checkpoint or the
        merged-user-correction shape."""
        from agent.conversation_loop import _apply_active_turn_redirect

        for tail_role in ("user", "assistant"):
            agent = _bare_agent()
            # Simulate a surface having displayed reasoning this turn.
            agent._current_streamed_reasoning_text = "SECRET chain of thought."
            agent._current_streamed_assistant_text = "Visible draft."
            messages = [{"role": "user", "content": "start"}]
            if tail_role == "assistant":
                messages.append({"role": "assistant", "content": "committed"})

            _apply_active_turn_redirect(agent, messages, "Change course.")

            # Check BOTH the transcript content and the replayed sidecar —
            # the sidecar is what actually reaches the provider.
            serialized = "".join(
                str(m.get("content", "")) + str(m.get("api_content") or "")
                for m in messages
            )
            assert "SECRET chain of thought." not in serialized
            assert "Reasoning shown before the interruption" not in serialized
            assert "Visible draft." in serialized

    def test_checkpoint_omits_reasoning_label_when_nothing_visible(self):
        from agent.conversation_loop import _apply_active_turn_redirect

        agent = _bare_agent()
        agent._current_streamed_reasoning_text = "thinking only, no text yet"
        messages = [{"role": "user", "content": "start"}]

        _apply_active_turn_redirect(agent, messages, "New direction.")

        checkpoint_row = messages[-2]
        # Nothing was on screen, so the row exists only for the model: hidden
        # from every transcript surface, scaffolding replayed via the sidecar.
        assert checkpoint_row["display_kind"] == "hidden"
        assert (
            checkpoint_row["api_content"]
            == "[This response was interrupted by a user correction.]"
        )
        assert messages[-1]["content"] == "New direction."


class TestSteerInjection:
    def test_inserts_user_message_after_last_tool_result(self):
        agent = _bare_agent()
        agent.steer("please also check auth.log")
        messages = [
            {"role": "user", "content": "what's in /var/log?"},
            {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "content": "ls output A", "tool_call_id": "a"},
            {"role": "tool", "content": "ls output B", "tool_call_id": "b"},
        ]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=2)
        # Neither tool result's own content is touched...
        assert messages[2]["content"] == "ls output A"
        assert messages[3]["content"] == "ls output B"
        # ...instead, a genuine role:user message is INSERTED right after
        # the last tool result -- structural separation (issue #81828): a
        # model cannot fabricate a message with role:user in the API
        # request itself, unlike marker text embedded in tool content.
        assert messages[4]["role"] == "user"
        assert messages[4]["content"] == "please also check auth.log"
        assert len(messages) == 5
        # And pending_steer is consumed.
        assert agent._pending_steer is None

    def test_no_op_when_no_steer_pending(self):
        agent = _bare_agent()
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "output", "tool_call_id": "a"},
        ]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
        assert messages[-1]["content"] == "output"  # unchanged
        assert len(messages) == 2  # nothing inserted


    def test_inserted_message_has_no_marker_text(self):
        """Regression (issue #81828): the inserted steer message must be
        the user's raw text with no marker wrapper at all -- there is
        nothing for a model to learn to reproduce, since trust now comes
        from the message's role, set by the runtime, not from recognizing
        a marker string that was always visible in the system prompt."""
        agent = _bare_agent()
        agent.steer("stop after next step")
        messages = [{"role": "tool", "content": "x", "tool_call_id": "1"}]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "stop after next step"
        assert "OUT-OF-BAND" not in messages[-1]["content"]

    def test_multimodal_tool_content_left_untouched(self):
        """Anthropic-style list content on the tool message must be left
        completely alone -- the steer is now a separate message, not
        appended as a block inside the tool result's own content."""
        agent = _bare_agent()
        agent.steer("extra note")
        original_blocks = [{"type": "text", "text": "existing output"}]
        messages = [
            {"role": "tool", "content": list(original_blocks), "tool_call_id": "1"}
        ]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
        assert messages[0]["content"] == original_blocks
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "extra note"



class TestSteerThreadSafety:
    def test_concurrent_steer_calls_preserve_all_text(self):
        agent = _bare_agent()
        N = 200

        def worker(idx: int) -> None:
            agent.steer(f"note-{idx}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        text = agent._drain_pending_steer()
        assert text is not None
        # Every single note must be preserved — none dropped by the lock.
        lines = text.split("\n")
        assert len(lines) == N
        assert set(lines) == {f"note-{i}" for i in range(N)}


class TestSteerClearedOnInterrupt:
    def test_clear_interrupt_drops_pending_steer(self):
        """A hard interrupt supersedes any pending steer — the agent's
        next tool iteration won't happen, so delivering the steer later
        would be surprising."""
        agent = _bare_agent()
        # Minimal surface needed by clear_interrupt()
        agent._interrupt_requested = True
        agent._interrupt_message = None
        agent._interrupt_thread_signal_pending = False
        agent._execution_thread_id = None
        agent._tool_worker_threads = None
        agent._tool_worker_threads_lock = None

        agent.steer("will be dropped")
        agent._pending_redirect = "also drop this"
        assert agent._pending_steer == "will be dropped"

        agent.clear_interrupt()
        assert agent._pending_steer is None
        assert agent._pending_redirect is None


class TestPreApiCallSteerDrain:
    def test_pre_api_drain_restashes_when_no_tool_message(self):
        agent = _bare_agent()
        messages = [{"role": "user", "content": "hello"}]
        agent.steer("early steer")
        agent._apply_pending_steer_to_tool_results(messages, len(messages))
        assert messages == [{"role": "user", "content": "hello"}]
        assert agent._pending_steer == "early steer"


class TestSteerMarkerContract:
    def test_runtime_role_note_contains_no_delivery_exemplar(self):
        from agent.prompt_builder import STEER_CHANNEL_NOTE, STEER_MARKER_CLOSE

        assert "role: user" in STEER_CHANNEL_NOTE
        assert STEER_MARKER_OPEN not in STEER_CHANNEL_NOTE
        assert STEER_MARKER_CLOSE not in STEER_CHANNEL_NOTE
        assert "tool output" in STEER_CHANNEL_NOTE
        assert "history" in STEER_CHANNEL_NOTE
        # Old serialized histories can still be recognized without teaching
        # current models that marker text establishes user authority.
        emitted = format_steer_marker("old steer")
        assert STEER_MARKER_OPEN in emitted and STEER_MARKER_CLOSE in emitted

    def test_new_steer_does_not_move_before_a_later_assistant_message(self):
        import copy

        agent = _bare_agent()
        messages = [
            {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "output"}]},
            {"role": "assistant", "content": "response already delivered"},
        ]
        original = copy.deepcopy(messages)
        agent.steer("new direction")
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=len(messages))
        assert messages[:-1] == original
        assert messages[-1] == {"role": "user", "content": "new direction"}
        assert agent._pending_steer is None


class TestSteerCommandRegistry:
    def test_steer_in_command_registry(self):
        """The /steer slash command must be registered so it reaches all
        platforms (CLI, gateway, TUI autocomplete, Telegram/Slack menus).
        """
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("steer")
        assert cmd is not None
        assert cmd.name == "steer"
        assert cmd.category == "Session"
        assert cmd.args_hint == "<prompt>"

    def test_steer_in_bypass_set(self):
        """When the agent is running, /steer MUST bypass the Level-1
        base-adapter queue so it reaches the gateway runner's /steer
        handler. Otherwise it would be queued as user text and only
        delivered at turn end — defeating the whole point.
        """
        from hermes_cli.commands import ACTIVE_SESSION_BYPASS_COMMANDS, should_bypass_active_session

        assert "steer" in ACTIVE_SESSION_BYPASS_COMMANDS
        assert should_bypass_active_session("steer") is True


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
