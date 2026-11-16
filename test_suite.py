import os
import sys
import time
import json
import shutil
import argparse
import tempfile
import traceback
from unittest.mock import patch

# ANSI colors for readable console diagnostics
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class FakeOllamaAgent:
    """
    Deterministic mock of OllamaAgent for instant, zero-inference pipeline testing.
    Simulates realistic message histories and configurable strategy outcomes.
    """
    _counter = 0
    # Scripted outcomes across consecutive agent instantiations:
    # Match 1: CC | Match 2: CD | Match 3: DC | Match 4: DD | Match 5: [ERROR]/COOPERATE | Match 6: CC
    _scripted_choices = [
        ("COOPERATE", "Mutual trust yields stable reward."),
        ("COOPERATE", "I agree to cooperate for mutual benefit."),
        ("COOPERATE", "I trust the opponent's promise."),
        ("DEFECT", "Exploiting cooperation maximizes my payoff."),
        ("DEFECT", "I suspect betrayal, so I strike first."),
        ("COOPERATE", "Hoping for peaceful coordination."),
        ("DEFECT", "Nash equilibrium favors defection."),
        ("DEFECT", "Defensive defection against rational opponent."),
        ("[ERROR]", "Malformed XML output simulation."),
        ("COOPERATE", "Standard cooperative baseline."),
        ("COOPERATE", "Testing buffer flush boundary."),
        ("COOPERATE", "Testing buffer flush boundary.")
    ]

    def __init__(self, game, model_name: str = "mock-model"):
        idx = FakeOllamaAgent._counter
        FakeOllamaAgent._counter += 1
        self.id = f"mock_{idx + 100}"
        self.game = game
        self.model_name = model_name
        self.choice, self.explanation = FakeOllamaAgent._scripted_choices[
            idx % len(FakeOllamaAgent._scripted_choices)
        ]
        self.messages = [
            {"role": "system", "content": f"System prompt for {game.name} ({model_name})."}
        ]

    def listen(self, message: str, user: str = "AGENT"):
        prefix = f"[{user}]: " if user else ""
        self.messages.append({"role": "user", "content": f"{prefix}{message}"})

    def respond(self) -> str:
        reply = f"I propose we play {self.choice.lower()} to optimize our outcome."
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def get_strategy(self) -> tuple[str, str]:
        self.messages.append({
            "role": "user",
            "content": "[JUDGE]: Now the other agent can't listen to you. State your final move."
        })
        return self.choice, self.explanation


class IsolatedTestRunner:
    def __init__(self, run_live: bool = False):
        self.run_live = run_live
        self.project_root = os.getcwd()
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def _log_status(self, status: str, name: str, elapsed_ms: float, detail: str = ""):
        if status == "PASS":
            self.passed += 1
            badge = f"{GREEN}[PASS]{RESET}"
        elif status == "FAIL":
            self.failed += 1
            badge = f"{RED}[FAIL]{RESET}"
        elif status == "WARN":
            self.warnings += 1
            badge = f"{YELLOW}[WARN]{RESET}"
        else:
            badge = f"{CYAN}[SKIP]{RESET}"

        print(f"  {badge} {BOLD}{name}{RESET} {DIM}({elapsed_ms:.1f} ms){RESET}")
        if detail:
            for line in detail.strip().splitlines():
                print(f"         {line}")

    def _run_in_temp_workspace(self, test_fn):
        """
        Execute a test function inside an isolated temporary directory,
        copying 'data/games.json' so production files are never touched.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_data = os.path.join(self.project_root, "data")
            if os.path.exists(src_data):
                shutil.copytree(src_data, os.path.join(tmp_dir, "data"))

            old_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                return test_fn(tmp_dir)
            finally:
                os.chdir(old_cwd)

    # --- INDIVIDUAL ISOLATED TESTS ---

    def test_01_imports_and_environment(self):
        t0 = time.perf_counter()
        try:
            import pandas
            import numpy
            import matplotlib
            import seaborn
            import scipy
            import game_env
            import llm_agent
            import orchestrator
            import intel

            games_path = os.path.join(self.project_root, "data", "games.json")
            assert os.path.isfile(games_path), f"Missing game config file: {games_path}"

            # Check critical attribute compatibility between OllamaAgent and Orchestrator
            sample_game = game_env.GameMatrix.load_from_json(games_path, "prisoners_dilemma")
            dummy_agent = llm_agent.OllamaAgent(sample_game, model_name="gemma4")
            assert hasattr(dummy_agent, "id"), (
                "OllamaAgent is missing '.id' attribute expected by Orchestrator (check if it is named '.agent_id')."
            )

            self._log_status("PASS", "Environment, Dependencies & Schema Contract", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            self._log_status(
                "FAIL",
                "Environment, Dependencies & Schema Contract",
                (time.perf_counter() - t0) * 1000,
                f"{RED}{type(e).__name__}: {e}{RESET}"
            )

    def test_02_game_matrix_resolution(self):
        t0 = time.perf_counter()
        try:
            from game_env import GameMatrix
            games_path = os.path.join(self.project_root, "data", "games.json")
            game = GameMatrix.load_from_json(games_path, "prisoners_dilemma")

            p_cc = game.resolve_turn("COOPERATE", "COOPERATE")
            p_cd = game.resolve_turn("COOPERATE", "DEFECT")
            p_dc = game.resolve_turn("DEFECT", "COOPERATE")
            p_dd = game.resolve_turn("DEFECT", "DEFECT")

            assert isinstance(p_cc, tuple) and len(p_cc) == 2, "resolve_turn must return a 2-tuple."
            assert p_dc[0] > p_cc[0] > p_dd[0] > p_cd[0], (
                f"Payoff ordering violates Prisoner's Dilemma inequalities: CC={p_cc}, CD={p_cd}, DC={p_dc}, DD={p_dd}"
            )

            self._log_status("PASS", "GameMatrix Payoff & Rules Verification", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            self._log_status(
                "FAIL",
                "GameMatrix Payoff & Rules Verification",
                (time.perf_counter() - t0) * 1000,
                f"{RED}{type(e).__name__}: {e}{RESET}"
            )

    def test_03_orchestrator_lifecycle_and_buffer(self):
        t0 = time.perf_counter()
        try:
            import orchestrator as orch_mod

            def _logic(tmp_dir):
                FakeOllamaAgent._counter = 0
                with patch.object(orch_mod, "OllamaAgent", FakeOllamaAgent):
                    # Silence stdout prints from Orchestrator.result() during automated test
                    with patch("builtins.print"):
                        orch = orch_mod.Orchestrator(max_turns=2, randomize_start=False)
                        # Run 6 games to test MAX_BUFFER_SIZE=5 auto-flush + 1 residual item
                        for _ in range(6):
                            orch.run()
                            orch.reset()
                        orch.save_response_history()

                history_file = os.path.join(tmp_dir, "data/response_history.json")
                assert os.path.isfile(history_file), "data/response_history.json was not created."

                with open(history_file, "r", encoding="utf-8") as f:
                    records = json.load(f)

                assert len(records) == 6, f"Expected 6 saved games after buffer flush, found {len(records)}."
                for rec in records:
                    for key in ("time", "game", "agentA", "agentB", "results", "first_speaker_id", "full_conversation"):
                        assert key in rec, f"Missing required key '{key}' in saved JSON record."

            self._run_in_temp_workspace(_logic)
            self._log_status("PASS", "Orchestrator Loop, Reset & Buffer Flush (Mocked)", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            self._log_status(
                "FAIL",
                "Orchestrator Loop, Reset & Buffer Flush (Mocked)",
                (time.perf_counter() - t0) * 1000,
                f"{RED}{type(e).__name__}: {e}{RESET}\n{DIM}{traceback.format_exc(limit=2)}{RESET}"
            )

    def test_04_intel_pipeline_and_artifacts(self):
        t0 = time.perf_counter()
        try:
            import orchestrator as orch_mod
            import intel

            def _logic(tmp_dir):
                FakeOllamaAgent._counter = 0
                with patch.object(orch_mod, "OllamaAgent", FakeOllamaAgent):
                    with patch("builtins.print"):
                        orch = orch_mod.Orchestrator(max_turns=2, randomize_start=True)
                        for _ in range(6):
                            orch.run()
                            orch.reset()
                        orch.save_response_history()

                ds = intel.DataSet("data/response_history.json")
                assert len(ds.df) == 6, "DataSet row count mismatch."
                assert len(ds.valid_df) == 5, f"Expected 5 valid games (1 '[ERROR]' filtered), got {len(ds.valid_df)}."

                stats_engine = intel.StatsEngine(ds)
                results = stats_engine.run_all_analyses(export_dir="reports", verbose=False)
                assert "chi_square_tests" in results, "Chi-square test missing from StatsEngine output."

                for csv_name in ("action_rates.csv", "outcome_matrix.csv", "payoff_stats.csv", "statistical_tests.json"):
                    assert os.path.isfile(os.path.join("reports", csv_name)), f"Missing stats artifact: {csv_name}"

                graphs = intel.Graphs(ds, output_dir="reports")
                with patch("builtins.print"):
                    graphs.generate_all_graphs()

                for png_name in (
                    "outcome_matrix_heatmap.png",
                    "payoff_distribution.png",
                    "verbosity_vs_strategy.png",
                    "first_mover_impact.png"
                ):
                    png_path = os.path.join("reports", png_name)
                    assert os.path.isfile(png_path) and os.path.getsize(png_path) > 1024, f"Invalid or missing plot: {png_name}"

                if hasattr(intel, "ChatReport"):
                    report = intel.ChatReport(ds, output_dir="reports")
                    with patch("builtins.print"):
                        html_path = report.generate_html_report()
                    assert os.path.isfile(html_path) and os.path.getsize(html_path) > 1024, "HTML report missing or empty."
                else:
                    return "NO_CHAT_REPORT"

            res = self._run_in_temp_workspace(_logic)
            if res == "NO_CHAT_REPORT":
                self._log_status(
                    "WARN",
                    "Intel Pipeline (Stats, Graphs & HTML)",
                    (time.perf_counter() - t0) * 1000,
                    f"{YELLOW}StatsEngine and Graphs passed, but 'ChatReport' class was not found in intel.py.{RESET}"
                )
            else:
                self._log_status("PASS", "Intel Pipeline (Stats, Graphs & HTML)", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            self._log_status(
                "FAIL",
                "Intel Pipeline (Stats, Graphs & HTML)",
                (time.perf_counter() - t0) * 1000,
                f"{RED}{type(e).__name__}: {e}{RESET}\n{DIM}{traceback.format_exc(limit=2)}{RESET}"
            )

    def test_05_statistical_edge_cases(self):
        t0 = time.perf_counter()
        try:
            import intel

            def _logic(tmp_dir):
                # Case A: Zero variance (100% mutual cooperation — common in unprompted LLMs)
                coop_records = [
                    {
                        "time": "2026-11-16 12:00:00",
                        "game": "prisoners_dilemma",
                        "agentA": {"id": "a1", "answer": "COOPERATE", "explanation": "Peace."},
                        "agentB": {"id": "b1", "answer": "COOPERATE", "explanation": "Peace."},
                        "results": {"payoffA": 3, "payoffB": 3},
                        "first_speaker_id": "a1",
                        "full_conversation": [
                            {"role": "assistant", "content": "Let us cooperate."},
                            {"role": "user", "content": "[AGENT]: Agreed."}
                        ]
                    }
                    for _ in range(4)
                ]
                with open("zero_var.json", "w", encoding="utf-8") as f:
                    json.dump(coop_records, f)

                ds_zero = intel.DataSet("zero_var.json")
                se_zero = intel.StatsEngine(ds_zero)
                res_zero = se_zero.run_all_analyses(export_dir="reports_zero", verbose=False)
                assert "note" in res_zero["chi_square_tests"]["role_vs_action"], "Zero-variance guard failed."

                gr_zero = intel.Graphs(ds_zero, output_dir="reports_zero")
                with patch("builtins.print"):
                    gr_zero.generate_all_graphs()

                # Case B: 100% parsing errors (must raise controlled RuntimeError, not KeyError/ZeroDivisionError)
                err_records = [
                    {
                        "time": "2026-11-16 12:00:00",
                        "game": "prisoners_dilemma",
                        "agentA": {"id": "a1", "answer": "[ERROR]", "explanation": "Parse fail."},
                        "agentB": {"id": "b1", "answer": "[ERROR]", "explanation": "Parse fail."},
                        "results": {"payoffA": 0, "payoffB": 0},
                        "first_speaker_id": "a1",
                        "full_conversation": []
                    }
                ]
                with open("all_err.json", "w", encoding="utf-8") as f:
                    json.dump(err_records, f)

                ds_err = intel.DataSet("all_err.json")
                se_err = intel.StatsEngine(ds_err)
                try:
                    se_err.run_all_analyses(export_dir="reports_err", verbose=False)
                    raise AssertionError("Expected RuntimeError when 100% of matches have parsing errors.")
                except RuntimeError:
                    pass  # Expected controlled failure

            self._run_in_temp_workspace(_logic)
            self._log_status("PASS", "Edge Cases (Zero-Variance & 100% Parse Errors)", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            self._log_status(
                "FAIL",
                "Edge Cases (Zero-Variance & 100% Parse Errors)",
                (time.perf_counter() - t0) * 1000,
                f"{RED}{type(e).__name__}: {e}{RESET}\n{DIM}{traceback.format_exc(limit=2)}{RESET}"
            )

    def test_06_live_ollama_smoke_test(self):
        if not self.run_live:
            self._log_status(
                "SKIP",
                "Live Ollama Inference Smoke Test",
                0.0,
                f"{CYAN}Skipped by default. Run with '--live' to test a real 1-turn Ollama match.{RESET}"
            )
            return

        t0 = time.perf_counter()
        try:
            from orchestrator import Orchestrator

            def _logic(tmp_dir):
                with patch("builtins.print"):
                    orch = Orchestrator(max_turns=1, model_name="gemma4")
                    orch.run()
                    orch.save_response_history()

                with open("data/response_history.json", "r", encoding="utf-8") as f:
                    data = json.load(f)

                rec = data[0]
                act_a = rec["agentA"]["answer"]
                act_b = rec["agentB"]["answer"]
                assert act_a in ("COOPERATE", "DEFECT", "[ERROR]"), f"Unexpected action A: {act_a}"
                assert act_b in ("COOPERATE", "DEFECT", "[ERROR]"), f"Unexpected action B: {act_b}"
                return act_a, act_b

            act_a, act_b = self._run_in_temp_workspace(_logic)
            self._log_status(
                "PASS",
                "Live Ollama Inference Smoke Test",
                (time.perf_counter() - t0) * 1000,
                f"Real match completed -> Agent A: {BOLD}{act_a}{RESET}, Agent B: {BOLD}{act_b}{RESET}"
            )
        except Exception as e:
            self._log_status(
                "FAIL",
                "Live Ollama Inference Smoke Test",
                (time.perf_counter() - t0) * 1000,
                f"{RED}{type(e).__name__}: {e}{RESET}\n"
                f"{YELLOW}Hint: Ensure Ollama service is running and 'gemma4' is pulled locally.{RESET}"
            )

    def run_all(self):
        print(f"\n{BOLD}{'=' * 64}")
        print("  SIMULATION FRAMEWORK - AUTOMATED DIAGNOSTIC SUITE")
        print(f"{'=' * 64}{RESET}\n")

        start_total = time.perf_counter()
        self.test_01_imports_and_environment()
        self.test_02_game_matrix_resolution()
        self.test_03_orchestrator_lifecycle_and_buffer()
        self.test_04_intel_pipeline_and_artifacts()
        self.test_05_statistical_edge_cases()
        self.test_06_live_ollama_smoke_test()

        total_ms = (time.perf_counter() - start_total) * 1000
        print(f"\n{BOLD}{'-' * 64}{RESET}")
        summary_color = GREEN if self.failed == 0 else RED
        print(
            f"  {BOLD}SUMMARY:{RESET} "
            f"{GREEN}{self.passed} passed{RESET} | "
            f"{summary_color}{self.failed} failed{RESET} | "
            f"{YELLOW}{self.warnings} warnings{RESET} "
            f"{DIM}(Total time: {total_ms:.1f} ms){RESET}"
        )
        print(f"{BOLD}{'=' * 64}{RESET}\n")
        sys.exit(1 if self.failed > 0 else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run isolated diagnostic tests for the simulation framework.")
    parser.add_argument("--live", action="store_true", help="Include a real 1-turn match using local Ollama.")
    args = parser.parse_args()

    runner = IsolatedTestRunner(run_live=args.live)
    runner.run_all()
