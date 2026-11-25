from matplotlib.colors import LinearSegmentedColormap
from matplotlib.container import BarContainer
import os, json, html, re, argparse, ollama
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from string import Template


def _classify_message(msg: dict, id_a: str = "", id_b: str = "", has_agent_spoken: bool = False) -> tuple[str, str]:
    """
    Classify a raw conversation message into ('system' | 'judge_open' | 'judge_close' | 'agent_a' | 'agent_b' | 'skip', cleaned_text).
    Supports both Agent-A-perspective histories ('assistant' vs 'user' with/without '[AGENT]:' or '[ID]:')
    and explicit speaker metadata ('sender', 'agent_id', 'speaker_id', 'name').
    """
    role = str(msg.get("role", "")).strip().lower()
    raw_content = str(msg.get("content", ""))
    stripped = raw_content.strip()

    if role == "system":
        clean_sys = re.sub(r"^\[SYSTEM\]\s*:?\s*", "", stripped, count=1, flags=re.IGNORECASE).strip()
        return "system", clean_sys

    if role == "judge" or re.match(r"^\[(?:SYSTEM_)?JUDGE\]", stripped, flags=re.IGNORECASE):
        clean_judge = re.sub(r"^\[(?:SYSTEM_)?JUDGE\]\s*:?\s*", "", stripped, count=1, flags=re.IGNORECASE).strip()
        if has_agent_spoken:
            return "judge_close", clean_judge
        return "judge_open", clean_judge

    # Check explicit sender/agent identifiers if present in the JSON record
    speaker_meta = str(
        msg.get("sender")
        or msg.get("agent_id")
        or msg.get("speaker_id")
        or msg.get("name")
        or ""
    ).strip()

    # Extract optional leading bracketed tag such as [4140]: or [AGENT]:
    prefix_match = re.match(r"^\[([^\]]+)\]\s*:?\s*", stripped)
    prefix_tag = prefix_match.group(1).strip() if prefix_match else ""

    clean_text = re.sub(
        r"^\[(?:AGENT|AGENT\s*[AB]|PLAYER\s*[AB]|\d+)\]\s*:?\s*",
        "",
        stripped,
        count=1,
        flags=re.IGNORECASE,
    ).strip()

    effective_speaker = speaker_meta or prefix_tag
    if effective_speaker:
        meta_upper = effective_speaker.upper()
        if (id_a and effective_speaker == str(id_a)) or meta_upper in {"A", "AGENT A", "AGENT_A", "PLAYER A", "PLAYER_A"}:
            return "agent_a", clean_text
        if (id_b and effective_speaker == str(id_b)) or meta_upper in {"B", "AGENT B", "AGENT_B", "PLAYER B", "PLAYER_B"}:
            return "agent_b", clean_text

    # Standard Agent-A-centric role mapping
    if role == "assistant":
        return "agent_a", clean_text
    if role == "user":
        return "agent_b", clean_text

    return "skip", clean_text


def _extract_diplomatic_messages(
    messages: list, id_a: str = "", id_b: str = "", max_exchanges: int = 5
) -> tuple[str, str, list[dict], int]:
    """
    Extract and normalize the diplomatic conversation history before the closing [JUDGE] prompt.
    Trims any unreciprocated trailing message beyond `max_exchanges` full Agent A / Agent B exchanges
    (resolving the 10 vs 11 messages off-by-one asymmetry) and assigns an exchange-based turn index
    (1 exchange = 1 pair of replies -> max `max_exchanges` turns, or fewer on early termination).
    """
    system_prompt = ""
    judge_open = ""
    agent_msgs = []
    has_agent_spoken = False

    for msg in messages:
        kind, clean_text = _classify_message(
            msg, id_a=str(id_a), id_b=str(id_b), has_agent_spoken=has_agent_spoken
        )
        if kind == "system":
            system_prompt = clean_text
        elif kind == "judge_open":
            judge_open = clean_text
        elif kind == "judge_close":
            break
        elif kind in {"agent_a", "agent_b"}:
            has_agent_spoken = True
            agent_msgs.append({"speaker": kind, "text": clean_text})

    # Cap at `max_exchanges * 2` individual messages so 10- and 11-message logs both yield 5 exchanges
    max_messages = max_exchanges * 2
    if len(agent_msgs) > max_messages:
        agent_msgs = agent_msgs[:max_messages]

    # Count turns per A/B exchange pair rather than per individual reply
    for idx, item in enumerate(agent_msgs):
        item["turn"] = (idx // 2) + 1

    total_turns = (len(agent_msgs) + 1) // 2
    return system_prompt, judge_open, agent_msgs, total_turns


class DataSet:
    def __init__(self, file_path: str, max_exchanges: int = 5):
        self.file_path = file_path
        self.max_exchanges = max_exchanges
        self.raw_records = []
        self._raw_turn_records = []
        self.df = self.load_data()
        self.error_rate = round(float(self.df["has_error"].mean() * 100), 2)
        self.valid_df = self._get_valid_df()
        self.long_df = self._build_long_agent_df(self.valid_df) if not self.valid_df.empty else pd.DataFrame()
        self.turn_df = self._build_turn_df(self.valid_df) if not self.valid_df.empty else pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """
        Load raw JSON data and flatten it into a tabular DataFrame
        enriched with derived metrics per match and per individual agent.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.raw_records = json.load(f)
        except Exception as e:
            raise ValueError(f"Error in the loading of data from {self.file_path}: {e}")

        if not self.raw_records:
            raise ValueError(f"The file {self.file_path} is empty.")

        rows = []
        self._raw_turn_records = []
        for idx, rec in enumerate(self.raw_records):
            match_id = idx + 1
            act_a = rec["agentA"]["answer"]
            act_b = rec["agentB"]["answer"]
            id_a = rec["agentA"]["id"]
            id_b = rec["agentB"]["id"]
            first_id = rec.get("first_speaker_id")

            has_error = (act_a == "[ERROR]") or (act_b == "[ERROR]")
            joint_outcome = "ERROR" if has_error else f"{act_a} / {act_b}"

            # Compute diplomatic verbosity and normalized exchange turns (excluding system prompt and JUDGE messages)
            words_a, words_b, msgs_a, msgs_b, turns = self._extract_diplomatic_metrics(
                rec.get("full_conversation", []),
                id_a=str(id_a),
                id_b=str(id_b),
                max_exchanges=self.max_exchanges,
            )

            # Extract [DONE] token occurrences, termination timing, and per-turn message lengths
            done_a, done_b, first_done_turn, msg_turns = self._extract_done_and_turn_details(
                rec.get("full_conversation", []),
                id_a=str(id_a),
                id_b=str(id_b),
                max_exchanges=self.max_exchanges,
            )

            for item in msg_turns:
                self._raw_turn_records.append({
                    "match_id": match_id,
                    "turn": item["turn"],
                    "role": "Agent A" if item["speaker"] == "agent_a" else "Agent B",
                    "action": act_a if item["speaker"] == "agent_a" else act_b,
                    "joint_outcome": joint_outcome,
                    "words": item["words"],
                    "done_count": item["done_count"],
                    "has_done": item["done_count"] > 0,
                    "has_error": has_error,
                })

            total_words = words_a + words_b
            total_done = done_a + done_b
            has_done_a = done_a > 0
            has_done_b = done_b > 0
            has_done = total_done > 0

            # Classify how the diplomatic negotiation ended
            if has_done_a and has_done_b:
                termination_type = "Mutual [DONE]"
            elif has_done_a or has_done_b:
                termination_type = "Unilateral [DONE]"
            else:
                termination_type = "Max Turns (No [DONE])"

            retries_a = int(rec.get("agentA", {}).get("sanity_retries", 0))
            retries_b = int(rec.get("agentB", {}).get("sanity_retries", 0))
            fallbacks_a = int(rec.get("agentA", {}).get("hardcoded_fallbacks", 0))
            fallbacks_b = int(rec.get("agentB", {}).get("hardcoded_fallbacks", 0))
            reasons_a = rec.get("agentA", {}).get("sanity_fail_reasons", {}) or {}
            reasons_b = rec.get("agentB", {}).get("sanity_fail_reasons", {}) or {}
            
            rows.append({
                "match_id": match_id,
                "time": rec.get("time"),
                "game": rec.get("game"),
                "agentA_id": id_a,
                "agentB_id": id_b,
                "action_A": act_a,
                "action_B": act_b,
                "joint_outcome": joint_outcome,
                "payoff_A": rec["results"]["payoffA"],
                "payoff_B": rec["results"]["payoffB"],
                "expl_A": rec["agentA"]["explanation"],
                "expl_B": rec["agentB"]["explanation"],
                "first_speaker": "Agent A" if first_id == id_a else ("Agent B" if first_id == id_b else "Unknown"),
                "turns": turns,
                "msgs_A": msgs_a,
                "msgs_B": msgs_b,
                "words_A": words_a,
                "words_B": words_b,
                "words_per_turn_A": round(words_a / max(msgs_a, 1), 2),
                "words_per_turn_B": round(words_b / max(msgs_b, 1), 2),
                "total_words": total_words,
                "avg_exchange_words": round(total_words / max(turns, 1), 2),
                "done_count_A": done_a,
                "done_count_B": done_b,
                "total_done": total_done,
                "has_done_A": has_done_a,
                "has_done_B": has_done_b,
                "has_done": has_done,
                "first_done_turn": first_done_turn if first_done_turn is not None else np.nan,
                "termination_type": termination_type,
                "done_per_100_words": round((total_done / total_words) * 100, 3) if total_words > 0 else 0.0,
                "done_per_100_words_A": round((done_a / words_a) * 100, 3) if words_a > 0 else 0.0,
                "done_per_100_words_B": round((done_b / words_b) * 100, 3) if words_b > 0 else 0.0,
                "has_error": has_error,
                "sanity_retries_A": retries_a,
                "sanity_retries_B": retries_b,
                "total_sanity_retries": retries_a + retries_b,
                "hardcoded_fallbacks_A": fallbacks_a,
                "hardcoded_fallbacks_B": fallbacks_b,
                "total_hardcoded_fallbacks": fallbacks_a + fallbacks_b,
                "has_hardcoded_fallback": (fallbacks_a + fallbacks_b) > 0,
                "sanity_fail_reasons_A": reasons_a,
                "sanity_fail_reasons_B": reasons_b,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _extract_diplomatic_metrics(
        messages: list, id_a: str = "", id_b: str = "", max_exchanges: int = 5
    ) -> tuple[int, int, int, int, int]:
        """
        Count the words exchanged by both agents, individual message counts, and the total
        number of A/B exchanges during the balanced diplomatic phase only, stopping before the closing [JUDGE] intervention.
        """
        _, _, agent_msgs, total_turns = _extract_diplomatic_messages(
            messages, id_a=id_a, id_b=id_b, max_exchanges=max_exchanges
        )
        words_a = 0
        words_b = 0
        msgs_a = 0
        msgs_b = 0
        for item in agent_msgs:
            clean_words = re.sub(r"\[DONE\]", "", item["text"], flags=re.IGNORECASE).split()
            if item["speaker"] == "agent_a":
                words_a += len(clean_words)
                msgs_a += 1
            elif item["speaker"] == "agent_b":
                words_b += len(clean_words)
                msgs_b += 1
        return words_a, words_b, msgs_a, msgs_b, total_turns

    @staticmethod
    def _extract_done_and_turn_details(
        messages: list, id_a: str = "", id_b: str = "", max_exchanges: int = 5
    ) -> tuple[int, int, int | None, list[dict]]:
        """
        Extract per-agent [DONE] counts, the first exchange turn where [DONE] appears,
        and per-message word counts across the balanced diplomatic phase.
        """
        _, _, agent_msgs, _ = _extract_diplomatic_messages(
            messages, id_a=id_a, id_b=id_b, max_exchanges=max_exchanges
        )
        done_a = 0
        done_b = 0
        first_done_turn = None
        msg_turns = []

        for item in agent_msgs:
            done_matches = re.findall(r"\[DONE\]", item["text"], flags=re.IGNORECASE)
            d_cnt = len(done_matches)
            clean_words = re.sub(r"\[DONE\]", "", item["text"], flags=re.IGNORECASE).split()
            if d_cnt > 0 and first_done_turn is None:
                first_done_turn = item["turn"]
            if item["speaker"] == "agent_a":
                done_a += d_cnt
            elif item["speaker"] == "agent_b":
                done_b += d_cnt
            msg_turns.append({
                "turn": item["turn"],
                "speaker": item["speaker"],
                "words": len(clean_words),
                "done_count": d_cnt,
            })

        return done_a, done_b, first_done_turn, msg_turns

    def _get_valid_df(self) -> pd.DataFrame:
        """Return only matches without parsing errors."""
        return self.df[~self.df["has_error"]].copy()

    def _build_long_agent_df(self, valid_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame from match-level to agent-level (2 rows per match)
        to facilitate aggregated statistical tests.
        """
        done_role_a = np.where(
            valid_df["has_done_A"],
            "Sent [DONE]",
            np.where(valid_df["has_done_B"], "Partner Sent [DONE]", "No [DONE] (Max Turns)")
        )
        done_role_b = np.where(
            valid_df["has_done_B"],
            "Sent [DONE]",
            np.where(valid_df["has_done_A"], "Partner Sent [DONE]", "No [DONE] (Max Turns)")
        )

        df_a = pd.DataFrame({
            "match_id": valid_df["match_id"],
            "role": "Agent A",
            "action": valid_df["action_A"],
            "payoff": valid_df["payoff_A"],
            "words": valid_df["words_A"],
            "words_per_turn": valid_df["words_per_turn_A"],
            "turns": valid_df["turns"],
            "is_initiator": valid_df["first_speaker"] == "Agent A",
            "done_count": valid_df["done_count_A"],
            "used_done": valid_df["has_done_A"],
            "partner_used_done": valid_df["has_done_B"],
            "match_has_done": valid_df["has_done"],
            "termination_type": valid_df["termination_type"],
            "done_role": done_role_a,
            "done_per_100_words": valid_df["done_per_100_words_A"],
            "first_done_turn": valid_df["first_done_turn"],
            "sanity_retries": valid_df["sanity_retries_A"],
            "hardcoded_fallbacks": valid_df["hardcoded_fallbacks_A"],
            "has_hardcoded_fallback": valid_df["hardcoded_fallbacks_A"] > 0,
            "sanity_tier": np.where(
                valid_df["hardcoded_fallbacks_A"] > 0,
                "Hardcoded Fallback",
                np.where(valid_df["sanity_retries_A"] > 0, "Corrected (1-10 Retries)", "Clean (0 Retries)"),
            ),
        })
        df_b = pd.DataFrame({
            "match_id": valid_df["match_id"],
            "role": "Agent B",
            "action": valid_df["action_B"],
            "payoff": valid_df["payoff_B"],
            "words": valid_df["words_B"],
            "words_per_turn": valid_df["words_per_turn_B"],
            "turns": valid_df["turns"],
            "is_initiator": valid_df["first_speaker"] == "Agent B",
            "done_count": valid_df["done_count_B"],
            "used_done": valid_df["has_done_B"],
            "partner_used_done": valid_df["has_done_A"],
            "match_has_done": valid_df["has_done"],
            "termination_type": valid_df["termination_type"],
            "done_role": done_role_b,
            "done_per_100_words": valid_df["done_per_100_words_B"],
            "first_done_turn": valid_df["first_done_turn"],
            "sanity_retries": valid_df["sanity_retries_B"],
            "hardcoded_fallbacks": valid_df["hardcoded_fallbacks_B"],
            "has_hardcoded_fallback": valid_df["hardcoded_fallbacks_B"] > 0,
            "sanity_tier": np.where(
                valid_df["hardcoded_fallbacks_B"] > 0,
                "Hardcoded Fallback",
                np.where(valid_df["sanity_retries_B"] > 0, "Corrected (1-10 Retries)", "Clean (0 Retries)"),
            ),
        })
        return pd.concat([df_a, df_b], ignore_index=True)

    def _build_turn_df(self, valid_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a message-level DataFrame (one row per diplomatic message in valid matches)
        enriched with the match termination category for turn-by-turn variance analysis.
        """
        if not self._raw_turn_records:
            return pd.DataFrame()
        raw_turns = pd.DataFrame(self._raw_turn_records)
        valid_turns = raw_turns[~raw_turns["has_error"]].copy()
        term_map = valid_df.set_index("match_id")["termination_type"]
        valid_turns["termination_type"] = valid_turns["match_id"].map(term_map)
        return valid_turns


class StatsEngine:
    def __init__(self, dataset: DataSet):
        self.dataset = dataset
        self.results = {}

    def _compute_action_rates(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """Compute percentage action rates globally and broken down by role."""
        summary = pd.crosstab(long_df["role"], long_df["action"], normalize="index") * 100
        global_rates = (long_df["action"].value_counts(normalize=True) * 100).rename("Global")
        summary = pd.concat([summary, global_rates.to_frame().T]).fillna(0.0)
        return summary.round(2)

    def _compute_outcome_matrix(self, valid_df: pd.DataFrame) -> pd.DataFrame:
        """Build the percentage contingency matrix (Agent A x Agent B)."""
        matrix = pd.crosstab(valid_df["action_A"], valid_df["action_B"], normalize="all") * 100
        return matrix.round(2)

    def _compute_payoff_stats(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean, standard deviation, median, and sum of payoffs by action and role."""
        return long_df.groupby(["role", "action"])["payoff"].agg(
            count="count",
            mean="mean",
            std="std",
            median="median",
            total="sum"
        ).round(3)

    def _compute_first_mover_stats(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """Compare choices between the agent opening the conversation (Initiator) and the one responding (Responder)."""
        temp = long_df.copy()
        temp["turn_role"] = np.where(temp["is_initiator"], "Initiator", "Responder")
        return (pd.crosstab(temp["turn_role"], temp["action"], normalize="index") * 100).round(2)

    def _compute_termination_stats(self, valid_df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute match counts, share of matches, strategy rates, exchange length statistics,
        and [DONE] word-proportional density across negotiation termination types.
        """
        match_agg = valid_df.groupby("termination_type").agg(
            matches=("match_id", "count"),
            mean_turns=("turns", "mean"),
            std_turns=("turns", "std"),
            mean_total_words=("total_words", "mean"),
            mean_done_per_100_words=("done_per_100_words", "mean"),
        )
        match_agg["match_share_pct"] = (match_agg["matches"] / len(valid_df)) * 100

        action_rates = pd.crosstab(long_df["termination_type"], long_df["action"], normalize="index") * 100
        combined = match_agg.join(action_rates).fillna(0.0)
        return combined.round(2)

    def _compute_sanity_stats(self, valid_df: pd.DataFrame, long_df: pd.DataFrame) -> dict:
        """Aggregate sanity check retry counts, hardcoded fallbacks, and failure reasons."""
        reason_totals: dict[str, int] = {}
        for col in ("sanity_fail_reasons_A", "sanity_fail_reasons_B"):
            for reasons_dict in valid_df[col]:
                if isinstance(reasons_dict, dict):
                    for k, v in reasons_dict.items():
                        reason_totals[k] = reason_totals.get(k, 0) + int(v)

        reason_df = (
            pd.DataFrame(
                sorted(reason_totals.items(), key=lambda x: x[1], reverse=True),
                columns=["reason", "count"],
            )
            if reason_totals
            else pd.DataFrame(columns=["reason", "count"])
        )
        if not reason_df.empty:
            reason_df["share_pct"] = ((reason_df["count"] / reason_df["count"].sum()) * 100).round(2)

        tier_action_pct = (pd.crosstab(long_df["sanity_tier"], long_df["action"], normalize="index") * 100).round(2)
        return {
            "reason_breakdown": reason_df,
            "strategy_by_sanity_tier_pct": tier_action_pct,
        }

    def _test_sanity_vs_action(self, long_df: pd.DataFrame) -> dict:
        """Test whether sanity retries or intervention tier correlate with chosen strategy."""
        results = {}
        actions = sorted(long_df["action"].unique())
        if len(actions) >= 2:
            g0 = long_df[long_df["action"] == actions[0]]["sanity_retries"]
            g1 = long_df[long_df["action"] == actions[1]]["sanity_retries"]
            u_stat, p_val = stats.mannwhitneyu(g0, g1, alternative="two-sided")
            results["retries_mann_whitney"] = {
                "compared_actions": f"{actions[0]} vs {actions[1]}",
                "u_statistic": round(float(u_stat), 4),
                "p_value": round(float(p_val), 4),
                "significant_005": bool(p_val < 0.05),
            }

        ct_tier = pd.crosstab(long_df["sanity_tier"], long_df["action"])
        if ct_tier.shape[0] > 1 and ct_tier.shape[1] > 1:
            chi2, p_val, dof, _ = stats.chi2_contingency(ct_tier)
            results["sanity_tier_vs_action"] = {
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p_val), 4),
                "dof": int(dof),
                "significant_005": bool(p_val < 0.05),
            }
        else:
            results["sanity_tier_vs_action"] = {"note": "Insufficient variance for Chi-square test."}
        return results

    def _test_chi_square_independence(self, long_df: pd.DataFrame) -> dict:
        """
        Perform the Chi-square test of independence on:
        1) Role (Agent A vs Agent B) vs Chosen action
        2) Speaking turn (Initiator vs Responder) vs Chosen action
        """
        results = {}

        # Test 1: Role vs Action
        ct_role = pd.crosstab(long_df["role"], long_df["action"])
        if ct_role.shape[0] > 1 and ct_role.shape[1] > 1:
            chi2, p_val, dof, _ = stats.chi2_contingency(ct_role)
            results["role_vs_action"] = {
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p_val), 4),
                "dof": int(dof),
                "significant_005": bool(p_val < 0.05)
            }
        else:
            results["role_vs_action"] = {"note": "Insufficient variance for Chi-square test."}

        # Test 2: Initiator vs Responder
        turn_role = np.where(long_df["is_initiator"], "Initiator", "Responder")
        ct_turn = pd.crosstab(turn_role, long_df["action"])
        if ct_turn.shape[0] > 1 and ct_turn.shape[1] > 1:
            chi2, p_val, dof, _ = stats.chi2_contingency(ct_turn)
            results["first_mover_vs_action"] = {
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p_val), 4),
                "dof": int(dof),
                "significant_005": bool(p_val < 0.05)
            }
        else:
            results["first_mover_vs_action"] = {"note": "Insufficient variance for Chi-square test."}

        return results

    def _test_termination_vs_action(self, long_df: pd.DataFrame) -> dict:
        """
        Test statistical association between negotiation termination type / [DONE] role
        and chosen strategy, as well as [DONE] word-density differences across strategies.
        """
        results = {}

        ct_term = pd.crosstab(long_df["termination_type"], long_df["action"])
        if ct_term.shape[0] > 1 and ct_term.shape[1] > 1:
            chi2, p_val, dof, _ = stats.chi2_contingency(ct_term)
            results["termination_type_vs_action"] = {
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p_val), 4),
                "dof": int(dof),
                "significant_005": bool(p_val < 0.05)
            }
        else:
            results["termination_type_vs_action"] = {"note": "Insufficient variance for Chi-square test."}

        ct_role = pd.crosstab(long_df["done_role"], long_df["action"])
        if ct_role.shape[0] > 1 and ct_role.shape[1] > 1:
            chi2, p_val, dof, _ = stats.chi2_contingency(ct_role)
            results["done_role_vs_action"] = {
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p_val), 4),
                "dof": int(dof),
                "significant_005": bool(p_val < 0.05)
            }
        else:
            results["done_role_vs_action"] = {"note": "Insufficient variance for Chi-square test."}

        actions = sorted(long_df["action"].unique())
        if len(actions) >= 2:
            g0 = long_df[long_df["action"] == actions[0]]["done_per_100_words"]
            g1 = long_df[long_df["action"] == actions[1]]["done_per_100_words"]
            u_stat, p_val = stats.mannwhitneyu(g0, g1, alternative="two-sided")
            results["done_density_mann_whitney"] = {
                "compared_actions": f"{actions[0]} vs {actions[1]}",
                "u_statistic": round(float(u_stat), 4),
                "p_value": round(float(p_val), 4),
                "significant_005": bool(p_val < 0.05)
            }

        return results

    def _test_symmetry_binomial(self, valid_df: pd.DataFrame) -> dict:
        """
        Exact binomial test on asymmetric outcomes (A != B) to check
        whether there is a systematic imbalance favoring one of the two agents.
        """
        asymmetric = valid_df[valid_df["action_A"] != valid_df["action_B"]]
        n_asym = len(asymmetric)
        if n_asym == 0:
            return {"note": "No asymmetric outcomes detected in the sample."}

        a_wins = int((asymmetric["payoff_A"] > asymmetric["payoff_B"]).sum())
        res = stats.binomtest(k=a_wins, n=n_asym, p=0.5, alternative="two-sided")
        return {
            "asymmetric_games": n_asym,
            "agentA_exploits_count": a_wins,
            "agentB_exploits_count": n_asym - a_wins,
            "p_value": round(float(res.pvalue), 4),
            "significant_asymmetry_005": bool(res.pvalue < 0.05)
        }

    def _test_payoff_differences(self, long_df: pd.DataFrame) -> dict:
        """
        Non-parametric Mann-Whitney U test to check whether payoffs differ
        significantly across chosen strategies.
        """
        actions = long_df["action"].unique()
        if len(actions) < 2:
            return {"note": "Only one action present: impossible to compare payoffs."}

        act_0, act_1 = actions[0], actions[1]
        group_0 = long_df[long_df["action"] == act_0]["payoff"]
        group_1 = long_df[long_df["action"] == act_1]["payoff"]

        u_stat, p_val = stats.mannwhitneyu(group_0, group_1, alternative="two-sided")
        return {
            "compared_actions": f"{act_0} vs {act_1}",
            "u_statistic": round(float(u_stat), 4),
            "p_value": round(float(p_val), 4),
            "significant_005": bool(p_val < 0.05)
        }

    def _test_verbosity_correlation(self, long_df: pd.DataFrame) -> dict:
        """
        Point-biserial correlation between normalized diplomatic verbosity (words per turn)
        and strategy choice (binary-encoded).
        """
        actions = sorted(long_df["action"].unique())
        if len(actions) < 2 or long_df["words_per_turn"].std() == 0:
            return {"note": "Insufficient variance to calculate correlation with verbosity."}

        target_action = "DEFECT" if "DEFECT" in actions else actions[1]
        binary_target = (long_df["action"] == target_action).astype(int)

        r_pb, p_val = stats.pointbiserialr(binary_target, long_df["words_per_turn"])
        return {
            "metric": "words_per_turn",
            "target_action": target_action,
            "point_biserial_r": round(float(r_pb), 4),
            "p_value": round(float(p_val), 4),
            "significant_005": bool(p_val < 0.05)
        }

    def run_all_analyses(self, export_dir: str = "reports", verbose: bool = True) -> dict:
        """
        Run all tabular analyses and private statistical tests,
        save summaries to disk, and return the complete results dictionary.
        """
        os.makedirs(export_dir, exist_ok=True)
        valid_df = self.dataset.valid_df
        long_df = self.dataset.long_df

        if valid_df.empty:
            raise RuntimeError("No valid games available for analysis (100% parsing errors).")

        self.results = {
            "meta": {
                "total_games": len(self.dataset.df),
                "valid_games": len(valid_df),
                "parsing_error_rate_pct": self.dataset.error_rate,
                "mean_exchange_turns": round(float(valid_df["turns"].mean()), 2),
                "std_exchange_turns": round(float(valid_df["turns"].std()), 2) if len(valid_df) > 1 else 0.0,
                "min_exchange_turns": int(valid_df["turns"].min()),
                "max_exchange_turns": int(valid_df["turns"].max()),
                "done_termination_rate_pct": round(float(valid_df["has_done"].mean() * 100), 2),
                "mean_done_per_100_words": round(float(valid_df["done_per_100_words"].mean()), 3),
                "total_sanity_retries": int(valid_df["total_sanity_retries"].sum()),
                "mean_sanity_retries_per_match": round(float(valid_df["total_sanity_retries"].mean()), 2),
                "retry_match_rate_pct": round(float((valid_df["total_sanity_retries"] > 0).mean() * 100), 2),
                "total_hardcoded_fallbacks": int(valid_df["total_hardcoded_fallbacks"].sum()),
                "hardcoded_fallback_match_rate_pct": round(float(valid_df["has_hardcoded_fallback"].mean() * 100), 2),
            },
            "action_rates_pct": self._compute_action_rates(long_df),
            "outcome_matrix_pct": self._compute_outcome_matrix(valid_df),
            "payoff_stats": self._compute_payoff_stats(long_df),
            "first_mover_pct": self._compute_first_mover_stats(long_df),
            "termination_stats": self._compute_termination_stats(valid_df, long_df),
            "chi_square_tests": self._test_chi_square_independence(long_df),
            "termination_independence_test": self._test_termination_vs_action(long_df),
            "symmetry_binomial_test": self._test_symmetry_binomial(valid_df),
            "payoff_mann_whitney_test": self._test_payoff_differences(long_df),
            "verbosity_correlation_test": self._test_verbosity_correlation(long_df)
        }

        self.results["action_rates_pct"].to_csv(os.path.join(export_dir, "action_rates.csv"))
        self.results["outcome_matrix_pct"].to_csv(os.path.join(export_dir, "outcome_matrix.csv"))
        self.results["payoff_stats"].to_csv(os.path.join(export_dir, "payoff_stats.csv"))
        self.results["termination_stats"].to_csv(os.path.join(export_dir, "termination_stats.csv"))
        sanity_info = self._compute_sanity_stats(valid_df, long_df)
        self.results["sanity_stats"] = sanity_info
        self.results["sanity_vs_action_test"] = self._test_sanity_vs_action(long_df)
        sanity_info["reason_breakdown"].to_csv(os.path.join(export_dir, "sanity_fail_reasons.csv"), index=False)
        sanity_info["strategy_by_sanity_tier_pct"].to_csv(os.path.join(export_dir, "strategy_by_sanity_tier.csv"))

        stats_summary = {
            "meta": self.results["meta"],
            "chi_square_tests": self.results["chi_square_tests"],
            "termination_independence_test": self.results["termination_independence_test"],
            "symmetry_binomial_test": self.results["symmetry_binomial_test"],
            "payoff_mann_whitney_test": self.results["payoff_mann_whitney_test"],
            "verbosity_correlation_test": self.results["verbosity_correlation_test"],
            "sanity_vs_action_test": self.results["sanity_vs_action_test"],
        }
        with open(os.path.join(export_dir, "statistical_tests.json"), "w", encoding="utf-8") as f:
            json.dump(stats_summary, f, indent=4)

        if verbose:
            self._print_report()

        return self.results

    def _print_report(self):
        """Print a clean summary of the results to the terminal."""
        print("\n" + "=" * 60)
        print("INTEL REPORT - STATISTICAL ANALYSIS OF THE SIMULATION")
        print("=" * 60)
        print(f"Total games: {self.results['meta']['total_games']} | "
              f"Valid: {self.results['meta']['valid_games']} | "
              f"Parsing errors: {self.results['meta']['parsing_error_rate_pct']}% | "
              f"Avg turns (exchanges): {self.results['meta']['mean_exchange_turns']}")
        print("\n--- Choice Rates (%) ---")
        print(self.results["action_rates_pct"])
        print("\n--- Joint Outcomes Matrix (% of total) ---")
        print(self.results["outcome_matrix_pct"])
        print("\n--- Payoff Statistics ---")
        print(self.results["payoff_stats"])
        print("\n--- Termination Type Statistics ---")
        print(self.results["termination_stats"])
        print("\n--- Statistical Tests ---")
        print(json.dumps({
            "chi_square": self.results["chi_square_tests"],
            "termination_independence": self.results["termination_independence_test"],
            "symmetry_binomial": self.results["symmetry_binomial_test"],
            "payoff_mann_whitney": self.results["payoff_mann_whitney_test"],
            "verbosity_correlation": self.results["verbosity_correlation_test"]
        }, indent=2))
        print("=" * 60 + "\n")

    @staticmethod
    def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
        """Compute Wilson score 95% confidence interval for a binomial proportion (in %)."""
        if n == 0:
            return 0.0, 0.0
        p = k / n
        denom = 1 + (z ** 2) / n
        center = (p + (z ** 2) / (2 * n)) / denom
        half_width = (z * np.sqrt((p * (1 - p) + (z ** 2) / (4 * n)) / n)) / denom
        return max(0.0, (center - half_width) * 100), min(100.0, (center + half_width) * 100)


class Graphs:
    def __init__(self, dataset: DataSet, output_dir: str = "reports"):
        self.dataset = dataset
        self.output_dir = output_dir
        sns.set_theme(style="whitegrid", context="notebook")
        self.primary_blue = "#31688e"
        self.primary_green = "#35b779"
        self.strategy_colors = {
            "COOPERATE": self.primary_blue,
            "DEFECT": self.primary_green,
        }

    def _get_strategy_palette(self, categories) -> dict:
        """Map strategy labels consistently to the reference blue/green palette."""
        cats = list(categories)
        fallback = sns.color_palette("viridis", max(len(cats), 2)).as_hex()
        return {
            cat: self.strategy_colors.get(str(cat).upper(), fallback[idx % len(fallback)])
            for idx, cat in enumerate(cats)
        }

    def _plot_outcome_matrix(self, valid_df: pd.DataFrame):
        """
        Generate a 2x2 matrix (heatmap) with percentages and absolute counts
        of joint outcomes (Agent A vs Agent B).
        """
        counts = pd.crosstab(valid_df["action_A"], valid_df["action_B"])
        pcts = pd.crosstab(valid_df["action_A"], valid_df["action_B"], normalize="all") * 100

        annot_matrix = np.empty(counts.shape, dtype=object)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                annot_matrix[i, j] = f"{pcts.iloc[i, j]:.1f}%\n(n={counts.iloc[i, j]})"

        blue_green_cmap = LinearSegmentedColormap.from_list(
            "viridis_bg", ["#f4f9f8", "#74c69d", self.primary_blue]
        )

        plt.figure(figsize=(6.8, 5.2))
        ax = sns.heatmap(
            pcts,
            annot=annot_matrix,
            fmt="",
            cmap=blue_green_cmap,
            cbar_kws={"label": "Valid Matches (%)"},
            vmin=0,
            vmax=100,
            linewidths=1.5,
            linecolor="white",
            annot_kws={"size": 13, "weight": "bold"}
        )
        ax.set_title("Joint Outcomes (Agent A vs Agent B)", fontsize=13.5, pad=12)
        ax.set_xlabel("Agent B Choice", fontsize=11.5)
        ax.set_ylabel("Agent A Choice", fontsize=11.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "outcome_matrix_heatmap.png"), dpi=300)
        plt.close()

    def _plot_verbosity_vs_action(self, long_df: pd.DataFrame):
        """
        Analyze the relationship between normalized diplomatic verbosity (words per turn)
        and the final choice, overlaying empirical quartile rates on the logistic curve.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
        actions = sorted(long_df["action"].unique())
        palette = self._get_strategy_palette(actions)

        sns.boxplot(
            data=long_df,
            x="action",
            y="words_per_turn",
            order=actions,
            hue="action",
            hue_order=actions,
            dodge=False,
            palette=palette,
            boxprops={"alpha": 0.85},
            ax=axes[0],
            width=0.45,
        )
        if axes[0].get_legend():
            axes[0].get_legend().remove()

        sns.stripplot(
            data=long_df,
            x="action",
            y="words_per_turn",
            order=actions,
            color="#1f2937",
            alpha=0.45,
            jitter=0.15,
            ax=axes[0],
        )
        axes[0].set_title("Verbosity per Turn by Strategy", fontsize=12)
        axes[0].set_xlabel("Strategy", fontsize=11)
        axes[0].set_ylabel("Words per Turn", fontsize=11)

        if len(actions) >= 2 and long_df["words_per_turn"].std() > 0:
            target_action = "DEFECT" if "DEFECT" in actions else actions[1]
            temp = long_df.copy()
            temp["target_bin"] = (temp["action"] == target_action).astype(int)
            try:
                sns.regplot(
                    data=temp,
                    x="words_per_turn",
                    y="target_bin",
                    logistic=True,
                    ci=95,
                    scatter_kws={"alpha": 0.35, "color": self.primary_blue, "s": 28, "label": "Observed (0/1)"},
                    line_kws={"color": self.primary_green, "lw": 2.4, "label": "Logistic fit (95% CI)"},
                    ax=axes[1],
                )
                temp["bin"] = pd.qcut(temp["words_per_turn"], q=4, duplicates="drop")
                binned = temp.groupby("bin", observed=False).agg(
                    x_mean=("words_per_turn", "mean"),
                    y_rate=("target_bin", "mean"),
                )
                axes[1].scatter(
                    binned["x_mean"],
                    binned["y_rate"],
                    color="#111827",
                    marker="D",
                    s=52,
                    zorder=5,
                    label="Quartile rate",
                )
                axes[1].set_title(f"P({target_action}) vs. Words per Turn", fontsize=12)
                axes[1].set_xlabel("Words per Turn", fontsize=11)
                axes[1].set_ylabel(f"P({target_action})", fontsize=11)
                axes[1].set_ylim(-0.05, 1.05)
                axes[1].legend(loc="upper left", fontsize=9, frameon=True)
            except RuntimeError:
                axes[1].text(0.5, 0.5, "Install 'statsmodels' for logistic curve", ha="center", va="center")
                axes[1].set_axis_off()
        else:
            axes[1].text(0.5, 0.5, "Insufficient variance for logistic curve", ha="center", va="center")
            axes[1].set_axis_off()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "verbosity_vs_strategy.png"), dpi=300)
        plt.close()

    def _plot_first_mover_comparison(self, long_df: pd.DataFrame):
        """
        Visualize the percentage rate of actions between the conversation starter (Initiator)
        and the respondent (Responder).
        """
        temp = long_df.copy()
        temp["turn_role"] = np.where(temp["is_initiator"], "Initiator (1st)", "Responder (2nd)")
        ct_pct = (pd.crosstab(temp["turn_role"], temp["action"], normalize="index") * 100).reset_index()
        melted = ct_pct.melt(id_vars="turn_role", var_name="action", value_name="percentage")
        palette = self._get_strategy_palette(melted["action"].unique())

        plt.figure(figsize=(7.2, 4.9))
        ax = sns.barplot(data=melted, x="turn_role", y="percentage", hue="action", palette=palette)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=10)

        ax.set_title("Strategy Choice by Speaking Order", fontsize=13.5, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("Strategy Share (%)", fontsize=11.5)
        ax.set_ylim(0, 112)
        plt.legend(title="Strategy")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "first_mover_impact.png"), dpi=300)
        plt.close()

    def _plot_done_proportion_vs_words(self, valid_df: pd.DataFrame, long_df: pd.DataFrame):
        """
        Visualize the distribution of [DONE] density per 100 spoken words
        across final chosen strategies.
        """
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        actions = sorted(long_df["action"].unique())
        palette = self._get_strategy_palette(actions)

        sns.boxplot(
            data=long_df,
            x="action",
            y="done_per_100_words",
            order=actions,
            hue="action",
            hue_order=actions,
            dodge=False,
            palette=palette,
            boxprops={"alpha": 0.85},
            width=0.45,
            ax=ax,
        )
        if ax.get_legend():
            ax.get_legend().remove()

        sns.stripplot(
            data=long_df,
            x="action",
            y="done_per_100_words",
            order=actions,
            color="#1f2937",
            alpha=0.45,
            jitter=0.15,
            ax=ax,
        )
        ax.set_title("[DONE] Density by Strategy", fontsize=12.5)
        ax.set_xlabel("Strategy", fontsize=11)
        ax.set_ylabel("[DONE] per 100 Words", fontsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "done_proportion_vs_words.png"), dpi=300)
        plt.close()

    def _plot_termination_vs_cooperation(self, valid_df: pd.DataFrame, long_df: pd.DataFrame):
        """
        Analyze the relationship between negotiation termination mode and the choice to cooperate:
        - Left panel: agent strategy rates across match termination categories.
        - Right panel: agent strategy rates depending on whether the agent or partner emitted [DONE].
        """
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
        palette = self._get_strategy_palette(long_df["action"].unique())

        term_order = [
            t for t in ["Mutual [DONE]", "Unilateral [DONE]", "Max Turns (No [DONE])"]
            if t in long_df["termination_type"].unique()
        ]
        ct_term = (pd.crosstab(long_df["termination_type"], long_df["action"], normalize="index") * 100).reindex(term_order)
        term_counts = valid_df["termination_type"].value_counts()
        ct_term.index = [f"{t}\n(n={term_counts.get(t, 0)} matches)" for t in ct_term.index]
        melted_term = ct_term.reset_index().melt(id_vars="index", var_name="action", value_name="percentage")

        ax0 = sns.barplot(data=melted_term, x="index", y="percentage", hue="action", palette=palette, ax=axes[0])
        for container in ax0.containers:
            ax0.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9.5)
        axes[0].set_title("Strategy by Termination Type", fontsize=12)
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Strategy Share (%)", fontsize=11)
        axes[0].set_ylim(0, 114)
        axes[0].legend(title="Strategy")

        role_order = [
            r for r in ["Sent [DONE]", "Partner Sent [DONE]", "No [DONE] (Max Turns)"]
            if r in long_df["done_role"].unique()
        ]
        ct_role = (pd.crosstab(long_df["done_role"], long_df["action"], normalize="index") * 100).reindex(role_order)
        role_counts = long_df["done_role"].value_counts()
        ct_role.index = [f"{r}\n(n={role_counts.get(r, 0)} agents)" for r in ct_role.index]
        melted_role = ct_role.reset_index().melt(id_vars="index", var_name="action", value_name="percentage")

        ax1 = sns.barplot(data=melted_role, x="index", y="percentage", hue="action", palette=palette, ax=axes[1])
        for container in ax1.containers:
            ax1.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9.5)
        axes[1].set_title("Strategy by [DONE] Sender", fontsize=12)
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Strategy Share (%)", fontsize=11)
        axes[1].set_ylim(0, 114)
        axes[1].legend(title="Strategy")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "termination_vs_cooperation.png"), dpi=300)
        plt.close()

    def _plot_exchange_length_mean_variance(self, valid_df: pd.DataFrame, turn_df: pd.DataFrame):
        """
        Combine boxplots with explicit Mean (μ) and Variance/SD (σ², ±1 SD) overlays for exchange length:
        - Left panel: number of exchanges (turns) per match grouped by joint outcome.
        - Right panel: message word length across exchange turns (Turn 1 to Turn N).
        """
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

        outcome_order = sorted(valid_df["joint_outcome"].unique())
        short_labels = [
            "/".join(part.strip()[0].upper() for part in out.split("/") if part.strip())
            if "/" in out else out
            for out in outcome_order
        ]
        outcome_palette = sns.color_palette("viridis", max(len(outcome_order), 2)).as_hex()[:len(outcome_order)]

        sns.boxplot(
            data=valid_df,
            x="joint_outcome",
            y="turns",
            order=outcome_order,
            hue="joint_outcome",
            hue_order=outcome_order,
            dodge=False,
            palette=outcome_palette,
            boxprops={"alpha": 0.78},
            width=0.45,
            ax=axes[0],
        )
        if axes[0].get_legend():
            axes[0].get_legend().remove()

        sns.stripplot(
            data=valid_df,
            x="joint_outcome",
            y="turns",
            order=outcome_order,
            color="#1f2937",
            alpha=0.5,
            jitter=0.14,
            ax=axes[0],
        )

        stats_out = valid_df.groupby("joint_outcome")["turns"].agg(["mean", "std", "var"]).reindex(outcome_order).fillna(0.0)
        x_pos = np.arange(len(outcome_order))
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(short_labels)

        axes[0].errorbar(
            x_pos,
            stats_out["mean"],
            yerr=stats_out["std"],
            fmt="D",
            color="#111827",
            ecolor="#111827",
            markeredgecolor="white",
            markeredgewidth=1.0,
            elinewidth=1.8,
            capsize=5,
            markersize=7.5,
            zorder=6,
            label="Mean ± 1 SD",
        )
        y_max_0 = max(valid_df["turns"].max(), (stats_out["mean"] + stats_out["std"]).max())
        for idx, row in enumerate(stats_out.itertuples()):
            axes[0].text(
                idx,
                y_max_0 + 0.22,
                f"μ={row.mean:.2f}\nσ²={row.var:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
                color="#111827",
                fontweight="bold",
            )
        axes[0].set_ylim(0.5, y_max_0 + 1.15)
        axes[0].set_title("Exchanges (Turns) by Outcome", fontsize=12)
        axes[0].set_xlabel("Outcome (C = Cooperate, D = Defect)", fontsize=11)
        axes[0].set_ylabel("Exchanges (Turns)", fontsize=11)
        axes[0].legend(loc="lower right", fontsize=9)

        if not turn_df.empty:
            turn_order = sorted(turn_df["turn"].unique())
            turn_palette = sns.color_palette("viridis", max(len(turn_order), 2)).as_hex()[:len(turn_order)]
            sns.boxplot(
                data=turn_df,
                x="turn",
                y="words",
                order=turn_order,
                hue="turn",
                hue_order=turn_order,
                dodge=False,
                palette=turn_palette,
                boxprops={"alpha": 0.78},
                width=0.48,
                ax=axes[1],
            )
            if axes[1].get_legend():
                axes[1].get_legend().remove()

            stats_turn = turn_df.groupby("turn")["words"].agg(["mean", "std", "var"]).reindex(turn_order).fillna(0.0)
            x_pos_t = np.arange(len(turn_order))
            axes[1].errorbar(
                x_pos_t,
                stats_turn["mean"],
                yerr=stats_turn["std"],
                fmt="-D",
                color="#111827",
                ecolor="#111827",
                markeredgecolor="white",
                markeredgewidth=1.0,
                elinewidth=1.6,
                capsize=4,
                markersize=6.5,
                zorder=6,
                label="Mean ± 1 SD",
            )
            y_max_1 = max(turn_df["words"].max(), (stats_turn["mean"] + stats_turn["std"]).max())
            for idx, row in enumerate(stats_turn.itertuples()):
                axes[1].text(
                    idx,
                    y_max_1 + max(0.8, y_max_1 * 0.03),
                    f"μ={row.mean:.1f}\nσ²={row.var:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="#111827",
                )
            axes[1].set_ylim(0, y_max_1 + max(4.0, y_max_1 * 0.20))
            axes[1].set_title("Message Length by Turn", fontsize=12)
            axes[1].set_xlabel("Turn", fontsize=11)
            axes[1].set_ylabel("Words per Message", fontsize=11)
            axes[1].legend(loc="upper right", fontsize=9)
        else:
            axes[1].text(0.5, 0.5, "No turn-level message data available", ha="center", va="center")
            axes[1].set_axis_off()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "exchange_length_mean_variance.png"), dpi=300)
        plt.close()

    def _plot_done_timing_and_turn_trajectory(self, turn_df: pd.DataFrame):
        """
        Explore temporal negotiation dynamics across turns:
        - Left panel: distribution of exchange turns at which [DONE] is emitted by final strategy.
        - Right panel: turn-by-turn message length trajectory (Mean ± 1 SD) by final strategy.
        """
        if turn_df.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
        turn_order = sorted(turn_df["turn"].unique())
        done_turns = turn_df[turn_df["has_done"]]
        palette = self._get_strategy_palette(sorted(turn_df["action"].unique()))

        if not done_turns.empty:
            ct_timing = pd.crosstab(done_turns["turn"], done_turns["action"]).reindex(turn_order, fill_value=0)
            melted_timing = ct_timing.reset_index().melt(id_vars="turn", var_name="action", value_name="count")
            ax0 = sns.barplot(data=melted_timing, x="turn", y="count", hue="action", palette=palette, ax=axes[0])
            for container in ax0.containers:
                ax0.bar_label(container, padding=2, fontsize=9)
            axes[0].set_title("[DONE] Timing by Turn", fontsize=12)
            axes[0].set_xlabel("Turn", fontsize=11)
            axes[0].set_ylabel("[DONE] Messages", fontsize=11)
            axes[0].legend(title="Strategy")
        else:
            axes[0].text(0.5, 0.5, "No [DONE] tokens emitted in valid matches", ha="center", va="center")
            axes[0].set_axis_off()

        for act, grp in turn_df.groupby("action"):
            color = palette.get(act, self.primary_blue)
            agg_t = grp.groupby("turn")["words"].agg(["mean", "std"]).reindex(turn_order).fillna(0.0)
            axes[1].plot(turn_order, agg_t["mean"], marker="o", linewidth=2.2, color=color, label=f"{act} (Mean)")
            axes[1].fill_between(
                turn_order,
                np.maximum(0, agg_t["mean"] - agg_t["std"]),
                agg_t["mean"] + agg_t["std"],
                color=color,
                alpha=0.18,
                label=f"{act} (±1 SD)",
            )
        axes[1].set_xticks(turn_order)
        axes[1].set_title("Message Length Trajectory (Mean ± 1 SD)", fontsize=12)
        axes[1].set_xlabel("Turn", fontsize=11)
        axes[1].set_ylabel("Words per Message", fontsize=11)
        axes[1].legend(fontsize=8.5, frameon=True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "done_timing_and_turn_trajectory.png"), dpi=300)
        plt.close()

    def generate_all_graphs(self):
        """
        Collector that runs all private graph generation methods
        and saves them to the configured output directory.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        valid_df = self.dataset.valid_df
        if valid_df.empty:
            print("No valid data available to generate graphs.")
            return

        long_df = self.dataset.long_df
        turn_df = self.dataset.turn_df

        self._plot_outcome_matrix(valid_df)
        self._plot_verbosity_vs_action(long_df)
        self._plot_first_mover_comparison(long_df)
        self._plot_done_proportion_vs_words(valid_df, long_df)
        self._plot_termination_vs_cooperation(valid_df, long_df)
        self._plot_exchange_length_mean_variance(valid_df, turn_df)
        self._plot_done_timing_and_turn_trajectory(turn_df)
        self._plot_sanity_checks_overview(valid_df)
        self._plot_sanity_retries_vs_strategy(long_df)
        print(f"Graphs saved successfully in the folder '{self.output_dir}/'.")

    def _plot_sanity_checks_overview(self, valid_df: pd.DataFrame):
        """
        Visualize sanity-check diagnostics across valid matches:
        - Left panel: breakdown of triggered sanity-check rejection reasons.
        - Right panel: match distribution across retry/fallback intervention tiers.
        """
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))

        # Left panel: aggregated failure reasons
        reason_totals: dict[str, int] = {}
        for col in ("sanity_fail_reasons_A", "sanity_fail_reasons_B"):
            for reasons_dict in valid_df[col]:
                if isinstance(reasons_dict, dict):
                    for k, v in reasons_dict.items():
                        reason_totals[k] = reason_totals.get(k, 0) + int(v)

        if reason_totals:
            reason_df = (
                pd.DataFrame(sorted(reason_totals.items(), key=lambda x: x[1], reverse=True), columns=["reason", "count"])
            )
            total_fails = reason_df["count"].sum()
            ax0 = sns.barplot(
                data=reason_df,
                y="reason",
                x="count",
                color=self.primary_blue,
                ax=axes[0],
            )
            max_cnt = reason_df["count"].max()
            for bar, cnt in zip(ax0.patches, reason_df["count"]):
                pct = (cnt / total_fails) * 100
                ax0.text(
                    cnt + max(0.15, max_cnt * 0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f"{cnt} ({pct:.1f}%)",
                    va="center",
                    fontsize=9.5,
                )
            axes[0].set_xlim(0, max_cnt * 1.25)
            axes[0].set_title("Sanity Check Rejection Reasons", fontsize=12)
            axes[0].set_xlabel("Triggered Rejections (Count)", fontsize=11)
            axes[0].set_ylabel("")
        else:
            axes[0].text(0.5, 0.5, "No sanity check retries triggered\n(100% clean first-pass)", ha="center", va="center", fontsize=11)
            axes[0].set_title("Sanity Check Rejection Reasons", fontsize=12)
            axes[0].set_axis_off()

        # Right panel: match-level intervention tiers
        def _match_tier(row) -> str:
            if row["total_hardcoded_fallbacks"] > 0:
                return "Hardcoded\nFallback (>10)"
            if row["total_sanity_retries"] == 0:
                return "Clean\n(0 retries)"
            if row["total_sanity_retries"] <= 3:
                return "Light Retry\n(1–3 retries)"
            return "Heavy Retry\n(4–10 retries)"

        tier_order = [
            "Clean\n(0 retries)",
            "Light Retry\n(1–3 retries)",
            "Heavy Retry\n(4–10 retries)",
            "Hardcoded\nFallback (>10)",
        ]
        tiers = valid_df.apply(_match_tier, axis=1)
        tier_counts = tiers.value_counts().reindex(tier_order, fill_value=0)
        tier_df = pd.DataFrame({"tier": tier_counts.index, "matches": tier_counts.values})
        tier_palette = [self.primary_green, "#74c69d", self.primary_blue, "#dc2626"]

        ax1 = sns.barplot(
            data=tier_df,
            x="tier",
            y="matches",
            hue="tier",
            hue_order=tier_order,
            dodge=False,
            palette=tier_palette,
            ax=axes[1],
        )
        if axes[1].get_legend():
            axes[1].get_legend().remove()

        n_matches = max(len(valid_df), 1)
        for bar, cnt in zip(ax1.patches, tier_df["matches"]):
            pct = (cnt / n_matches) * 100
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                cnt + max(0.15, tier_df["matches"].max() * 0.02),
                f"{cnt}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9.5,
            )
        axes[1].set_ylim(0, max(tier_df["matches"].max() * 1.24, 1.5))
        axes[1].set_title("Match Distribution by Sanity Intervention", fontsize=12)
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Number of Matches", fontsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sanity_checks_overview.png"), dpi=300)
        plt.close()

    def _plot_sanity_retries_vs_strategy(self, long_df: pd.DataFrame):
        """
        Analyze whether sanity retries or hardcoded fallbacks are associated with final strategy:
        - Left panel: distribution of sanity retries per agent by chosen strategy.
        - Right panel: strategy rates (%) across agent sanity intervention tiers.
        """
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
        actions = sorted(long_df["action"].unique())
        palette = self._get_strategy_palette(actions)

        # Left panel: retries by strategy
        sns.boxplot(
            data=long_df,
            x="action",
            y="sanity_retries",
            order=actions,
            hue="action",
            hue_order=actions,
            dodge=False,
            palette=palette,
            boxprops={"alpha": 0.85},
            width=0.45,
            ax=axes[0],
        )
        if axes[0].get_legend():
            axes[0].get_legend().remove()

        sns.stripplot(
            data=long_df,
            x="action",
            y="sanity_retries",
            order=actions,
            color="#1f2937",
            alpha=0.45,
            jitter=0.15,
            ax=axes[0],
        )
        axes[0].set_title("Sanity Retries per Agent by Strategy", fontsize=12)
        axes[0].set_xlabel("Strategy", fontsize=11)
        axes[0].set_ylabel("Sanity Retries", fontsize=11)

        # Right panel: strategy rates by sanity intervention tier
        tier_order = [
            t for t in ["Clean (0 Retries)", "Corrected (1-10 Retries)", "Hardcoded Fallback"]
            if t in long_df["sanity_tier"].unique()
        ]
        ct_tier = (pd.crosstab(long_df["sanity_tier"], long_df["action"], normalize="index") * 100).reindex(tier_order)
        tier_counts = long_df["sanity_tier"].value_counts()
        ct_tier.index = [f"{t}\n(n={tier_counts.get(t, 0)} agents)" for t in ct_tier.index]
        melted_tier = ct_tier.reset_index().melt(id_vars="index", var_name="action", value_name="percentage")

        ax1 = sns.barplot(data=melted_tier, x="index", y="percentage", hue="action", palette=palette, ax=axes[1])
        for container in ax1.containers:
            ax1.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9.5)
        axes[1].set_title("Strategy Choice by Sanity Intervention", fontsize=12)
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Strategy Share (%)", fontsize=11)
        axes[1].set_ylim(0, 114)
        axes[1].legend(title="Strategy")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sanity_retries_vs_strategy.png"), dpi=300)
        plt.close()


class ComparisonGraphs:
    def __init__(
        self,
        baseline_dataset: DataSet,
        negotiation_dataset: DataSet,
        output_dir: str = "reports/comparison",
        baseline_label: str = "Baseline (0 Turns)",
        negotiation_label: str = "Negotiation (Chat)",
    ):
        self.baseline_ds = baseline_dataset
        self.neg_ds = negotiation_dataset
        self.output_dir = output_dir
        self.baseline_label = baseline_label
        self.negotiation_label = negotiation_label
        sns.set_theme(style="whitegrid", context="notebook")
        self.primary_blue = "#31688e"
        self.primary_green = "#35b779"
        self.condition_palette = {
            self.baseline_label: "#64748b",
            self.negotiation_label: self.primary_blue,
        }

    def _plot_joint_outcomes_comparison(self, base_valid: pd.DataFrame, neg_valid: pd.DataFrame):
        """
        Compare joint outcome distributions between silent baseline and negotiation:
        - Left panel: 4 canonical payoff matrix quadrants (C/C, C/D, D/C, D/D).
        - Right panel: 3 strategic regimes with Chi-square shift test.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))

        outcomes_4 = [
            "COOPERATE / COOPERATE",
            "COOPERATE / DEFECT",
            "DEFECT / COOPERATE",
            "DEFECT / DEFECT",
        ]
        short_4 = [
            "C / C\n(Mutual Coop)",
            "C / D\n(A Exploited)",
            "D / C\n(B Exploited)",
            "D / D\n(Mutual Defect)",
        ]

        rows_4 = []
        for cond, df in [(self.baseline_label, base_valid), (self.negotiation_label, neg_valid)]:
            n = len(df)
            counts = df["joint_outcome"].value_counts()
            for out, s_lbl in zip(outcomes_4, short_4):
                c = int(counts.get(out, 0))
                rows_4.append({
                    "Condition": cond,
                    "Outcome": s_lbl,
                    "Percentage": (c / n * 100) if n > 0 else 0.0,
                })
        df_4 = pd.DataFrame(rows_4)

        ax0 = sns.barplot(
            data=df_4,
            x="Outcome",
            y="Percentage",
            hue="Condition",
            palette=self.condition_palette,
            ax=axes[0],
        )
        for container in ax0.containers:
            if isinstance(container, BarContainer):
                ax0.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)
        axes[0].set_title("Joint Outcomes Breakdown (4 Quadrants)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Share of Valid Matches (%)", fontsize=11)
        axes[0].set_ylim(0, 112)
        axes[0].legend(title="Condition", frameon=True)

        def _regime(out: str) -> str:
            if out == "COOPERATE / COOPERATE":
                return "Mutual Cooperation\n(C/C)"
            if out == "DEFECT / DEFECT":
                return "Mutual Defection\n(D/D - Nash)"
            return "Asymmetric Betrayal\n(C/D or D/C)"

        regimes = [
            "Mutual Cooperation\n(C/C)",
            "Asymmetric Betrayal\n(C/D or D/C)",
            "Mutual Defection\n(D/D - Nash)",
        ]
        rows_3 = []
        contingency = []
        for cond, df in [(self.baseline_label, base_valid), (self.negotiation_label, neg_valid)]:
            n = len(df)
            counts = df["joint_outcome"].map(_regime).value_counts()
            row_counts = []
            for reg in regimes:
                c = int(counts.get(reg, 0))
                row_counts.append(c)
                rows_3.append({
                    "Condition": cond,
                    "Regime": reg,
                    "Percentage": (c / n * 100) if n > 0 else 0.0,
                })
            contingency.append(row_counts)
        df_3 = pd.DataFrame(rows_3)

        cont_arr = np.array(contingency)
        active_cols = cont_arr.sum(axis=0) > 0
        if active_cols.sum() >= 2 and cont_arr.sum() > 0:
            chi2, p_val, _, _ = stats.chi2_contingency(cont_arr[:, active_cols])
            stat_note = f"Chi² = {chi2:.2f}, p = {p_val:.4f}"
        else:
            stat_note = "Insufficient variance for Chi²"

        ax1 = sns.barplot(
            data=df_3,
            x="Regime",
            y="Percentage",
            hue="Condition",
            palette=self.condition_palette,
            ax=axes[1],
        )
        for container in ax1.containers:
            if isinstance(container, BarContainer):
                ax1.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9.5)
        axes[1].set_title(f"Strategic Regimes ({stat_note})", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Share of Valid Matches (%)", fontsize=11)
        axes[1].set_ylim(0, 112)
        axes[1].legend(title="Condition", frameon=True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comparison_joint_outcomes.png"), dpi=300)
        plt.close()

    def _plot_cooperation_and_payoff(self, base_long: pd.DataFrame, neg_long: pd.DataFrame):
        """
        Compare individual cooperation rates (95% Wilson CI + Fisher's exact test)
        and mean individual payoffs (95% CI + Mann-Whitney U test + Nash/Pareto benchmarks).
        """
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
        conditions = [self.baseline_label, self.negotiation_label]
        colors = [self.condition_palette[c] for c in conditions]

        coop_rates, yerr_low, yerr_high, counts_info = [], [], [], []
        k_list, n_list = [], []

        for df in [base_long, neg_long]:
            n = len(df)
            k = int((df["action"] == "COOPERATE").sum())
            rate = (k / n * 100) if n > 0 else 0.0
            lo, hi = StatsEngine._wilson_ci(k, n)
            coop_rates.append(rate)
            yerr_low.append(rate - lo)
            yerr_high.append(hi - rate)
            counts_info.append(f"{rate:.1f}%\n({k}/{n})")
            k_list.append(k)
            n_list.append(n)

        _, p_coop = stats.fisher_exact([
            [k_list[0], n_list[0] - k_list[0]],
            [k_list[1], n_list[1] - k_list[1]],
        ])

        x_pos = np.arange(len(conditions))
        bars0 = axes[0].bar(
            x_pos,
            coop_rates,
            yerr=[yerr_low, yerr_high],
            capsize=6,
            width=0.48,
            color=colors,
            edgecolor="#1f2937",
            linewidth=0.8,
            error_kw={"elinewidth": 1.5, "ecolor": "#1f2937"},
        )
        for bar, lbl, hi_err in zip(bars0, counts_info, yerr_high):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + hi_err + 2.5,
                lbl,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([f"{c}\n(n={n} agents)" for c, n in zip(conditions, n_list)], fontsize=10.5)
        axes[0].set_ylim(0, 118)
        axes[0].set_ylabel("Cooperation Rate (% ± 95% Wilson CI)", fontsize=11)
        axes[0].set_title(f"Individual Cooperation Rate (Fisher p = {p_coop:.4f})", fontsize=12, fontweight="bold")

        pay_means, pay_ci = [], []
        for df in [base_long, neg_long]:
            vals = df["payoff"].astype(float)
            m = float(vals.mean())
            se = float(vals.sem()) if len(vals) > 1 else 0.0
            pay_means.append(m)
            pay_ci.append(1.96 * se)

        _, p_pay = stats.mannwhitneyu(base_long["payoff"], neg_long["payoff"], alternative="two-sided")

        bars1 = axes[1].bar(
            x_pos,
            pay_means,
            yerr=pay_ci,
            capsize=6,
            width=0.48,
            color=colors,
            edgecolor="#1f2937",
            linewidth=0.8,
            error_kw={"elinewidth": 1.5, "ecolor": "#1f2937"},
        )
        for bar, m, ci in zip(bars1, pay_means, pay_ci):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci + 0.12,
                f"{m:.2f} pts",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        axes[1].axhline(3.0, color=self.primary_green, linestyle="--", linewidth=1.4, label="Pareto Optimum (CC = 3.0)")
        axes[1].axhline(1.0, color="#dc2626", linestyle=":", linewidth=1.4, label="Nash Equilibrium (DD = 1.0)")
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([f"{c}\n(n={n} agents)" for c, n in zip(conditions, n_list)], fontsize=10.5)
        axes[1].set_ylim(0, 4.2)
        axes[1].set_ylabel("Mean Individual Payoff (Points ± 95% CI)", fontsize=11)
        axes[1].set_title(f"Mean Payoff per Agent (Mann-Whitney p = {p_pay:.4f})", fontsize=12, fontweight="bold")
        axes[1].legend(loc="upper left", frameon=True, fontsize=9.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comparison_cooperation_and_payoff.png"), dpi=300)
        plt.close()

    def _plot_payoff_by_strategy_condition(self, base_long: pd.DataFrame, neg_long: pd.DataFrame):
        """
        Compare expected payoff of COOPERATE vs DEFECT across baseline and negotiation
        to evaluate whether pre-game communication makes cooperation pay off over defection.
        """
        df_b = base_long.copy()
        df_b["Condition"] = self.baseline_label
        df_n = neg_long.copy()
        df_n["Condition"] = self.negotiation_label
        combined = pd.concat([df_b, df_n], ignore_index=True)

        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        ax = sns.barplot(
            data=combined,
            x="action",
            y="payoff",
            order=["COOPERATE", "DEFECT"],
            hue="Condition",
            palette=self.condition_palette,
            errorbar=("ci", 95),
            capsize=0.08,
            ax=ax,
        )
        for container in ax.containers:
            if isinstance(container, BarContainer):
                ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9.5)

        ax.axhline(3.0, color=self.primary_green, linestyle="--", linewidth=1.2, alpha=0.8, label="Mutual Coop (3.0)")
        ax.axhline(1.0, color="#dc2626", linestyle=":", linewidth=1.2, alpha=0.8, label="Mutual Defect (1.0)")
        ax.set_title("Expected Payoff by Chosen Strategy Across Conditions", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Chosen Strategy", fontsize=11)
        ax.set_ylabel("Mean Payoff (Points ± 95% CI)", fontsize=11)
        ax.set_ylim(0, 5.4)
        ax.legend(title="Condition / Reference", frameon=True, fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comparison_payoff_by_strategy.png"), dpi=300)
        plt.close()

    def generate_all_comparison_graphs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.baseline_ds.valid_df.empty or self.neg_ds.valid_df.empty:
            print("Insufficient valid data in baseline or negotiation dataset for comparison graphs.")
            return

        self._plot_joint_outcomes_comparison(self.baseline_ds.valid_df, self.neg_ds.valid_df)
        self._plot_cooperation_and_payoff(self.baseline_ds.long_df, self.neg_ds.long_df)
        self._plot_payoff_by_strategy_condition(self.baseline_ds.long_df, self.neg_ds.long_df)
        print(f"Comparison graphs saved successfully in '{self.output_dir}/'.")


class ChatReport:
    def __init__(self, dataset: DataSet, output_dir: str = "reports", template_path: str | None = None):
        self.dataset = dataset
        self.output_dir = output_dir
        if template_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            template_path = os.path.join(base_dir, "templates", "report_template.html")
        self.template_path = template_path

    @staticmethod
    def _action_badge(action: str) -> str:
        """Return a color-coded HTML badge for an agent's chosen action."""
        clean = html.escape(str(action))
        upper = str(action).upper()
        if "COOP" in upper:
            cls = "badge-coop"
        elif "DEFECT" in upper:
            cls = "badge-defect"
        else:
            cls = "badge-error"
        return f'<span class="badge {cls}">{clean}</span>'

    def _render_conversation_bubbles(
        self, messages: list, id_a: str, id_b: str, first_id: str
    ) -> tuple[str, str, int]:
        """
        Convert raw message history into strictly separated left/right HTML chat bubbles
        (Player A on the left, Player B on the right), extract the system prompt separately,
        and label each Agent A / Agent B exchange pair with a single turn counter (1 to max_exchanges).
        """
        raw_sys, raw_judge, agent_msgs, total_turns = _extract_diplomatic_messages(
            messages,
            id_a=str(id_a),
            id_b=str(id_b),
            max_exchanges=self.dataset.max_exchanges,
        )

        system_prompt = html.escape(raw_sys) if raw_sys else ""
        bubbles_html = []

        if raw_judge:
            bubbles_html.append(f"""
            <div class="msg-row msg-center">
                <div class="bubble bubble-judge">JUDGE: {html.escape(raw_judge)}</div>
            </div>
            """)

        init_tag_a = '<span class="badge badge-initiator">Initiator</span>' if str(first_id) == str(id_a) else ""
        init_tag_b = '<span class="badge badge-initiator">Initiator</span>' if str(first_id) == str(id_b) else ""

        for item in agent_msgs:
            turn_counter = item["turn"]
            clean_text = item["text"]
            if item["speaker"] == "agent_a":
                bubbles_html.append(f"""
                <div class="msg-row msg-left">
                    <div class="msg-sender">Turn {turn_counter} &bull; Player A (ID: {html.escape(str(id_a))}){init_tag_a}</div>
                    <div class="bubble bubble-a">{html.escape(clean_text)}</div>
                </div>
                """)
            else:
                bubbles_html.append(f"""
                <div class="msg-row msg-right">
                    <div class="msg-sender">Turn {turn_counter} &bull; Player B (ID: {html.escape(str(id_b))}){init_tag_b}</div>
                    <div class="bubble bubble-b">{html.escape(clean_text)}</div>
                </div>
                """)

        return system_prompt, "\n".join(bubbles_html), total_turns

    def _render_match_card(self, idx: int, rec: dict) -> str:
        """Render a single simulation match as a complete HTML card."""
        act_a = rec["agentA"]["answer"]
        act_b = rec["agentB"]["answer"]
        id_a = rec["agentA"]["id"]
        id_b = rec["agentB"]["id"]
        expl_a = html.escape(str(rec["agentA"]["explanation"]))
        expl_b = html.escape(str(rec["agentB"]["explanation"]))
        pay_a = rec["results"]["payoffA"]
        pay_b = rec["results"]["payoffB"]
        first_id = rec.get("first_speaker_id")
        timestamp = html.escape(str(rec.get("time", "")))
        game_name = html.escape(str(rec.get("game", "unknown")))

        if act_a == "[ERROR]" or act_b == "[ERROR]":
            category = "ERROR"
            outcome_badge = '<span class="badge badge-error">PARSING ERROR</span>'
        elif act_a == act_b and "COOP" in act_a.upper():
            category = "MUTUAL_COOP"
            outcome_badge = f'<span class="badge badge-coop">{html.escape(act_a)} / {html.escape(act_b)}</span>'
        elif act_a == act_b:
            category = "MUTUAL_DEFECT"
            outcome_badge = f'<span class="badge badge-defect">{html.escape(act_a)} / {html.escape(act_b)}</span>'
        else:
            category = "ASYMMETRIC"
            outcome_badge = f'<span class="badge badge-mixed">{html.escape(act_a)} / {html.escape(act_b)}</span>'

        sys_prompt, chat_html, total_turns = self._render_conversation_bubbles(
            rec.get("full_conversation", []), id_a, id_b, first_id
        )

        init_lane_a = ' <span class="badge badge-initiator">Initiator</span>' if str(first_id) == str(id_a) else ""
        init_lane_b = ' <span class="badge badge-initiator">Initiator</span>' if str(first_id) == str(id_b) else ""

        sys_block = f"""
        <details class="sys-prompt">
            <summary>Show System Prompt</summary>
            <pre>{sys_prompt}</pre>
        </details>
        """ if sys_prompt else ""

        return f"""
        <section class="match-card" data-category="{category}">
            <div class="match-header">
                <div>
                    <span class="match-title">Match #{idx + 1}</span>
                    <span class="match-sub"> &bull; {game_name} &bull; {timestamp} &bull; {total_turns} turns</span>
                </div>
                <div>{outcome_badge}</div>
            </div>
            {sys_block}
            <div class="chat-lanes-bar">
                <span class="lane-label-a">&larr; Player A (ID: {html.escape(str(id_a))}){init_lane_a}</span>
                <span class="lane-label-b">Player B (ID: {html.escape(str(id_b))}){init_lane_b} &rarr;</span>
            </div>
            <div class="chat-box">
                {chat_html}
            </div>
            <div class="verdict-grid">
                <div class="verdict-card">
                    <div class="verdict-top">
                        <strong>Player A (ID: {html.escape(str(id_a))})</strong>
                        <div>
                            {self._action_badge(act_a)}
                            <span class="payoff-pill">Payoff: {pay_a}</span>
                        </div>
                    </div>
                    <p class="explanation-text">{expl_a}</p>
                </div>
                <div class="verdict-card">
                    <div class="verdict-top">
                        <strong>Player B (ID: {html.escape(str(id_b))})</strong>
                        <div>
                            {self._action_badge(act_b)}
                            <span class="payoff-pill">Payoff: {pay_b}</span>
                        </div>
                    </div>
                    <p class="explanation-text">{expl_b}</p>
                </div>
            </div>
        </section>
        """

    def generate_html_report(self, filename: str = "conversations_report.html") -> str:
        """
        Collector method that compiles all simulated matches into a standalone,
        filterable HTML document using an external HTML template.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.dataset.raw_records:
            print("No records available to generate the HTML report.")
            return ""

        with open(self.template_path, "r", encoding="utf-8") as f:
            template = Template(f.read())

        cards_html = "\n".join(
            self._render_match_card(idx, rec)
            for idx, rec in enumerate(self.dataset.raw_records)
        )

        full_html = template.safe_substitute(
            SOURCE_PATH=html.escape(self.dataset.file_path),
            TOTAL_MATCHES=len(self.dataset.raw_records),
            VALID_MATCHES=len(self.dataset.valid_df),
            ERROR_RATE=self.dataset.error_rate,
            CARDS_HTML=cards_html,
        )

        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_html)

        print(f"HTML conversation report saved to '{output_path}'.")
        return output_path

    
def _summarize_bucket(
    explanations: list[str],
    category_label: str,
    focus_instruction: str,
    model_name: str = "gemma4",
    chunk_size: int = 30,
) -> str:
    if not explanations:
        return "No instances recorded in this category."

    chunks = [explanations[i : i + chunk_size] for i in range(0, len(explanations), chunk_size)]
    partial_summaries = []

    for idx, chunk in enumerate(chunks, start=1):
        numbered = "\n".join(f"- {text}" for text in chunk)
        prompt = (
            f"You are analyzing post-game decision explanations from a Prisoner's Dilemma experiment.\n"
            f"Category: {category_label} (batch {idx}/{len(chunks)}, {len(chunk)} explanations).\n"
            f"{focus_instruction}\n\n"
            f"Explanations:\n{numbered}\n\n"
            f"Format your response in clean Markdown using ONLY '###' and '####' subheadings (do not use '#' or '##', and do not add separate Executive Summary or Conclusion headers). "
            f"Provide a concise, structured synthesis of the recurring reasoning patterns, key arguments, and strategic drivers."
        )

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": 16384},
        )
        partial_summaries.append(response["message"]["content"].strip())

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    combined = "\n\n---\n\n".join(
        f"Batch {i + 1} Summary:\n{summary}" for i, summary in enumerate(partial_summaries)
    )
    reduce_prompt = (
        f"Synthesize the following batch summaries for the '{category_label}' category "
        f"({len(explanations)} total explanations) into a single coherent, non-redundant analysis:\n\n{combined}"
    )
    final_response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": reduce_prompt}],
        options={"num_ctx": 16384},
    )
    return final_response["message"]["content"].strip()


def summarize_explanations(
    json_path: str = "data/response_history.json",
    output_path: str = "reports/explanations_summary.md",
    model_name: str = "gemma4",
    chunk_size: int = 30,
) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    buckets = {
        "mutual_cooperation": [],  # C / C -> entrambe le spiegazioni
        "mutual_defection": [],    # D / D -> entrambe le spiegazioni
        "betrayed": [],            # C / D -> solo il lato COOPERATE (tradito)
        "betrayers": [],           # D / C -> solo il lato DEFECT (traditore)
    }

    for rec in records:
        ans_a = rec["agentA"]["answer"]
        ans_b = rec["agentB"]["answer"]
        expl_a = rec["agentA"]["explanation"]
        expl_b = rec["agentB"]["explanation"]

        if ans_a == "COOPERATE" and ans_b == "COOPERATE":
            buckets["mutual_cooperation"].extend([expl_a, expl_b])
        elif ans_a == "DEFECT" and ans_b == "DEFECT":
            buckets["mutual_defection"].extend([expl_a, expl_b])
        elif ans_a == "COOPERATE" and ans_b == "DEFECT":
            buckets["betrayed"].append(expl_a)
            buckets["betrayers"].append(expl_b)
        elif ans_a == "DEFECT" and ans_b == "COOPERATE":
            buckets["betrayers"].append(expl_a)
            buckets["betrayed"].append(expl_b)

    categories_spec = [
        (
            "mutual_cooperation",
            "1. ENTRAMBI FEDELI (Mutual Cooperation: COOPERATE / COOPERATE)",
            "Analyze why both agents chose to cooperate and honor mutual trust over the dominant strategy.",
        ),
        (
            "mutual_defection",
            "2. ENTRAMBI SLEALI (Mutual Defection: DEFECT / DEFECT)",
            "Analyze why both agents converged on defection (mistrust, risk aversion, or pure Nash equilibrium reasoning).",
        ),
        (
            "betrayed",
            "3. TRADITI (Exploited Cooperators: COOPERATE against DEFECT)",
            "Analyze ONLY the perspective of the agents who cooperated while their opponent defected: what led them to trust or accept the risk?",
        ),
        (
            "betrayers",
            "4. TRADITORI (Opportunistic Defectors: DEFECT against COOPERATE)",
            "Analyze ONLY the perspective of the agents who defected while their opponent cooperated: how did they justify exploiting the opponent's trust?",
        ),
    ]

    sections = [f"EXPLANATIONS SUMMARY (Total games analyzed: {len(records)})\n" + "=" * 60]

    for key, title, focus in categories_spec:
        items = buckets[key]
        summary = _summarize_bucket(
            explanations=items,
            category_label=title,
            focus_instruction=focus,
            model_name=model_name,
            chunk_size=chunk_size,
        )
        sections.append(f"\n{title}\nCount: {len(items)} explanations\n{'-' * 60}\n{summary}")

    full_report = "\n".join(sections).strip() + "\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    return full_report


def generate_all_pipeline(
    json_path: str = "data/response_history.json",
    output_dir: str = "reports",
    model_name: str = "gemma4",
    baseline_path: str | None = None,
):
    dataset = DataSet(json_path)

    stats_engine = StatsEngine(dataset)
    stats_engine.run_all_analyses(export_dir=output_dir)

    graphs = Graphs(dataset, output_dir=output_dir)
    graphs.generate_all_graphs()

    chat_report = ChatReport(dataset, output_dir=output_dir)
    chat_report.generate_html_report()

    summarize_explanations(
        json_path=json_path,
        output_path=os.path.join(output_dir, "explanations_summary.md"),
        model_name=model_name,
    )

    if baseline_path:
        baseline_ds = DataSet(baseline_path)
        comp_graphs = ComparisonGraphs(
            baseline_dataset=baseline_ds,
            negotiation_dataset=dataset,
            output_dir=os.path.join(output_dir, "comparison"),
        )
        comp_graphs.generate_all_comparison_graphs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze simulation history and generate reports/graphs.")
    parser.add_argument("-f", "--file", type=str, default="data/response_history.json", help="Path to the input simulation JSON file.")
    parser.add_argument("-b", "--baseline", type=str, default=None, help="Optional path to baseline JSON history to generate comparison graphs.")
    parser.add_argument("-o", "--output_dir", type=str, default="reports", help="Directory where reports and graphs will be saved.")
    parser.add_argument("-m", "--model_name", type=str, default="gemma4", help="LLM model name used to summarize explanations.")
    args = parser.parse_args()

    generate_all_pipeline(
        json_path=args.file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        baseline_path=args.baseline,
    )
