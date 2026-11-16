import os, json, html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class DataSet:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_records = []
        self.df = self.load_data()
        self.error_rate = round(float(self.df["has_error"].mean() * 100), 2)
        self.valid_df = self._get_valid_df()
        self.long_df = self._build_long_agent_df(self.valid_df) if not self.valid_df.empty else pd.DataFrame()

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
        for idx, rec in enumerate(self.raw_records):
            act_a = rec["agentA"]["answer"]
            act_b = rec["agentB"]["answer"]
            id_a = rec["agentA"]["id"]
            id_b = rec["agentB"]["id"]
            first_id = rec.get("first_speaker_id")

            has_error = (act_a == "[ERROR]") or (act_b == "[ERROR]")
            joint_outcome = "ERROR" if has_error else f"{act_a} / {act_b}"

            # Compute diplomatic verbosity (excluding system prompt and JUDGE messages)
            words_a, words_b = self._extract_diplomatic_word_counts(rec.get("full_conversation", []))

            rows.append({
                "match_id": idx + 1,
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
                "words_A": words_a,
                "words_B": words_b,
                "total_words": words_a + words_b,
                "has_error": has_error
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _extract_diplomatic_word_counts(messages: list) -> tuple[int, int]:
        """
        Count the words exchanged by both agents during the diplomatic phase only,
        stopping the count before the final [JUDGE] intervention.
        """
        words_a = 0
        words_b = 0
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                continue
            if role == "user" and content.startswith("[JUDGE]: Now the other agent can't listen to you"):
                break
            if role == "assistant":
                words_a += len(content.split())
            elif role == "user" and content.startswith("[AGENT]:"):
                clean_text = content.replace("[AGENT]:", "", 1).strip()
                words_b += len(clean_text.split())
        return words_a, words_b

    def _get_valid_df(self) -> pd.DataFrame:
        """Return only matches without parsing errors."""
        return self.df[~self.df["has_error"]].copy()

    def _build_long_agent_df(self, valid_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame from match-level to agent-level (2 rows per match)
        to facilitate aggregated statistical tests.
        """
        df_a = pd.DataFrame({
            "match_id": valid_df["match_id"],
            "role": "Agent A",
            "action": valid_df["action_A"],
            "payoff": valid_df["payoff_A"],
            "words": valid_df["words_A"],
            "is_initiator": valid_df["first_speaker"] == "Agent A"
        })
        df_b = pd.DataFrame({
            "match_id": valid_df["match_id"],
            "role": "Agent B",
            "action": valid_df["action_B"],
            "payoff": valid_df["payoff_B"],
            "words": valid_df["words_B"],
            "is_initiator": valid_df["first_speaker"] == "Agent B"
        })
        return pd.concat([df_a, df_b], ignore_index=True)


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
        Point-biserial correlation between diplomatic verbosity (word count)
        and strategy choice (binary-encoded).
        """
        actions = sorted(long_df["action"].unique())
        if len(actions) < 2 or long_df["words"].std() == 0:
            return {"note": "Insufficient variance to calculate correlation with verbosity."}

        # Binary encoding (e.g., DEFECT = 1 if present, otherwise second action alphabetically)
        target_action = "DEFECT" if "DEFECT" in actions else actions[1]
        binary_target = (long_df["action"] == target_action).astype(int)

        r_pb, p_val = stats.pointbiserialr(binary_target, long_df["words"])
        return {
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
                "parsing_error_rate_pct": self.dataset.error_rate
            },
            "action_rates_pct": self._compute_action_rates(long_df),
            "outcome_matrix_pct": self._compute_outcome_matrix(valid_df),
            "payoff_stats": self._compute_payoff_stats(long_df),
            "first_mover_pct": self._compute_first_mover_stats(long_df),
            "chi_square_tests": self._test_chi_square_independence(long_df),
            "symmetry_binomial_test": self._test_symmetry_binomial(valid_df),
            "payoff_mann_whitney_test": self._test_payoff_differences(long_df),
            "verbosity_correlation_test": self._test_verbosity_correlation(long_df)
        }

        # Export tables and statistical test reports
        self.results["action_rates_pct"].to_csv(os.path.join(export_dir, "action_rates.csv"))
        self.results["outcome_matrix_pct"].to_csv(os.path.join(export_dir, "outcome_matrix.csv"))
        self.results["payoff_stats"].to_csv(os.path.join(export_dir, "payoff_stats.csv"))

        stats_summary = {
            "meta": self.results["meta"],
            "chi_square_tests": self.results["chi_square_tests"],
            "symmetry_binomial_test": self.results["symmetry_binomial_test"],
            "payoff_mann_whitney_test": self.results["payoff_mann_whitney_test"],
            "verbosity_correlation_test": self.results["verbosity_correlation_test"]
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
              f"Parsing errors: {self.results['meta']['parsing_error_rate_pct']}%")
        print("\n--- Choice Rates (%) ---")
        print(self.results["action_rates_pct"])
        print("\n--- Joint Outcomes Matrix (% of total) ---")
        print(self.results["outcome_matrix_pct"])
        print("\n--- Payoff Statistics ---")
        print(self.results["payoff_stats"])
        print("\n--- Statistical Tests ---")
        print(json.dumps({
            "chi_square": self.results["chi_square_tests"],
            "symmetry_binomial": self.results["symmetry_binomial_test"],
            "payoff_mann_whitney": self.results["payoff_mann_whitney_test"],
            "verbosity_correlation": self.results["verbosity_correlation_test"]
        }, indent=2))
        print("=" * 60 + "\n")


class Graphs:
    def __init__(self, dataset: DataSet, output_dir: str = "reports"):
        self.dataset = dataset
        self.output_dir = output_dir
        sns.set_theme(style="whitegrid", context="notebook")

    def _plot_outcome_matrix(self, valid_df: pd.DataFrame):
        """
        Generate a 2x2 matrix (heatmap) with percentages and absolute counts
        of joint outcomes (Agent A vs Agent B).
        """
        counts = pd.crosstab(valid_df["action_A"], valid_df["action_B"])
        pcts = pd.crosstab(valid_df["action_A"], valid_df["action_B"], normalize="all") * 100

        # Combined labels: "45.0%\n(n=9)"
        annot_matrix = np.empty(counts.shape, dtype=object)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                annot_matrix[i, j] = f"{pcts.iloc[i, j]:.1f}%\n(n={counts.iloc[i, j]})"

        plt.figure(figsize=(7, 5.5))
        ax = sns.heatmap(
            pcts,
            annot=annot_matrix,
            fmt="",
            cmap="Blues",
            cbar_kws={"label": "Percentage on valid matches (%)"},
            vmin=0,
            vmax=100,
            linewidths=1.5,
            linecolor="white",
            annot_kws={"size": 13, "weight": "bold"}
        )
        ax.set_title("Matrix of Joint Outcomes (Agent A vs Agent B)", fontsize=14, pad=14)
        ax.set_xlabel("Choice Agent B", fontsize=12)
        ax.set_ylabel("Choice Agent A", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "outcome_matrix_heatmap.png"), dpi=300)
        plt.close()

    def _plot_payoff_distribution(self, long_df: pd.DataFrame):
        """
        Compare payoff distribution by chosen action and role using boxplot + stripplot.
        """
        plt.figure(figsize=(8, 5))
        ax = sns.boxplot(
            data=long_df,
            x="action",
            y="payoff",
            hue="role",
            palette="Set2",
            width=0.5,
            fliersize=0
        )
        sns.stripplot(
            data=long_df,
            x="action",
            y="payoff",
            hue="role",
            dodge=True,
            palette="dark:black",
            alpha=0.45,
            jitter=0.15,
            legend=False,
            ax=ax
        )
        ax.set_title("Payoff Distribution by Strategy and Role", fontsize=14, pad=12)
        ax.set_xlabel("Selected Strategy", fontsize=12)
        ax.set_ylabel("Obtained Payoff", fontsize=12)
        plt.legend(title="Role")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "payoff_distribution.png"), dpi=300)
        plt.close()

    def _plot_verbosity_vs_action(self, long_df: pd.DataFrame):
        """
        Analyze the relationship between the number of words produced in the diplomatic phase and the final choice.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left panel: word distribution by strategy
        sns.boxplot(data=long_df, x="action", y="words", hue="action", palette="Pastel1", ax=axes[0], width=0.45, legend=False)
        sns.stripplot(data=long_df, x="action", y="words", color="black", alpha=0.5, jitter=0.15, ax=axes[0])
        axes[0].set_title("Diplomatic Verbosity by Final Choice", fontsize=12)
        axes[0].set_xlabel("Selected Strategy", fontsize=11)
        axes[0].set_ylabel("Words written in the diplomatic phase", fontsize=11)

        # Right panel: estimated probability (logistic regression)
        actions = sorted(long_df["action"].unique())
        if len(actions) >= 2 and long_df["words"].std() > 0:
            target_action = "DEFECT" if "DEFECT" in actions else actions[1]
            temp = long_df.copy()
            temp["target_bin"] = (temp["action"] == target_action).astype(int)
            sns.regplot(
                data=temp,
                x="words",
                y="target_bin",
                logistic=True,
                ci=95,
                scatter_kws={"alpha": 0.4, "color": "#2b5c8f"},
                line_kws={"color": "#d95f02", "lw": 2},
                ax=axes[1]
            )
            axes[1].set_title(f"Probability of '{target_action}' vs Verbosity", fontsize=12)
            axes[1].set_xlabel("Words written in the diplomatic phase", fontsize=11)
            axes[1].set_ylabel(f"Probability P({target_action})", fontsize=11)
            axes[1].set_ylim(-0.05, 1.05)
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
        temp["turn_role"] = np.where(temp["is_initiator"], "Initiator (1° a parlare)", "Responder (2° a parlare)")
        ct_pct = (pd.crosstab(temp["turn_role"], temp["action"], normalize="index") * 100).reset_index()
        melted = ct_pct.melt(id_vars="turn_role", var_name="action", value_name="percentage")

        plt.figure(figsize=(7.5, 5))
        ax = sns.barplot(data=melted, x="turn_role", y="percentage", hue="action", palette="viridis")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=10)

        ax.set_title("Impact of Speaking Turn on Final Choice", fontsize=14, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.set_ylim(0, 112)
        plt.legend(title="Strategy")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "first_mover_impact.png"), dpi=300)
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

        self._plot_outcome_matrix(valid_df)
        self._plot_payoff_distribution(long_df)
        self._plot_verbosity_vs_action(long_df)
        self._plot_first_mover_comparison(long_df)
        print(f"Graphs saved successfully in the folder '{self.output_dir}/'.")


class ChatReport:
    def __init__(self, dataset: DataSet, output_dir: str = "reports"):
        self.dataset = dataset
        self.output_dir = output_dir

    # --- PRIVATE TEMPLATE & RENDERING METHODS ---

    def _get_css(self) -> str:
        """Return embedded CSS styles for a clean, chat-like inspection interface."""
        return """
        <style>
            :root {
                --bg: #f4f6f9;
                --card-bg: #ffffff;
                --text: #1f2937;
                --muted: #6b7280;
                --border: #e5e7eb;
                --agent-a-bg: #eff6ff;
                --agent-a-border: #bfdbfe;
                --agent-b-bg: #ecfdf5;
                --agent-b-border: #a7f3d0;
                --judge-bg: #fef3c7;
                --judge-text: #92400e;
                --coop: #16a34a;
                --defect: #dc2626;
                --error: #6b7280;
            }
            * { box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 24px;
                line-height: 1.5;
            }
            .container { max-width: 980px; margin: 0 auto; }
            header {
                background: var(--card-bg);
                padding: 20px 24px;
                border-radius: 12px;
                border: 1px solid var(--border);
                margin-bottom: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }
            header h1 { margin: 0 0 6px 0; font-size: 1.5rem; }
            .meta-bar { color: var(--muted); font-size: 0.9rem; margin-bottom: 14px; }
            .filters { display: flex; flex-wrap: wrap; gap: 8px; }
            .filter-btn {
                border: 1px solid var(--border);
                background: #f9fafb;
                padding: 6px 14px;
                border-radius: 999px;
                cursor: pointer;
                font-size: 0.85rem;
                font-weight: 600;
                color: var(--text);
                transition: all 0.15s ease;
            }
            .filter-btn:hover { background: #e5e7eb; }
            .filter-btn.active { background: #111827; color: #ffffff; border-color: #111827; }

            .match-card {
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 12px;
                margin-bottom: 24px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            }
            .match-header {
                padding: 14px 20px;
                background: #f9fafb;
                border-bottom: 1px solid var(--border);
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 10px;
            }
            .match-title { font-weight: 700; font-size: 1.05rem; }
            .match-sub { font-size: 0.82rem; color: var(--muted); }
            .badge {
                display: inline-block;
                padding: 3px 10px;
                border-radius: 6px;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                text-transform: uppercase;
            }
            .badge-coop { background: #dcfce7; color: #166534; }
            .badge-defect { background: #fee2e2; color: #991b1b; }
            .badge-mixed { background: #fef3c7; color: #92400e; }
            .badge-error { background: #e5e7eb; color: #374151; }
            .badge-initiator {
                background: #e0e7ff;
                color: #3730a3;
                font-size: 0.7rem;
                padding: 2px 6px;
                margin-left: 6px;
            }

            details.sys-prompt {
                padding: 10px 20px;
                background: #fafafa;
                border-bottom: 1px solid var(--border);
                font-size: 0.85rem;
            }
            details.sys-prompt summary { cursor: pointer; color: var(--muted); font-weight: 600; }
            details.sys-prompt pre {
                white-space: pre-wrap;
                background: #f3f4f6;
                padding: 10px;
                border-radius: 6px;
                margin-top: 8px;
                font-size: 0.8rem;
                color: #374151;
            }

            .chat-box {
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            .msg-row { display: flex; flex-direction: column; max-width: 78%; }
            .msg-left { align-self: flex-start; }
            .msg-right { align-self: flex-end; }
            .msg-center { align-self: center; max-width: 90%; text-align: center; }

            .msg-sender {
                font-size: 0.75rem;
                font-weight: 700;
                color: var(--muted);
                margin-bottom: 4px;
            }
            .msg-right .msg-sender { text-align: right; }

            .bubble {
                padding: 10px 14px;
                border-radius: 12px;
                font-size: 0.92rem;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .bubble-a {
                background: var(--agent-a-bg);
                border: 1px solid var(--agent-a-border);
                border-bottom-left-radius: 2px;
            }
            .bubble-b {
                background: var(--agent-b-bg);
                border: 1px solid var(--agent-b-border);
                border-bottom-right-radius: 2px;
            }
            .bubble-judge {
                background: var(--judge-bg);
                color: var(--judge-text);
                font-size: 0.82rem;
                font-weight: 600;
                border-radius: 999px;
                padding: 6px 16px;
            }

            .verdict-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 14px;
                padding: 16px 20px;
                background: #f9fafb;
                border-top: 1px solid var(--border);
            }
            @media (max-width: 700px) { .verdict-grid { grid-template-columns: 1fr; } }
            .verdict-card {
                background: #ffffff;
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 12px 14px;
            }
            .verdict-top {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }
            .payoff-pill {
                font-weight: 700;
                font-size: 0.85rem;
                background: #f3f4f6;
                padding: 2px 8px;
                border-radius: 6px;
            }
            .explanation-text {
                font-size: 0.86rem;
                color: #374151;
                margin: 0;
                white-space: pre-wrap;
            }
        </style>
        """

    def _get_js(self) -> str:
        """Return vanilla JS for interactive filtering by match outcome."""
        return """
        <script>
            function filterMatches(category, btn) {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                document.querySelectorAll('.match-card').forEach(card => {
                    if (category === 'ALL' || card.dataset.category === category) {
                        card.style.display = 'block';
                    } else {
                        card.style.display = 'none';
                    }
                });
            }
        </script>
        """

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

    def _render_conversation_bubbles(self, messages: list, id_a: str, id_b: str, first_id: str) -> tuple[str, str]:
        """
        Convert raw message history (stored from Agent A's perspective) into
        HTML chat bubbles and extract the system prompt separately.
        """
        system_prompt = ""
        bubbles_html = []

        init_tag_a = '<span class="badge badge-initiator">Initiator</span>' if first_id == id_a else ""
        init_tag_b = '<span class="badge badge-initiator">Initiator</span>' if first_id == id_b else ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = html.escape(content)
                continue

            # Stop rendering chat bubbles when the final secret decision prompt begins
            if role == "user" and content.startswith("[JUDGE]: Now the other agent can't listen to you"):
                break

            if role == "user" and content.startswith("[JUDGE]:"):
                clean_judge = html.escape(content.replace("[JUDGE]:", "", 1).strip())
                bubbles_html.append(f"""
                <div class="msg-row msg-center">
                    <div class="bubble bubble-judge">JUDGE: {clean_judge}</div>
                </div>
                """)
            elif role == "assistant":
                clean_a = html.escape(content.strip())
                bubbles_html.append(f"""
                <div class="msg-row msg-left">
                    <div class="msg-sender">Agent A (ID: {html.escape(str(id_a))}){init_tag_a}</div>
                    <div class="bubble bubble-a">{clean_a}</div>
                </div>
                """)
            elif role == "user" and content.startswith("[AGENT]:"):
                clean_b = html.escape(content.replace("[AGENT]:", "", 1).strip())
                bubbles_html.append(f"""
                <div class="msg-row msg-right">
                    <div class="msg-sender">Agent B (ID: {html.escape(str(id_b))}){init_tag_b}</div>
                    <div class="bubble bubble-b">{clean_b}</div>
                </div>
                """)

        return system_prompt, "\n".join(bubbles_html)

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

        # Determine filter category and header badge
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

        sys_prompt, chat_html = self._render_conversation_bubbles(
            rec.get("full_conversation", []), id_a, id_b, first_id
        )

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
                    <span class="match-sub"> &bull; {game_name} &bull; {timestamp}</span>
                </div>
                <div>{outcome_badge}</div>
            </div>
            {sys_block}
            <div class="chat-box">
                {chat_html}
            </div>
            <div class="verdict-grid">
                <div class="verdict-card">
                    <div class="verdict-top">
                        <strong>Agent A (ID: {html.escape(str(id_a))})</strong>
                        <div>
                            {self._action_badge(act_a)}
                            <span class="payoff-pill">Payoff: {pay_a}</span>
                        </div>
                    </div>
                    <p class="explanation-text">{expl_a}</p>
                </div>
                <div class="verdict-card">
                    <div class="verdict-top">
                        <strong>Agent B (ID: {html.escape(str(id_b))})</strong>
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
        filterable HTML document and saves it to the output directory.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.dataset.raw_records:
            print("No records available to generate the HTML report.")
            return ""

        cards_html = "\n".join(
            self._render_match_card(idx, rec)
            for idx, rec in enumerate(self.dataset.raw_records)
        )

        total = len(self.dataset.raw_records)
        valid = len(self.dataset.valid_df)
        err_rate = self.dataset.error_rate

        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simulation Conversations Report</title>
    {self._get_css()}
</head>
<body>
    <div class="container">
        <header>
            <h1>Socio-Economic Simulation &mdash; Conversations Inspector</h1>
            <div class="meta-bar">
                Source: <strong>{html.escape(self.dataset.file_path)}</strong> &bull;
                Total Matches: <strong>{total}</strong> &bull;
                Valid: <strong>{valid}</strong> &bull;
                Parsing Error Rate: <strong>{err_rate}%</strong>
            </div>
            <div class="filters">
                <button class="filter-btn active" onclick="filterMatches('ALL', this)">All ({total})</button>
                <button class="filter-btn" onclick="filterMatches('MUTUAL_COOP', this)">Mutual Cooperation</button>
                <button class="filter-btn" onclick="filterMatches('ASYMMETRIC', this)">Asymmetric (Betrayal)</button>
                <button class="filter-btn" onclick="filterMatches('MUTUAL_DEFECT', this)">Mutual Defection</button>
                <button class="filter-btn" onclick="filterMatches('ERROR', this)">Parsing Errors</button>
            </div>
        </header>
        <main>
            {cards_html}
        </main>
    </div>
    {self._get_js()}
</body>
</html>
"""
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_html)

        print(f"HTML conversation report saved to '{output_path}'.")
        return output_path
