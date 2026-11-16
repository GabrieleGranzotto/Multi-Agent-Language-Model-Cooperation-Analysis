import os, json
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
            color="black",
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
        sns.boxplot(data=long_df, x="action", y="words", palette="Pastel1", ax=axes[0], width=0.45)
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
