# Can Language Create Cooperation? Natural-Language Negotiation in Strategic Multi-Agent LLM Interactions

An experimental multi-agent framework investigating how pre-play natural-language communication (*cheap talk*) shapes cooperation, equilibrium selection, and strategic deception among Large Language Model (LLM) agents in classic normal-form games.

Built on a fully local inference stack (Ollama), the project pairs a two-phase negotiation-and-commitment protocol with an automated behavioral and linguistic analytics pipeline.

## Repository Structure

```text
MALLM/
├── data
│   ├── games.json              # The file where are defined all playable games
│   └── results.json            # The file where all the results are stored
├── orchestrator.py             # Two-phase game loop (diplomatic dialogue + isolated judgment)
├── game_env.py                 # Normal-form payoff matrices (Prisoner's Dilemma, Stag Hunt, etc.)
├── llm_agent.py                # Local Ollama wrapper with structured JSON parsing & fallbacks
├── intel_reports/              # Auto-generated CSV datasets, LaTeX tables, and 300 DPI figures
├── main.py                     # Multi-episode simulation runner
├── intel.py                    # Standalone statistical, linguistic, and visualization CLI
└── README.md
```

## Core Architecture

### 1. Strategic Game Environments

Agents interact in canonical two-player normal-form games characterized by tension between individual incentives and collective welfare (such as the Prisoner's Dilemma and Stag Hunt), where joint outcomes map to `{CC, CD, DC, DD}` and corresponding payoff matrices.

### 2. Two-Phase Interaction Protocol

Each simulated match is coordinated by an `Orchestrator` enforcing strict information boundaries across two sequential stages:

* **Phase 1 — Diplomatic Negotiation (Cheap Talk):** Prior to committing to a move, both LLM agents engage in an unconstrained natural-language dialogue over `max_turns` exchanges, allowing them to propose deals, signal intentions, or attempt persuasion.
* **Phase 2 — Isolated Decision Elicitation (`[JUDGE]`):** Once negotiation ends, each agent is queried independently in an isolated context to commit to a final action (`answer`) and disclose its private internal rationale (`explanation`) via a strict JSON schema.

### 3. Local Inference & Fault-Tolerant Parsing

All experiments run locally via **Ollama** (e.g., `gemma4`), removing external API dependencies and rate limits. To guarantee uninterrupted multi-episode runs, the client enforces structured JSON extraction with automatic validation and fallback recovery against malformed outputs.

## Key Extensions Beyond Reference Literature

While inspired by recent behavioral game-theory benchmarks for LLMs (e.g., *Akata et al., 2025*), this implementation introduces several methodological extensions:

1. **Unconstrained Pre-Play Cheap Talk:** Extends standard matrix-only or action-history prompts by adding a multi-turn natural-language negotiation phase before action selection, directly testing whether non-binding dialogue shifts play from Nash defection toward Pareto-optimal cooperation.
2. **Private Rationale Auditing & Deception Detection:** Introduces structured `[JUDGE]` elicitation of private `explanation` fields alongside public dialogue transcripts, making it possible to distinguish genuine cooperative alignment from *strategic deception* (promising cooperation during diplomacy while defecting in private).
3. **Linguistic-Behavioral Correlation Pipeline (`intel.py`):** Goes beyond aggregate payoff reporting by measuring message verbosity (word count during negotiation) and fitting logistic regressions to quantify how communication length and rhetorical patterns correlate with cooperative vs. defective decisions.
4. **Automated Research Artifact Generation:** A standalone post-processing CLI (`intel.py`) automatically parses the latest simulation logs to export CSV summaries, publication-ready `booktabs` LaTeX tables, and high-resolution plots.

## Usage

### Prerequisites

Ensure [Ollama](https://ollama.com/) is installed and running locally with your target model pulled:

```bash
ollama pull gemma4
pip install numpy pandas matplotlib seaborn scipy requests tqdm
```

### Running Simulations

Execute the multi-game simulation suite:

```bash
python main.py
```

This runs the two-phase protocol across the configured games and episodes, saving detailed transcripts, private rationales, and outcome logs as timestamped JSON files.

### Generating Analytics, Tables & Plots

Run the post-processing pipeline on the most recent simulation output (or specify a custom file):

```bash
# Automatically process the latest results JSON
python intel.py

# Or specify an explicit input file and output folder
python intel.py --input path/to/results.json --output-dir ./intel_reports/custom_run
```

## Key Analytical Highlights

* **Equilibrium Selection via Cheap Talk:** Pre-play natural-language negotiation substantially increases mutual cooperation (`CC`) in coordination and social-dilemma settings compared to silent baselines, demonstrating that LLM agents leverage dialogue to align expectations.
* **Alignment vs. Strategic Deception:** Cross-referencing diplomatic transcripts against private `[JUDGE]` explanations reveals that cooperation is not uniform—under strong temptation payoffs, agents occasionally weaponize persuasive dialogue to induce opponent cooperation before defecting.
* **Communication Verbosity Dynamics:** Quantitative analysis of dialogue length highlights distinct behavioral regimes between cooperative consensus-building and protracted, deceptive, or non-committal exchanges.

## References

* Farrell, J., & Rabin, M. (1996). *Cheap Talk*. Journal of Economic Perspectives, 10(3), 103–118.
* Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. UIST.
* Akata, E., Schulz, L., Coda-Forno, J., Oh, S. J., Bethge, M., & Schulz, E. (2025). *Playing Repeated Games with Large Language Models*. Nature Human Behaviour.
