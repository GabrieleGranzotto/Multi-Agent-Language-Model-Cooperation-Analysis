import argparse
from orchestrator import Orchestrator
from intel import generate_all_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run socio-political game theory simulation with LLM agents.")
    parser.add_argument("-n", "--num_rep", type=int, default=5, help="Number of repetitions to run the simulation.")
    parser.add_argument("-t", "--max_turns", type=int, default=5, help="Maximum number of turns for the conversation between agents.")
    parser.add_argument("-m", "--model_name", type=str, default="gemma4", help="Name of the LLM model to use for the agents.")
    parser.add_argument("-o", "--other_model_name", type=str, default=None, help="Name of the LLM model to use for the other agent.")
    parser.add_argument("-r", "--randomizer", action="store_true", help="Randomize the starting agent.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("-d", "--debug", action="store_true", help="Log sanity-check rejections to console and data/sanity_debug.jsonl.")
    parser.add_argument("-f", "--file", type=str, default="data/response_history.json", help="Path to the JSON file where simulation history is saved.")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory where reports, graphs, and summaries are exported.")

    args = parser.parse_args()
    orchestrator = Orchestrator(
        max_turns=args.max_turns,
        model_name=args.model_name,
        other_model_name=args.other_model_name,
        randomize_start=args.randomizer,
        verbose=args.verbose,
        debug=args.debug,
        history_path=args.file,
    )
    print(f"Running simulation with {args.num_rep} repetitions, \nmax turns: {args.max_turns}, \nmodel: {args.model_name}, \nother model: {args.other_model_name}, \nrandom start: {args.randomizer}, \nverbose: {args.verbose}, \ndebug: {args.debug}")
    for i in range(args.num_rep):
        print(f"\n------- Simulation Run {i + 1}... -------\n")
        try:
            orchestrator.run()
            orchestrator.result()
            print("-----------------------------------\n")
            print("-----------------------------------\n")
        except Exception as e:
            print(f"[ERROR] occurred during Run {i + 1}: {e}")
        finally:
            orchestrator.reset()
    orchestrator.save_response_history()

    # Generate the report and explanations summary after all repetitions are completed
    generate_all_pipeline(
        json_path=args.file,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()