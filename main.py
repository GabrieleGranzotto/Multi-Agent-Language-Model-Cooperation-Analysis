import argparse
from orchestrator import Orchestrator
from intel import DataSet, StatsEngine, Graphs, ChatReport

def main():
    parser = argparse.ArgumentParser(description="Run socio-political game theory simulation with LLM agents.")
    parser.add_argument("-n", "--num_rep", type=int, default=5, help="Number of repetitions to run the simulation.")
    parser.add_argument("-t", "--max_turns", type=int, default=5, help="Maximum number of turns for the conversation between agents.")
    parser.add_argument("-m", "--model_name", type=str, default="gemma4", help="Name of the LLM model to use for the agents.")
    parser.add_argument("-o", "--other_model_name", type=str, default=None, help="Name of the LLM model to use for the other agent.")
    parser.add_argument("-r", "--randomizer", action="store_true", help="Randomize the starting agent.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()
    orchestrator = Orchestrator(max_turns=args.max_turns, 
                                model_name=args.model_name, 
                                other_model_name=args.other_model_name, 
                                randomize_start=args.randomizer, 
                                verbose=args.verbose)
    print(f"Running simulation with {args.num_rep} repetitions, \nmax turns: {args.max_turns}, \nmodel: {args.model_name}, \nother model: {args.other_model_name}, \nrandom start: {args.randomizer}, \nverbose: {args.verbose}")
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
            orchestrator.reset()  # Reset the orchestrator for a new conversation if needed
        orchestrator.reset()  # Reset the orchestrator for a new conversation if needed
    orchestrator.save_response_history()  # Save the response history to a JSON file
    
    # Create a new dataset from the response history
    dataset = DataSet("data/response_history.json")

    # Run all analyses on the dataset and export results to the "reports" directory
    stats_engine = StatsEngine(dataset)
    stats_engine.run_all_analyses(export_dir="reports")
    
    # Generate graphs from the dataset
    graphs = Graphs(dataset, output_dir="reports")
    graphs.generate_all_graphs()

    # Generate HTML report
    chat_report = ChatReport(dataset, output_dir="reports")
    chat_report.generate_html_report()


if __name__ == "__main__":
    main()