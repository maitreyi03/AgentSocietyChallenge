#!/usr/bin/env python3
"""
Quick runner script for the Advanced Mood Recommendation Agent
Handles environment setup and provides easy configuration
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

def check_environment():
    """Check if environment is properly configured"""
    print("=" * 70)
    print("ADVANCED MOOD RECOMMENDATION AGENT - ENVIRONMENT CHECK")
    print("=" * 70)
    
    # Check .env file
    env_path = Path(__file__).parent / '.env'
    if not env_path.exists():
        print("❌ ERROR: .env file not found!")
        print("   Please create .env file with your GEMINI_API_KEY")
        print("   You can copy .env.example to .env and add your key")
        return False
    
    # Load environment
    load_dotenv()
    
    # Check API key
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('api_key')
    if not api_key or api_key == 'YOUR_API_KEY_HERE' or api_key == 'your_gemini_api_key_here':
        print("❌ ERROR: GEMINI_API_KEY not configured!")
        print("   Please edit .env file and add your actual API key")
        print("   Get your key from: https://aistudio.google.com/app/apikey")
        return False
    
    print("✅ API Key configured")
    
    # Check data directory
    data_dir = Path.home() / 'data' / 'processed-data'
    if not data_dir.exists():
        print(f"⚠️  WARNING: Data directory not found at {data_dir}")
        print("   Make sure your data is in the correct location")
        print("   Or adjust the data_dir path in advanced_mood_agent.py")
    else:
        print(f"✅ Data directory found: {data_dir}")
    
    # Check task directory
    task_dir = Path(__file__).parent / 'example' / 'track2' / 'goodreads' / 'tasks'
    if not task_dir.exists():
        print(f"❌ ERROR: Task directory not found at {task_dir}")
        return False
    
    print(f"✅ Task directory found: {task_dir}")
    
    return True


def run_agent(num_tasks=50, max_workers=5, task_set="goodreads"):
    """Run the advanced mood agent"""
    
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("STARTING AGENT EVALUATION")
    print("=" * 70)
    print(f"Task Set: {task_set}")
    print(f"Number of Tasks: {num_tasks}")
    print(f"Max Workers: {max_workers}")
    print("=" * 70 + "\n")
    
    # Import here after environment check
    from advanced_mood_agent import AdvancedMoodRecAgent
    from websocietysimulator.simulator import Simulator
    from websocietysimulator.llm.llm import GeminiLLM
    
    load_dotenv()
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('api_key')
    
    # Configuration
    data_dir = os.path.expanduser("~/data/processed-data")
    
    # Initialize Simulator
    print("Initializing simulator...")
    simulator = Simulator(data_dir=data_dir, device="auto", cache=True)
    
    # Load tasks and ground truth
    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track2/{task_set}/tasks",
        groundtruth_dir=f"./example/track2/{task_set}/groundtruth"
    )
    
    # Set agent and LLM
    simulator.set_agent(AdvancedMoodRecAgent)
    simulator.set_llm(GeminiLLM(api_key=api_key))
    
    # Run simulation
    print(f"\nRunning simulation on {num_tasks} tasks...")
    print("This may take several minutes depending on the number of tasks...\n")
    
    try:
        agent_outputs = simulator.run_simulation(
            number_of_tasks=num_tasks,
            enable_threading=True,
            max_workers=max_workers
        )
        
        # Evaluate
        print("\n" + "=" * 70)
        print("EVALUATING RESULTS")
        print("=" * 70)
        
        evaluation_results = simulator.evaluate()
        
        # Save results
        output_file = f'./evaluation_results_advanced_{task_set}.json'
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        # Display results
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        metrics = evaluation_results.get('metrics', {})
        print(f"\n📊 Hit Rates:")
        print(f"   HR@1: {metrics.get('top_1_hit_rate', 0):.4f} ({metrics.get('top_1_hit_rate', 0)*100:.2f}%)")
        print(f"   HR@3: {metrics.get('top_3_hit_rate', 0):.4f} ({metrics.get('top_3_hit_rate', 0)*100:.2f}%)")
        print(f"   HR@5: {metrics.get('top_5_hit_rate', 0):.4f} ({metrics.get('top_5_hit_rate', 0)*100:.2f}%)")
        print(f"   Average HR: {metrics.get('average_hit_rate', 0):.4f} ({metrics.get('average_hit_rate', 0)*100:.2f}%)")
        
        print(f"\n📈 Task Statistics:")
        print(f"   Total Tasks: {metrics.get('total_scenarios', 0)}")
        print(f"   Top-1 Hits: {metrics.get('top_1_hits', 0)}")
        print(f"   Top-3 Hits: {metrics.get('top_3_hits', 0)}")
        print(f"   Top-5 Hits: {metrics.get('top_5_hits', 0)}")
        
        # Compare with baseline
        baseline_hr1 = 0.11
        baseline_hr3 = 0.2925
        baseline_hr5 = 0.4475
        
        improvement_hr1 = ((metrics.get('top_1_hit_rate', 0) - baseline_hr1) / baseline_hr1 * 100) if baseline_hr1 > 0 else 0
        improvement_hr3 = ((metrics.get('top_3_hit_rate', 0) - baseline_hr3) / baseline_hr3 * 100) if baseline_hr3 > 0 else 0
        improvement_hr5 = ((metrics.get('top_5_hit_rate', 0) - baseline_hr5) / baseline_hr5 * 100) if baseline_hr5 > 0 else 0
        
        print(f"\n📊 Improvement vs Baseline:")
        print(f"   HR@1: {improvement_hr1:+.2f}%")
        print(f"   HR@3: {improvement_hr3:+.2f}%")
        print(f"   HR@5: {improvement_hr5:+.2f}%")
        
        print(f"\n💾 Results saved to: {output_file}")
        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETE!")
        print("=" * 70 + "\n")
        
        return evaluation_results
        
    except Exception as e:
        print(f"\n❌ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Advanced Mood Recommendation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 10 tasks
  python run_agent.py --tasks 10 --workers 2
  
  # Medium test with 50 tasks
  python run_agent.py --tasks 50 --workers 5
  
  # Full evaluation with 400 tasks
  python run_agent.py --tasks 400 --workers 10
        """
    )
    
    parser.add_argument(
        '--tasks',
        type=int,
        default=50,
        help='Number of tasks to run (default: 50)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of parallel workers (default: 5)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='goodreads',
        choices=['goodreads', 'amazon', 'yelp'],
        help='Dataset to use (default: goodreads)'
    )
    
    args = parser.parse_args()
    
    run_agent(
        num_tasks=args.tasks,
        max_workers=args.workers,
        task_set=args.dataset
    )


if __name__ == "__main__":
    main()
