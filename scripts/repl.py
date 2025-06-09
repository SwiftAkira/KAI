import argparse
import readline
import torch
import atexit
from pressure_agi.engine import Agent, TextAdapter, LOG
from pressure_agi.engine.utils import TerminalSwitch, Dashboard

# Define a function to save memory on exit
def save_agent_memory(agent):
    """Saves the agent's memory to disk."""
    agent.save_memory()
    LOG.info("Agent memory saved.")

def main():
    parser = argparse.ArgumentParser(description="Run the Pressure AGI agent in REPL mode.")
    parser.add_argument("--dashboard", action="store_true", help="Enable the dashboard view.")
    args = parser.parse_args()

    # Initialize agent and its components
    agent = Agent.from_config('config.yaml')
    text_adapter = TextAdapter()

    # Register the save_memory function to be called on exit
    atexit.register(save_agent_memory, agent)

    if args.dashboard:
        # Advanced dashboard view
        with TerminalSwitch() as ts:
            dashboard = Dashboard(ts.stdscr)
            dashboard.header()
            LOG.info("Dashboard enabled. Press Ctrl+C to exit.")

            turn = 0
            last_entropy = 0.0
            
            while True:
                try:
                    dashboard.update(agent, turn)
                    text = dashboard.get_input()

                    if text is None: # Exit condition
                        break

                    # Let the agent think and decide on an action
                    action = agent.think(text, env=None) # No environment in REPL mode

                    # Translate the action to a response for the user
                    response = text_adapter.agent_to_user(action)
                    dashboard.set_agent_response(response)
                    
                    # Decay the resonance gain if entropy is increasing
                    if agent.critic.last_entropy > last_entropy:
                        agent.memory.resonance_gain *= 0.99
                    last_entropy = agent.critic.last_entropy

                    turn += 1

                except KeyboardInterrupt:
                    LOG.info("Exiting...")
                    break
    else:
        # Simple chat interface
        print("Agent is running in chat mode. Press Ctrl+C to exit.")
        last_entropy = 0.0
        while True:
            try:
                text = input("You: ")

                # Let the agent think and decide on an action
                action = agent.think(text, env=None) # No environment in REPL mode

                # Translate the action to a response for the user
                response = text_adapter.agent_to_user(action)
                
                # Always print the agent's response
                print(f"Agent: {response}")

                # Decay the resonance gain if entropy is increasing
                if agent.critic.last_entropy > last_entropy:
                    agent.memory.resonance_gain *= 0.99
                last_entropy = agent.critic.last_entropy

            except (KeyboardInterrupt, EOFError):
                LOG.info("Exiting...")
                break

if __name__ == "__main__":
    main() 