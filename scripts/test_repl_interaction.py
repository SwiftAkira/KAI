import subprocess
import time
import sys

def run_test():
    """
    Runs the REPL as a subprocess and interacts with it to test responses.
    """
    command = [sys.executable, "-m", "pressure_agi.demos.repl"]
    
    print(f"Starting test with command: {' '.join(command)}")
    
    # Start the REPL process
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1, # Line-buffered
        encoding='utf-8'
    )

    time.sleep(3) # Give the agent time to initialize

    questions = [
        "hello",
        "im happy",
        "i love it",
        "im sad",
        "i hate this",
        "amazing",
        "terrible"
    ]
    
    conversation = []

    # Read initial output
    try:
        for _ in range(5): # Read the startup messages
            line = process.stdout.readline()
            if not line:
                break
            print(line.strip())
            conversation.append(line.strip())
    except Exception as e:
        print(f"Error reading initial output: {e}")


    for q in questions:
        print(f"> {q}")
        conversation.append(f"> {q}")
        process.stdin.write(q + '\\n')
        process.stdin.flush()
        
        # Give agent time to respond
        time.sleep(1) 

        # Read the agent's response
        response = process.stdout.readline().strip()
        print(response)
        conversation.append(response)

    # Exit the repl
    print("> exit")
    process.stdin.write("exit\\n")
    process.stdin.flush()
    time.sleep(1)

    # Capture remaining output
    stdout, stderr = process.communicate()
    if stdout:
        print(stdout)
    if stderr:
        print("Errors:")
        print(stderr)
        
    process.wait()
    print("\\n--- Test Complete ---")
    
if __name__ == "__main__":
    run_test() 