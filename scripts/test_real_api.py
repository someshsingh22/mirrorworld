"""Test with real Azure OpenAI API."""

import subprocess
import sys


def test_real_api():
    """Test agent with real Azure API using pre-programmed responses."""
    print("\n=== Testing Agent with Real Azure API ===\n")

    # Pre-programmed responses
    responses = ["yes\n", "no\n", "yes\n"]

    print(f"Running 3-question interview with responses: {[r.strip() for r in responses]}")
    print("Using config: configs/agent_config_test.yaml\n")

    # Run the interview script
    process = subprocess.Popen(
        [sys.executable, "scripts/run_interview.py", "configs/agent_config_test.yaml"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr = process.communicate(input="".join(responses))

    print(stdout)

    if stderr:
        print("STDERR:", stderr)

    if process.returncode == 0:
        print("\n✅ Real API test passed!")
        return True
    else:
        print("\n❌ Real API test failed")
        return False


if __name__ == "__main__":
    success = test_real_api()
    sys.exit(0 if success else 1)
