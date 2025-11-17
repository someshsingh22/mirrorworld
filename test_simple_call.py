"""Simple test script to make a basic LLM call."""

from src.utils.llm import create_azure_llm


def main():
    """Test a simple LLM call by saying hi."""
    print("\n=== Testing Simple Azure OpenAI Call ===\n")
    
    try:
        # Initialize LLM
        print("Creating Azure OpenAI client...")
        model = create_azure_llm(
            model="azure/gpt-4o",
            temperature=0.7,
        )
        print("✓ Client created successfully\n")
        
        # Make a simple call
        print("Sending message: 'Hi, can you respond with a friendly greeting?'\n")
        response = model.invoke([
            {"role": "user", "content": "Hi, can you respond with a friendly greeting?"}
        ])
        
        print("✓ Response received:")
        print(f"  {response.content}\n")
        
        print("=== Test Successful! ===\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {str(e)}\n")
        print("Run 'python scripts/check_azure_credentials.py' for diagnostics.\n")


if __name__ == "__main__":
    main()

