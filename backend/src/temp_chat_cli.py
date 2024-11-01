import requests
import json

def chat():
    session_id = "test123"
    
    # First, create a user profile
    requests.post(
        'http://127.0.0.1:8000/users/',
        params={"session_id": session_id}
    )
    
    print("Career Path AI Assistant (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        # Get user input
        query = input("\nYou: ")
        if query.lower() == 'quit':
            break
            
        # Send request to API
        response = requests.post(
            'http://127.0.0.1:8000/chat',  # Use the chat endpoint instead
            params={
                "message": query,
                "session_id": session_id
            }
        )
        
        # Print full response for debugging
        print("\nStatus Code:", response.status_code)
        print("Full Response:", response.text)
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("\nAssistant:", result.get('response', 'No response found'))
            except json.JSONDecodeError:
                print("\nError: Could not parse JSON response")
        else:
            print("\nError:", response.status_code, response.text)

if __name__ == "__main__":
    chat()