from exa_py import Exa

exa = Exa(api_key="8fe70828-98a0-4195-bb8d-884227288beb")

result = exa.search(
  "PM working in FAANG with 1000 connections",
  num_results=30,
  category="linkedin profile",
  type="neural",
  use_autoprompt=True
) 
print(result)

# Get the URLs from the search results
urls = [result.results[i].url for i in range(len(result.results))]

# Fetch the contents for all URLs
contents = exa.get_contents(urls)

# Print each profile's content
for i, content in enumerate(contents.results):
    print(f"\n--- Profile {i+1} ---")
    print(content.text)

