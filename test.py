import json
import sys
import urllib.request

import anthropic
from dotenv import load_dotenv

# Reads ANTHROPIC_API_KEY from the .env file next to this script
load_dotenv()

MODEL = "claude-opus-5-5"


# =====================================================================
# 1. DEFINE THE CUSTOM TOOL (FUNCTION)
# =====================================================================
def get_crypto_price(coin_id: str) -> str:
    """
    Fetches the real-time price of a cryptocurrency from the CoinGecko API.
    Does not require any authorization or API keys.
    """
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    try:
        # Using standard urllib so no 'pip install requests' is needed
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            if coin_id in data and 'usd' in data[coin_id]:
                price = data[coin_id]['usd']
                return f"SUCCESS: The current price of {coin_id.upper()} is ${price:,} USD."
            return f"ERROR: Coin ID '{coin_id}' not found in API response."
    except Exception as e:
        return f"API Connection Error: {str(e)}"


# =====================================================================
# 2. DEFINE THE AI TOOL SPECIFICATION SCHEMA
# =====================================================================
# This tells Claude exactly what this tool does and what inputs it expects.
crypto_tool_schema = {
    "name": "get_crypto_price",
    "description": "Fetch the live market value price of a cryptocurrency in US Dollars (USD).",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "coin_id": {
                "type": "string",
                "description": "The CoinGecko coin ID: lowercase full name, e.g. 'bitcoin', 'ethereum', 'solana'. Map tickers like BTC or ETH to their full ID."
            }
        },
        "required": ["coin_id"],
        "additionalProperties": False
    }
}

TOOLS = {"get_crypto_price": get_crypto_price}


# =====================================================================
# 3. RUN THE AGENT TOOL-CALLING LOOP WITH CLAUDE
# =====================================================================
def run_agent(user_query: str):
    print("=" * 70)
    print(f"USER QUERY: \"{user_query}\"")
    print("=" * 70)

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_query}]

    while True:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system="You are a helpful assistant. Use the get_crypto_price tool for any cryptocurrency price question instead of guessing.",
            tools=[crypto_tool_schema],
            messages=messages,
            # If a safety classifier declines, the API retries on a fallback model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

        if response.stop_reason == "refusal":
            print("🤖 Claude declined this request.")
            return

        # Append the full content (not just text) so thinking blocks are preserved
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\n🤖 Claude called {block.name}({json.dumps(block.input)})")
            output = TOOLS[block.name](**block.input)
            print(f"-> Tool Output: {output}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
                "is_error": not output.startswith("SUCCESS"),
            })
        # All results go back in a single user message
        messages.append({"role": "user", "content": tool_results})

    print("\n" + "-" * 70)
    print("🤖 FINAL CHATBOT RESPONSE:")
    print("".join(b.text for b in response.content if b.type == "text"))
    print("-" * 70)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "Hey, can you check the current price of BTC and ETH?"
    run_agent(query)
