import json
import time
import os
import requests
from rag.retrieval import retrieve_and_answer
from orchestration.error_handlers import LLMTimeoutError, VectorDBError, OutOfDomainError

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
WEATHER_PATH = os.path.join(BASE_DIR, "data", "weather.json")
MARKET_PATH = os.path.join(BASE_DIR, "data", "market.json")

def fetch_mock_weather_api(location: str) -> dict:
    time.sleep(0.5)
    with open(WEATHER_PATH, "r") as f:
        data = json.load(f)
    for key in data.keys():
        if key in location.lower():
            return {"status": 200, "data": data[key]}
    return {"status": 404, "data": "Location not found"}

def fetch_mock_market_api(crop: str) -> dict:
    time.sleep(0.5)
    with open(MARKET_PATH, "r") as f:
        data = json.load(f)
    for key in data.keys():
        if key in crop.lower():
             return {"status": 200, "data": {key: data[key]}}
    return {"status": 404, "data": "Crop price unavailable"}

def route_query(english_query: str, session_history: str) -> dict:
    query_lower = english_query.lower()
    
    if "weather" in query_lower:
        api_response = fetch_mock_weather_api(query_lower)
        if api_response["status"] == 200:
            return {"answer": f"Weather API Data: {json.dumps(api_response['data'])}", "sources": ["Mock Weather API"], "context_used": []}
        else:
            return {"answer": "Weather data currently unavailable for this region.", "sources": [], "context_used": []}
            
    if "price" in query_lower or "market" in query_lower:
        api_response = fetch_mock_market_api(query_lower)
        if api_response["status"] == 200:
            return {"answer": f"Market API Data: {json.dumps(api_response['data'])}", "sources": ["Mock Mandi API"], "context_used": []}
        else:
            return {"answer": "Market information unavailable for this crop.", "sources": [], "context_used": []}

    if len(english_query.split()) < 3 and session_history == "No prior context.":
        return {"answer": "Please specify your question.", "sources": [], "context_used": []}

    try:
        return retrieve_and_answer(english_query, session_history)
    except requests.exceptions.ConnectionError:
        raise  # Bubble up to trigger the No Internet UI state
    except OutOfDomainError:
        raise
    except TimeoutError:
        raise LLMTimeoutError()
    except Exception as e:
        if "timeout" in str(e).lower() or "503" in str(e) or "504" in str(e):
            raise LLMTimeoutError()
        raise VectorDBError(str(e))
