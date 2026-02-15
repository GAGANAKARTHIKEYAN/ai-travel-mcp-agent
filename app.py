import streamlit as st
import os
import requests
import re
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# =====================================
# Load Environment Variables
# =====================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# =====================================
# Streamlit Setup
# =====================================
st.set_page_config(page_title="AI Travel Planner - MCP Agent", layout="wide")
st.title("✈️ AI Travel Planner — LangGraph MCP Agent")
st.markdown("LLM + Real Weather Forecast + Tool Orchestration")
st.divider()

# =====================================
# Utility: Extract City from Input
# =====================================
def extract_city(text):
    match = re.search(r"to\s+([A-Za-z\s]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

# =====================================
# WEATHER TOOL (Current + Forecast)
# =====================================
@tool
def weather_tool(city: str) -> str:
    """Get current weather and 5-day forecast of a city."""

    current_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"

    try:
        current_data = requests.get(current_url).json()
        forecast_data = requests.get(forecast_url).json()

        if "main" not in current_data:
            return "Weather data unavailable."

        temp = current_data["main"]["temp"]
        condition = current_data["weather"][0]["description"]

        forecast_text = ""
        for item in forecast_data["list"][:5]:
            date = item["dt_txt"]
            f_temp = item["main"]["temp"]
            f_desc = item["weather"][0]["description"]
            forecast_text += f"{date} → {f_temp}°C, {f_desc}\n"

        return f"""
Current Weather in {city}:
Temperature: {temp}°C
Condition: {condition}

5-Day Forecast:
{forecast_text}
"""
    except:
        return "Weather data unavailable."

# =====================================
# FLIGHT TOOL (Simulated)
# =====================================
@tool
def flight_tool(city: str) -> str:
    """Get flight options to a city."""
    return f"""
Flight Options to {city}:
- Emirates | Non-stop | $750
- Qatar Airways | 1 Stop | $680
- Lufthansa | Non-stop | $820
"""

# =====================================
# HOTEL TOOL (Simulated)
# =====================================
@tool
def hotel_tool(city: str) -> str:
    """Get hotel options in a city."""
    return f"""
Hotel Options in {city}:
- Grand Palace Hotel | 4.5⭐ | $180/night
- City Lights Resort | 4.2⭐ | $150/night
- Heritage Stay Inn | 4.0⭐ | $120/night
"""

tools = [weather_tool, flight_tool, hotel_tool]

# =====================================
# LLM Setup
# =====================================
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

agent = create_react_agent(llm, tools)

# =====================================
# UI
# =====================================
col1, col2 = st.columns([1, 2])

with col1:
    user_input = st.text_area(
        "Enter Travel Request:",
        placeholder="Plan a 3-day trip to London in May"
    )
    generate = st.button("Generate Plan")

with col2:
    if generate and user_input:

        city = extract_city(user_input)

        if not city:
            st.warning("Could not detect city. Please use format like: 'Trip to London'")
        else:
            with st.spinner("Generating travel plan..."):

                structured_prompt = f"""
You are a professional AI Travel Planner.

The destination city is STRICTLY: {city}

DO NOT change the city.
DO NOT assume another city.

User Request:
{user_input}

Provide:

1. One paragraph about {city}'s cultural & historical significance.
2. Current Weather and 5-Day Forecast (use weather_tool).
3. Travel Dates.
4. Flight Options (use flight_tool).
5. Hotel Options (use hotel_tool).
6. Day-wise itinerary.

Return clean Markdown only.
"""

                response = agent.invoke(
                    {"messages": [("user", structured_prompt)]}
                )

                last_message = response["messages"][-1].content

                if isinstance(last_message, list):
                    final_output = "".join(
                        item.get("text", "")
                        for item in last_message
                        if isinstance(item, dict)
                    )
                else:
                    final_output = last_message

                st.markdown(final_output)

    elif generate:
        st.warning("Please enter a travel request.")

st.divider()
st.caption("LangGraph MCP Travel Agent | Academic Project")
