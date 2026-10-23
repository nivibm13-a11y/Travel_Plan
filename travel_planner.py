
import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# STEP 1: Load environment variables from .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL", "openai/gpt-4o-mini")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is missing. Please check your .env file."
    )

# STEP 2: Configure the language model
llm = LLM(
    model=model_name,
    api_key=api_key
)

# STEP 3: Get city input from the user
city = input("Which city would you like to visit? ").strip()

if not city:
    raise ValueError("Please enter a city name.")

# STEP 4: Create the Travel Planner Agent
planner = Agent(
    role="Experienced Travel Planner",
    goal=(
        "Recommend exactly three interesting places to visit "
        "in the city selected by the user."
    ),
    backstory=(
        "You are an experienced travel planner who helps travellers "
        "discover interesting attractions. You provide concise and "
        "useful recommendations, explain why each place is worth "
        "visiting, and give one practical visiting tip."
    ),
    llm=llm,
    verbose=True
)

# STEP 5: Create one Task
travel_task = Task(
    description=(
        "Create a travel guide for {city}. Recommend exactly "
        "3 distinct places to visit in that city. For each place, "
        "include its name, a short description, and one practical "
        "visiting tip. Do not recommend a fourth place."
    ),
    expected_output=(
        "Exactly 3 numbered recommendations. Each must include "
        "the place name, a short description, and one visiting tip. "
        "Do not include an introduction, conclusion, or extra places."
    ),
    agent=planner
)

# STEP 6: Create the Crew
crew = Crew(
    agents=[planner],
    tasks=[travel_task],
    verbose=True
)

# STEP 7: Run the Crew
result = crew.kickoff(inputs={"city": city})

# STEP 8: Prepare the output text
travel_plan = str(result)

# STEP 9: Save output to a separate text file
output_file = Path(__file__).resolve().parent / "travel_plan.txt"

with open(output_file, "w", encoding="utf-8") as file:
    file.write(f"TRAVEL PLAN FOR {city.upper()}\n")
    file.write("=" * 40 + "\n\n")
    file.write(travel_plan)
    file.write("\n")

# STEP 10: Display output in the terminal
print("\n===== YOUR TRAVEL PLAN =====\n")
print(travel_plan)

print(f"\nTravel plan saved successfully to:\n{output_file}")
