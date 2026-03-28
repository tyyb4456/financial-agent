from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from crews import crews  # Import the crew definitions
import os
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to "*" if testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CrewRequest(BaseModel):
    crew_name: str
    inputs: Dict[str, Any]

# Store the last response in memory
latest_response: Optional[Dict[str, Any]] = None

@app.get("/")
def read_root():
    if latest_response is None:
        return {"message": "FastAPI is running! No response available yet."}
    return latest_response

@app.post("/execute_crew/")
async def execute_crew(request: CrewRequest):
    global latest_response  # Modify global variable

    # Check if the crew exists
    if request.crew_name not in crews:
        raise HTTPException(status_code=404, detail="Crew not found")

    crew_definition = crews[request.crew_name]

    # Validate required inputs
    missing_keys = [
        key for key in crew_definition.required_inputs.keys() if key not in request.inputs
    ]
    if missing_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required inputs: {', '.join(missing_keys)}",
        )

    # Filter inputs: include only valid required and optional keys
    valid_inputs = {k: v for k, v in request.inputs.items() if k in crew_definition.required_inputs or k in crew_definition.optional_inputs}

    # Execute the selected crew
    output = crew_definition.crew.kickoff(inputs=valid_inputs)

    latest_response = {"crew_name": request.crew_name, "output": output}  # Store response

    return latest_response

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
