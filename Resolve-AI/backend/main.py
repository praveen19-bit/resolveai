from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from tool import agent


myapp = FastAPI(title="ResolveAI API", version="1.0.0")
 
# Allow requests from your HTML frontend (any origin in dev)
myapp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # lock to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
@myapp.get("/chat")
def chat(complaint:str):

        
    result = agent.invoke({
        "messages": [   # ← FIX HERE
            {
                "role": "user",
                "content": f"""You are ResolveAI, a smart campus infrastructure complaint classifier{complaint}.
                using given tools to analysis the complaint.
                "title": ""
  "description" : ""
  "category": "",
  "priority": "",
  "department": "",
  "location" : ""

  RULES:
- Return ONLY the JSON. No explanation, no extra text, no markdown.
- Never wrap the JSON in code fences like ```json
- If unsure about category, use "Other"
- If unsure about priority, use "Medium"
If an image is provided, use it to better determine the priority and category.
If no image is provided, classify based on text only.
  """
            }
        ]
    })

    return result["messages"][-1].content

if __name__ == "__main__":
    
    uvicorn.run(myapp)