from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
import os , base64 , faiss 
from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(
    groq_api_key= os.getenv("GROQ_APIKEY"),
    temperature=0 ,
    model_name="llama-3.3-70b-versatile"
    )

vision_llm = ChatGroq(
    groq_api_key= os.getenv("GROQ_APIKEY"),
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"  # image + text
)


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@tool
def image_complaint_analyzer(image_path: str) -> str:
    """Analyze complaint from an image (broken device, wifi screenshot, etc.)"""

    base64_image = encode_image(image_path)

    IMAGE_ANALYSIS_SYSTEM_PROMPT = f"""
You are ResolveAI Vision — a specialized campus infrastructure complaint analyzer.
Your job is to examine the provided image and extract every detail relevant to
reporting and resolving the infrastructure issue shown.

{base64_image}

ANALYSIS FRAMEWORK — examine the image for ALL of these:
 
1. WHAT IS THE ISSUE?
   - Identify the exact problem visible (e.g. burst pipe, cracked ceiling,
     broken projector, flooded floor, exposed wiring, smashed furniture,
     no-signal screenshot, pest infestation, overflowing bin, etc.)
   - Note ALL issues visible — sometimes there are secondary problems.
 
2. HOW SEVERE IS IT?
   CRITICAL  → Immediate safety risk (exposed wires, flooding, fire hazard,
               structural collapse, gas leak, complete blackout)
   HIGH      → Major disruption (room unusable, equipment destroyed, large leak,
               wifi/internet completely down shown via screenshot)
   MEDIUM    → Partial problem (one device broken, minor leak, partial damage,
               slow network screenshot, AC not cooling)
   LOW       → Cosmetic / minor (chipped paint, minor stain, small crack,
               worn furniture, dim light)
 

 
4. WHAT IS VISIBLE IN THE SCENE?
   - Location clues: room numbers, signage, labels on equipment, block letters,
     notice boards, door numbers visible in the image
   - Equipment identifiers: model numbers, serial plates, asset tags
   - Scale of damage: small/contained vs widespread
   - Environmental context: classroom, lab, corridor, hostel, washroom, outdoor
 
5. WHAT IMMEDIATE ACTION IS NEEDED?
   - Suggest ONE specific first-response action for the maintenance team
     (e.g. "Isolate power to the affected circuit immediately",
      "Place warning cones around the wet area",
      "Replace the projector lamp — model XL-2400",
      "Call plumber to shut main valve before water spreads further")
 
6. EVIDENCE QUALITY
   - Is the image clear enough for a work order? (clear / blurry / partial)
   - What is MOST important visible detail that confirms the severity?
 
STRICT RULES:

- Describe ONLY what you can actually SEE. Never invent details.
- If something is not visible, write "not visible" or "not identifiable".
- If the image is a SCREENSHOT (error message, wifi status, speed test),
  treat the screen content as the evidence.
- If multiple issues are visible, classify by the MOST SEVERE one.
- priority must be upgraded to Critical if safety_hazard is true.
-Give output within 5-6 line and without any lose of content.
- Return ONLY valid text format.

"""

    response = vision_llm.invoke(IMAGE_ANALYSIS_SYSTEM_PROMPT)

    return response.content

#print(image_complaint_analyzer.run("test.jpg"))


@tool
def text_complaint_analyzer(complaint: str) -> str:
    """
    Analyze complaint and return category and priority
    """

    TEXT_ANALYSIS_SYSTEM_PROMPT =f"""You are ResolveAI, a smart campus infrastructure complaint classifier.
complaint:{complaint}
When given a complaint, you must return ONLY a JSON object with exactly these fields:

  "title": ""
  "description" : ""
  "category": "",
  "priority": "",
  "department": "",
  "location" : ""


FIELD DESCRIPTION

TITLE:
Short title describing the issue.

CATEGORY:
Type of infrastructure issue.

PRIORITY must be one of:
- Critical → safety hazard, exam disruption, complete outage → fix in 2 hours
- High     → classroom/lab unusable, water leak, wifi down  → fix in 6 hours
- Medium   → partial issue, AC not cooling, one device broken → fix in 12 hours
- Low      → cosmetic, minor inconvenience                  → fix in 24 hours

DEPARTMENT must be one of:
- Electrical    → lights, AC, fans, power, generator
- Network  → wifi, internet, computers, software, printer
- Audio-Visual  → projector, microphone, speaker, screen
- Civil         → doors, windows, furniture, water leak, plumbing
- Housekeeping  → cleaning, garbage, pest
- Security      → CCTV, gate, lock
- Transport     → bus, vehicle, driver
- Admin Office  → hostel, general

RULES:
- Return ONLY the JSON. No explanation, no extra text, no markdown.
- Never wrap the JSON in code fences like ```json
- If unsure about category, use "Other"
- If unsure about priority, use "Medium"
If an image is provided, use it to better determine the priority and category.
If no image is provided, classify based on text only.
"""
    response = llm.invoke(TEXT_ANALYSIS_SYSTEM_PROMPT)

    return response.content

#print(text_complaint_analyzer.run("All 30 computers in Lab 3 have lost network connectivity. Students are unable to access examination portal. Exam started 15 minutes ago. Urgent resolution required."))

tools = [text_complaint_analyzer]

agent = create_agent(llm ,tools=tools)

