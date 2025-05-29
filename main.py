from sqlmodel import Session, SQLModel, create_engine, select
from models import ChatSession, Message
from fastapi import Depends, FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from typing import List
import os
from uuid import UUID, uuid4
import time
from google import genai
import utils
import subprocess
import datetime
import subprocess
from google import genai
from google.genai import types
api_key = "gemini api key here"
client = genai.Client(api_key=api_key)
ADB = "scrcpy\\adb"

def adb_ss(save_dir="screenshots"):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ss_{timestamp}.png"
    device_path = f"/sdcard/{filename}"
    local_path = os.path.join(save_dir, filename)

    try:
        # Capture screenshot on device
        subprocess.run([ADB, "shell", "screencap", "-p", device_path], check=True)
        # Pull screenshot to local machine
        subprocess.run([ADB, "pull", device_path, local_path], check=True)
        # Optionally remove from device
        subprocess.run([ADB, "shell", "rm", device_path], check=True)

        print(f"Screenshot saved to: {local_path}")
        return local_path

    except subprocess.CalledProcessError as e:
        print("Error while taking screenshot:", e)
        return None
    
def touch_input(x:int, y:int):
    print(x," ", y)
    subprocess.run(f"{ADB} shell input tap {x} {y}", shell=True)


def tap_index(index: int, box_map: dict):
    print(box_map[index][0],box_map[index][1])
    touch_input(box_map[index][0],box_map[index][1])

def input_text(text:str, box_map:dict):
    subprocess.run(f"{ADB} shell input text '{text}'", shell=True)

def success():
    launch_app("com.example.chat_app")
    print("Success")

def loading():
    time.sleep(2)
    print("Loading")



def get_screen_size() -> tuple:
    """
    Get the device's screen width and height using adb.

    Returns:
        (width, height): tuple of integers representing screen resolution.
    """
    result = subprocess.run(f"{ADB} shell wm size", shell=True, capture_output=True, text=True)
    output = result.stdout.strip()
    if "Physical size:" in output:
        size_str = output.split("Physical size:")[1].strip()
        width, height = map(int, size_str.split("x"))
        return width, height
    else:
        raise RuntimeError("Unable to get screen size. Check ADB connection.")

def swipe_from_center(direction: str, distance_ratio: float = 0.3, duration_ms: int = 300):
    """
    Perform a swipe from the screen center in a given direction.

    Args:
        direction (str): 'up', 'down', 'left', 'right'.
        distance_ratio (float): How far to swipe as % of screen size (default 0.3 = 30%).
        duration_ms (int): Duration of swipe in milliseconds.
    """
    width, height = get_screen_size()
    cx, cy = width // 2, height // 2

    dx = int(width * distance_ratio)
    dy = int(height * distance_ratio)

    if direction == "up":
        end_x, end_y = cx, cy - dy
    elif direction == "down":
        end_x, end_y = cx, cy + dy
    elif direction == "left":
        end_x, end_y = cx - dx, cy
    elif direction == "right":
        end_x, end_y = cx + dx, cy
    else:
        raise ValueError("Invalid direction. Use 'up', 'down', 'left', or 'right'.")

    print(f"Swiping {direction} from ({cx}, {cy}) to ({end_x}, {end_y}) in {duration_ms}ms")
    subprocess.run(f"{ADB} shell input swipe {cx} {cy} {end_x} {end_y} {duration_ms}", shell=True)
def adb_shell(cmd):
    result = subprocess.run(f'{ADB} shell {cmd}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, encoding='utf-8')
    return result.stdout.strip()
def get_installed_packages():
    output = adb_shell('pm list packages -3')  # Only user-installed apps
    packages = [line.replace('package:', '').strip() for line in output.splitlines()]
    return packages

import cv2
import numpy as np
import datetime


def img_grid(image: str, nms_thresh=0.3, score_thresh=0.5, merge_distance=20):
    import cv2
    import numpy as np
    import datetime
    
    img = cv2.imread(image)
    if img is None:
        raise ValueError(f"Image at path '{image}' not found.")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Edge detection
    edges = cv2.Canny(img_blur, 100, 300)
    edges = cv2.GaussianBlur(edges, (11, 11), 5)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [[x, y, x + w, y + h] for x, y, w, h in boxes]

    # Filter nested boxes
    def is_inside(inner, outer):
        return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

    filtered_boxes = [b1 for i, b1 in enumerate(boxes) if not any(i != j and is_inside(b1, b2) for j, b2 in enumerate(boxes))]

    if not filtered_boxes:
        print("No valid bounding boxes found.")
        return {}, None

    # Merge close bounding boxes
    def should_merge(box1, box2, distance=merge_distance):
        # Calculate the center points
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        
        # Calculate Euclidean distance between centers
        dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return dist < distance

    def merge_boxes(box1, box2):
        # Create a new bounding box that encompasses both
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        return [x1, y1, x2, y2]

    # Iteratively merge close boxes until no more merges are possible
    merged = True
    merged_boxes = filtered_boxes.copy()
    
    while merged:
        merged = False
        i = 0
        while i < len(merged_boxes):
            j = i + 1
            while j < len(merged_boxes):
                if should_merge(merged_boxes[i], merged_boxes[j]):
                    # Merge boxes i and j
                    merged_boxes[i] = merge_boxes(merged_boxes[i], merged_boxes[j])
                    # Remove box j
                    merged_boxes.pop(j)
                    merged = True
                else:
                    j += 1
            i += 1

    # Apply NMS on the merged boxes
    boxes_np = np.array(merged_boxes, dtype=float)
    nms_input = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes_np]
    indices = cv2.dnn.NMSBoxes(nms_input, [1.0] * len(nms_input), score_thresh, nms_thresh)

    # Draw final bounding boxes with improved visibility
    bounding_box_dict = {}
    image_copy = img.copy()

    if len(indices) > 0:
        for idx in indices.flatten():
            x1, y1, x2, y2 = map(int, boxes_np[idx])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            bounding_box_dict[int(idx)] = (center_x, center_y)
            
            # Enhanced visualization:
            # 1. Draw rectangle with thicker border
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Thicker green border (3px)
            
            # 2. Create text with better visibility
            text = str(idx)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Draw text background box for contrast
            cv2.rectangle(image_copy, 
                         (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0] + 5, y1), 
                         (0, 255, 0), 
                         -1)  # Filled rectangle for text background
            
            # Draw text with black outline for better visibility on any background
            cv2.putText(image_copy, text, (x1 + 3, y1 - 7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)  # Thicker black outline
            cv2.putText(image_copy, text, (x1 + 3, y1 - 7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # White text

    assert img.shape == image_copy.shape, "Image resolution changed unexpectedly."

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_raw = f"output_raw_{timestamp}.jpg"

    # Save raw image first
    cv2.imwrite(filename_raw, image_copy)
    print(f"Raw image saved as: {filename_raw}")

    print(f"Bounding Boxes: {bounding_box_dict}")
    return bounding_box_dict, filename_raw



def launch_app_complete_goal(package_name:str,goal:str):
    launch_app(package_name)
    print("Goal: ", goal)
    # Function declaration for tool calling
    tap_index_declaration = {
        "name": "tap_index",
        "description": "number of the bounding box to tap",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "integer index of the bounding box",
                },
            },
            "required": ["index"],
        },
    }
    input_text_declaration = {
        "name": "input_text",
        "description": "Input text to active text field",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "text that you want to fill in the field",
                },
            },
            "required": ["text"],
        },
    }
    success_declaration = {
        "name": "success",
        "description": "goal completed",
       
    }
    loading_declaration = {
        "name": "loading",
        "description": "wait for the loading screen to finish loading",
       
    }
    swipe_from_center_declaration = {
    "name": "swipe_from_center",
    "description": "Swipe in a direction from the screen center (up, down, left, right).",
    "parameters": {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
                "description": "Direction to swipe from center of screen.",
            },
            "distance_ratio": {
                "type": "number",
                "default": 0.3,
                "description": "Fraction of screen size to swipe (0.3 means 30% distance).",
            },
            "duration_ms": {
                "type": "integer",
                "default": 300,
                "description": "Duration of the swipe in milliseconds.",
            },
        },
        "required": ["direction"],
    },
}




    # Generation config with tool declarations
    tools = types.Tool(function_declarations=[tap_index_declaration,input_text_declaration,success_declaration,loading_declaration,swipe_from_center_declaration])
    config = types.GenerateContentConfig(system_instruction=f"YOU ARE AN AGENTIC AI THAT HAS FULL ACCESS TO AN ANDROID DEVICE. I WILL SEND YOU PHONE SCREENSHOTS AND YOU HAVE TO ANALYSE IT AND CALL CORRECT FUNCTIONS TO COMPLETE YOUR GOAL, IF THE GOAL NEEDS TO BE DONE INSIDE APP USE THE COMPLETE GOAL FUNCTION AFTER OPENING APP. YOU CAN TAP ON ITEMS, INPUT TEXT IN ACTIVE FIELDS, SWIPE AND IF THERE IS A LOADING SCREEN YOU CAN CALL LOADING FUNCTION TO WAIT, DO NOT STOP UNTIL THE GOAL IS COMPLETE. IF SCREENSHOTS ARE REPEATING IT MEANS LAST ACTION FAILED SO TRY SOMETHING NEW, MAKE SMART ASSUMPTIONS IF DETAILS ARE NOT AVAILABLE, USE CASH ON DELIVERY WHEN AVAILABLE, USE ALREADY SAVED WORK LOCATION. SEARCH ONLY IN SEARCH TEXTFIELD, YOUR GOAL IS : {goal}",
        tools=[tools],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="any")
        )
    )
    # Configure the client

    chat = client.chats.create(model="gemini-2.5-flash-preview-04-17", config=config)
    time.sleep(2)
    # Loop 5 times
    for i in range(15):
        print(f"\n🔁 Iteration {i + 1}:")
        box_map, file = img_grid(adb_ss())

        with open(file, "rb") as f:
            img_bytes = f.read()
        response = chat.send_message([
        types.Part(text="Here's an screenshot, please analyze and run the required function to complete your goal."),
        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),])
        
        # Print each function call
        if response.candidates and response.candidates[0].content.parts[0].function_call:
                function_call = response.candidates[0].content.parts[0].function_call
                args_str = ", ".join(f"{key}={value}" for key, value in function_call.args.items())
                print(f"👉 {function_call.name}({args_str})")
                if function_call.name == "tap_index":
                    tap_index(**function_call.args, box_map=box_map) # Pass arguments as keyword arguments
                    
                    
                if function_call.name == "input_text":
                    input_text(**function_call.args,box_map=box_map) # Pass arguments as keyword arguments
                
                if function_call.name == "success":
                    success()
                    break
                if function_call.name == "loading":
                    loading()
                if function_call.name == "swipe_from_center":
                    swipe_from_center(**function_call.args)
                
        else:
            print("⚠️ No function call returned.")


def launch_app(package_name):
    print(f"Launching app: {package_name}")
    result = adb_shell(f'monkey -p {package_name} -c android.intent.category.LAUNCHER 1')
    if "Events injected: 1" in result:
        print(f"✅ {package_name} launched successfully.")
    else:
        print(f"❌ Failed to launch {package_name}.")
    return result


ai_model = "gemini-2.5-flash-preview-04-17"


sqlite_file_name = "chat.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Maybe run a migration script
    create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()
client = genai.Client(api_key=api_key)


@app.get("/{user_id}/c/{agent_id}")
async def chat_session(background_tasks: BackgroundTasks,
                       session: SessionDep,
                       user_id: str,
                       agent_id: str
                       ):
    # Check if user exists
    # user = session.exec(select(User).where(User.id == user_id)).first()
    user = True

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    # Check if user is premium
    if True:  # user.is_premium:
        # Create a new chat session for premium user
        chat_session = ChatSession(  # Add this if sid doesn't have a default
            uid=user_id,
            agent=agent_id,
        )
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)
        print(f'New session created: {chat_session.model_dump_json()}')

        return {
            "sid": str(chat_session.sid),
            "createdAt": chat_session.createdAt,
        }
    else:
        # Handle non-premium user
        raise HTTPException(
            status_code=403,
            detail="Subscription required for this feature"
        )

launch_app_declaration = {
    "name": "launch_app",
    "description": "Launch an app by its package name",
    "parameters": {
        "type": "object",
        "properties": {
            "package_name": {
                "type": "string",
                "enum": get_installed_packages(),
                "description": "Package name of the app to launch",
            },
        },
        "required": ["package_name"],
    },
}

launch_app_complete_goal_declaration = {
    "name": "launch_app_complete_goal",
    "description": "Launch an app by its package name and complete the goal inside it",
    "parameters": {
        "type": "object",
        "properties": {
            "package_name": {
                "type": "string",
                "enum": get_installed_packages(),
                "description": "Package name of the app to launch",
            },
            "goal": {
                "type": "string",
                "description": "detailed goal to be completed inside the app by ai agent with all the details",
            },
        },
        "required": ["package_name", "goal"],
    },
}

@app.websocket("/c/{chat_id}")
async def chat_endpoint(websocket: WebSocket, chat_id: UUID, session: SessionDep, uid: str | None = None):
    valid_session = session.exec(
        select(ChatSession).where(ChatSession.sid == chat_id)
    ).first()
    if valid_session:
        if valid_session.access == "public" or valid_session.uid == uid:
            await manager.connect(websocket)
            try:
                config = await websocket.receive_text()
                match config:
                    case 'all_history':
                        all_messages = valid_session.messages
                        await manager.broadcast(
                            Message(sid=chat_id, type="system_time",
                                    authorId='system', text=f'{valid_session.createdAt}', createdAt=valid_session.createdAt,).model_dump_json())
                        for i in all_messages:
                            await manager.broadcast(i.model_dump_json())
                    case _:
                        pass
                tools = types.Tool(function_declarations=[launch_app_declaration,launch_app_complete_goal_declaration])
                config = types.GenerateContentConfig(system_instruction=f"YOU ARE A PERSONAL ASSISTANT CHATBOT AND AGENTIC AI WHICH CAN RESPOND IN TEXT TO USER QUESTIONS AND AND OPEN ANY ANDROID APP ON AN CONNECTED ANDROID PHONE. ANALYZE THE USER DEMAND AND RESPOND ACCORDINGLY SELECT THE CORRECT APP PACKAGE NAME TO OPEN OR RESPOND WITH CHAT IF NO FUNCTION CALL IS NECESSARY, YOU CAN ONLY SEND TEXT RESPONSE OR ONLY FUNCTION RESPONSE ONLY TEXT RESPONSE WORKS. IF SOME DETAILS ARE MISSING ASK THE USER FOR MORE DETAILS AND THE DO FUNCTION CALLING. IF GIVING GOAL TO AN AGENT FUNCTION MAKE SURE TO GIVE DETAILED GOAL OF ATLEAST 2 LINES",
                tools=[tools])
                chat = client.chats.create(model=ai_model,config=config,)
                while True:

                    data = await websocket.receive_text()

                    response = chat.send_message(data)
                    if response.candidates and response.candidates[0].content.parts[0].function_call:
                        function_call = response.candidates[0].content.parts[0].function_call
                        args_str = ", ".join(f"{key}={value}" for key, value in function_call.args.items())
                        print(f"👉 {function_call.name}({args_str})")
                        if function_call.name == "launch_app":
                            launch_app(**function_call.args)
                        ai_message = Message(sid=chat_id,
                                         authorId="ai", type="text", text="Launching App")
                        if function_call.name == "launch_app_complete_goal":
                            launch_app_complete_goal(**function_call.args)
                            ai_message = Message(sid=chat_id,
                                         authorId="ai", type="text", text="Launching App and completing goal")
                    else:
                        ai_message = Message(sid=chat_id,
                                         authorId="ai", type="text", text=response.text)  # response.text
                    # check for device time mismatch on session continuation
                    user_message = Message(sid=chat_id, type="text",
                                           authorId='user001', text=data)
                    
                    await manager.broadcast(ai_message.model_dump_json())

                    # Save ai and user message

                    session.add_all([user_message, ai_message])
                    session.commit()

            except WebSocketDisconnect:
                manager.disconnect(websocket)
        else:
            # Unauthorized user (Chat exists)
            await websocket.accept()
            await websocket.close(code=1008)

    else:
        # Chat not found
        await websocket.accept()
        await websocket.close(code=1007)
