from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import json
import logging
from datetime import datetime

from ..models.chatbot import MedicalChatbot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = MedicalChatbot()

class ChatInput(BaseModel):
    """Schema for chat input"""
    message: str
    patient_id: Optional[str] = None

class QueueRequest(BaseModel):
    """Schema for queue status request"""
    specialty: str

@app.post("/chat")
async def chat(chat_input: ChatInput):
    """
    Chat endpoint
    
    Args:
        chat_input: ChatInput object containing message and optional patient_id
        
    Returns:
        Dict containing response, patient_id, state, and optional medical_record
    """
    try:
        # Start new conversation if needed
        if not hasattr(chatbot, 'current_patient') or chat_input.patient_id != chatbot.current_patient:
            response = chatbot.start_conversation(chat_input.patient_id)
            return {
                "response": response,
                "patient_id": chatbot.current_patient,
                "state": chatbot.conversation_state
            }
        
        # Get response for user message
        result = chatbot.get_response(chat_input.message)
        
        # Include medical record if conversation is completed
        if result["state"] == "completed" and result.get("collected_info"):
            return {
                "response": result["response"],
                "patient_id": chatbot.current_patient,
                "state": result["state"],
                "medical_record": result["collected_info"]
            }
        
        return {
            "response": result["response"],
            "patient_id": chatbot.current_patient,
            "state": result["state"]
        }
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/queue-status")
async def get_queue_status(request: QueueRequest):
    """
    Get queue status for a specialty
    
    Args:
        request: QueueRequest object containing specialty
        
    Returns:
        Dict containing queue status information
    """
    try:
        status = chatbot.get_queue_status(request.specialty)
        return status
    except Exception as e:
        logger.error(f"Error in queue status endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            data = json.loads(data)
            
            # Process message
            if not hasattr(chatbot, 'current_patient'):
                response = chatbot.start_conversation(data.get('patient_id'))
                await websocket.send_json({
                    "response": response,
                    "patient_id": chatbot.current_patient,
                    "state": chatbot.conversation_state
                })
            else:
                result = chatbot.get_response(data['message'])
                
                # Include medical record if conversation is completed
                if result["state"] == "completed" and result.get("collected_info"):
                    await websocket.send_json({
                        "response": result["response"],
                        "patient_id": chatbot.current_patient,
                        "state": result["state"],
                        "medical_record": result["collected_info"]
                    })
                else:
                    await websocket.send_json({
                        "response": result["response"],
                        "patient_id": chatbot.current_patient,
                        "state": result["state"]
                    })
                    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)