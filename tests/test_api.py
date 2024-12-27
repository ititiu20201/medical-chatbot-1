import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import json
import websockets
import asyncio

client = TestClient(app)

def test_chat_endpoint():
    """Test REST chat endpoint"""
    # Test initial conversation
    response = client.post(
        "/chat",
        json={"message": "Xin chào", "patient_id": None}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "patient_id" in data
    assert "state" in data

    # Test conversation flow
    patient_id = data["patient_id"]
    
    # Test name input
    response = client.post(
        "/chat",
        json={"message": "Nguyễn Văn A", "patient_id": patient_id}
    )
    assert response.status_code == 200
    assert "tuổi" in response.json()["response"].lower()

    # Test complete flow
    test_inputs = [
        "30",               # age
        "Nam",              # gender
        "test@test.com",    # contact
        "Đau đầu, sốt",    # symptoms
        "Không có",         # medical history
        "Có"               # booking confirmation
    ]

    for message in test_inputs:
        response = client.post(
            "/chat",
            json={"message": message, "patient_id": patient_id}
        )
        assert response.status_code == 200

def test_queue_status_endpoint():
    """Test queue status endpoint"""
    response = client.post(
        "/queue-status",
        json={"specialty": "Nội khoa"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "specialty" in data
    assert "current_number" in data
    assert "waiting_time" in data

@pytest.mark.asyncio
async def test_websocket():
    """Test WebSocket connection"""
    async with websockets.connect("ws://localhost:8000/ws") as websocket:
        # Test initial message
        await websocket.send(json.dumps({
            "message": "Xin chào",
            "patient_id": None
        }))
        response = await websocket.recv()
        data = json.loads(response)
        assert "response" in data
        assert "patient_id" in data

def test_error_handling():
    """Test API error handling"""
    # Test invalid input
    response = client.post(
        "/chat",
        json={"invalid": "data"}
    )
    assert response.status_code != 200

    # Test missing required fields
    response = client.post("/chat", json={})
    assert response.status_code != 200

def test_concurrent_requests():
    """Test handling concurrent requests"""
    async def make_requests():
        tasks = []
        for i in range(5):
            task = client.post(
                "/chat",
                json={"message": f"Test {i}", "patient_id": f"patient{i}"}
            )
            tasks.append(task)
        responses = await asyncio.gather(*tasks)
        return responses

    responses = asyncio.run(make_requests())
    assert all(r.status_code == 200 for r in responses)

def test_large_input():
    """Test handling large input"""
    large_text = "A" * 1000  # 1000 character string
    response = client.post(
        "/chat",
        json={"message": large_text, "patient_id": None}
    )
    assert response.status_code == 200

def test_vietnamese_character_handling():
    """Test handling Vietnamese characters"""
    vietnamese_text = "Xin chào, tôi bị đau đầu và sốt"
    response = client.post(
        "/chat",
        json={"message": vietnamese_text, "patient_id": None}
    )
    assert response.status_code == 200
    assert "response" in response.json()