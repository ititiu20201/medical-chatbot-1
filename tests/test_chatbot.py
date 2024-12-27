import pytest
from src.models.chatbot import MedicalChatbot
import json
import torch

class TestMedicalChatbot:
    @pytest.fixture
    def chatbot(self):
        return MedicalChatbot()

    def test_conversation_flow(self, chatbot):
        """Test complete conversation flow"""
        # Start conversation
        response = chatbot.start_conversation()
        assert chatbot.conversation_state == "greeting"
        assert "Xin chào" in response

        # Test name collection
        result = chatbot.get_response("Nguyễn Văn A")
        assert "tuổi" in result["response"].lower()

        # Test age collection
        result = chatbot.get_response("30")
        assert "giới tính" in result["response"].lower()

        # Test gender collection
        result = chatbot.get_response("Nam")
        assert "liên hệ" in result["response"].lower()

        # Test contact collection
        result = chatbot.get_response("0123456789")
        assert "triệu chứng" in result["response"].lower()
        assert chatbot.conversation_state == "symptoms"

        # Test symptom collection
        result = chatbot.get_response("Đau đầu, sốt nhẹ")
        assert chatbot.conversation_state == "medical_history"

        # Test medical history collection
        result = chatbot.get_response("Không có bệnh nền")
        assert chatbot.conversation_state == "confirm"

        # Test booking confirmation
        result = chatbot.get_response("Có")
        assert chatbot.conversation_state == "completed"
        assert result.get("collected_info") is not None

    def test_specialty_prediction(self, chatbot):
        """Test specialty prediction accuracy"""
        # Input symptoms
        result = chatbot._process_symptoms("Đau đầu, sốt cao, ho")
        predictions = chatbot.collected_info["predictions"]["specialties"]
        
        assert len(predictions) > 0
        assert "specialty" in predictions[0]
        assert "confidence" in predictions[0]
        assert predictions[0]["confidence"] > 0

    def test_medical_record_generation(self, chatbot):
        """Test medical record generation"""
        # Setup test data
        chatbot.collected_info = {
            "personal_info": {
                "name": "Test User",
                "age": 30,
                "gender": "Nam",
                "contact": "test@test.com"
            },
            "symptoms": ["Đau đầu", "Sốt"],
            "medical_history": {"description": "Không có"},
            "predictions": {
                "specialties": [{"specialty": "Nội khoa", "confidence": 0.8}]
            },
            "recommendations": {
                "treatments": ["Paracetamol"]
            }
        }

        record = chatbot._generate_medical_record()
        assert record["personal_info"]["name"] == "Test User"
        assert record["medical_info"]["symptoms"] == ["Đau đầu", "Sốt"]
        assert "recommendations" in record

    def test_error_handling(self, chatbot):
        """Test error handling"""
        # Test invalid age
        result = chatbot.get_response("invalid_age")
        assert "số" in result["response"].lower()

        # Test invalid gender
        result = chatbot.get_response("invalid_gender")
        assert "giới tính" in result["response"].lower()

    def test_queue_management(self, chatbot):
        """Test queue number generation and status"""
        status = chatbot.get_queue_status("Nội khoa")
        assert "specialty" in status
        assert "current_number" in status
        assert "waiting_time" in status

    @pytest.mark.asyncio
    async def test_multiple_conversations(self, chatbot):
        """Test handling multiple conversations"""
        # Start first conversation
        response1 = chatbot.start_conversation("patient1")
        assert chatbot.current_patient == "patient1"

        # Start second conversation
        response2 = chatbot.start_conversation("patient2")
        assert chatbot.current_patient == "patient2"

        # Verify conversations are separate
        assert chatbot.get_response("Test User 2")["patient_id"] == "patient2"