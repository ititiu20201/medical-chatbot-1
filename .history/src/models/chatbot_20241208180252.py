import torch
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime
import random

from .enhanced_phobert import EnhancedMedicalPhoBERT
from ..data.collector import DataCollector
from ..data.treatment_processor import TreatmentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalChatbot:
    def __init__(
        self,
        model_path: str = 'data/models/best_model.pt',
        config_path: str = 'configs/config.json',
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize Medical Chatbot"""
        self.load_models(model_path, config_path, device)
        self.initialize_components()
        self.reset_conversation()

    def load_models(self, model_path: str, config_path: str, device: str):
        """Load necessary models and configurations"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        model_dir = Path(model_path).parent
        with open(model_dir / 'specialty_map.json', 'r') as f:
            self.specialty_map = json.load(f)
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        self.model = EnhancedMedicalPhoBERT.from_pretrained(
            model_path,
            num_specialties=len(self.specialty_map),
            num_symptoms=self.config['model']['num_symptoms'],
            num_treatments=self.config['model']['num_treatments']
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

    def initialize_components(self):
        """Initialize supporting components"""
        self.data_collector = DataCollector()
        self.treatment_processor = TreatmentProcessor()

    def reset_conversation(self):
        """Reset conversation state"""
        self.conversation_state = "greeting"
        self.collected_info = {
            "personal_info": {},
            "symptoms": [],
            "medical_history": {},
            "predictions": {},
            "recommendations": {}
        }
        self.current_patient = None

    def start_conversation(self, patient_id: Optional[str] = None) -> str:
        """Start new conversation"""
        self.reset_conversation()
        
        if patient_id is None:
            patient_id = f"P{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.current_patient = patient_id
        
        # Check for returning patient
        patient_history = self.data_collector.get_patient_history(patient_id)
        if patient_history:
            self.collected_info["personal_info"] = patient_history.get("personal_info", {})
            return (
                f"Xin chào! Rất vui được gặp lại bạn {self.collected_info['personal_info'].get('name', '')}. "
                "Tôi có thể giúp gì cho bạn hôm nay?"
            )
        
        return (
            "Xin chào! Tôi là trợ lý y tế ảo. "
            "Tôi sẽ giúp bạn tìm hiểu về tình trạng sức khỏe và đề xuất chuyên khoa phù hợp. "
            "Xin cho biết họ tên đầy đủ của bạn?"
        )

    def _collect_basic_info(self, user_input: str) -> str:
        """Collect basic patient information"""
        info = self.collected_info["personal_info"]
        
        if "name" not in info:
            info["name"] = user_input
            return "Xin cho biết tuổi của bạn?"
            
        elif "age" not in info:
            try:
                info["age"] = int(user_input)
                return "Xin cho biết giới tính của bạn (Nam/Nữ/Khác)?"
            except:
                return "Xin lỗi, vui lòng nhập tuổi bằng số."
                
        elif "gender" not in info:
            if user_input.lower() in ["nam", "nữ", "khác"]:
                info["gender"] = user_input.lower()
                return "Vui lòng cho biết số điện thoại hoặc email để liên hệ?"
            else:
                return "Vui lòng chọn giới tính: Nam, Nữ hoặc Khác."
                
        elif "contact" not in info:
            info["contact"] = user_input
            self.conversation_state = "symptoms"
            return (
                "Cảm ơn thông tin của bạn. "
                "Bây giờ, xin hãy mô tả các triệu chứng bạn đang gặp phải?"
            )

    def _process_symptoms(self, user_input: str) -> str:
        """Process symptoms and make predictions"""
        self.collected_info["symptoms"].append(user_input)
        
        # Get model predictions
        inputs = self.tokenizer(
            user_input,
            return_tensors="pt",
            max_length=self.config['model']['max_length'],
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        
        # Process specialty predictions
        specialty_probs = torch.softmax(outputs["specialty_logits"], dim=-1)
        top_specs = torch.topk(specialty_probs[0], k=min(3, len(self.specialty_map)))
        
        self.collected_info["predictions"]["specialties"] = [
            {
                "specialty": self.specialty_map[str(idx.item())],
                "confidence": prob.item()
            }
            for idx, prob in zip(top_specs.indices, top_specs.values)
        ]
        
        self.conversation_state = "medical_history"
        return (
            "Bạn có đang mắc bệnh mãn tính nào không? "
            "Hoặc có tiền sử bệnh, phẫu thuật, dị ứng cần lưu ý không?"
        )

    def _collect_medical_history(self, user_input: str) -> str:
        """Collect and process medical history"""
        self.collected_info["medical_history"]["description"] = user_input
        
        # Get treatment recommendations
        treatments = self.treatment_processor.get_treatment_recommendation(
            symptoms=self.collected_info["symptoms"],
            medical_history=self.collected_info["medical_history"]
        )
        
        self.collected_info["recommendations"] = treatments
        self.conversation_state = "confirm"
        
        primary_specialty = self.collected_info["predictions"]["specialties"][0]["specialty"]
        return (
            f"Dựa trên thông tin bạn cung cấp, tôi đề xuất bạn nên khám tại khoa {primary_specialty}. "
            "Bạn có muốn xem bản tóm tắt thông tin và đặt lịch khám không? (Có/Không)"
        )

    def _generate_medical_record(self) -> Dict:
        """Generate medical record from collected information"""
        return {
            "patient_id": self.current_patient,
            "record_time": datetime.now().isoformat(),
            "personal_info": self.collected_info["personal_info"],
            "medical_info": {
                "symptoms": self.collected_info["symptoms"],
                "medical_history": self.collected_info["medical_history"]
            },
            "recommendations": {
                "specialties": self.collected_info["predictions"]["specialties"],
                "treatments": self.collected_info["recommendations"]
            }
        }

    def _handle_booking(self, user_input: str) -> str:
        """Handle appointment booking and record generation"""
        if user_input.lower() in ["có", "ok", "đồng ý"]:
            # Generate medical record
            medical_record = self._generate_medical_record()
            
            # Save patient data
            self.data_collector.create_patient_profile({
                "patient_id": self.current_patient,
                "medical_record": medical_record,
                "timestamp": datetime.now().isoformat()
            })
            
            # Get queue number (should be replaced with actual queue system)
            queue_number = self._get_queue_number(
                self.collected_info["predictions"]["specialties"][0]["specialty"]
            )
            
            response = (
                f"Đã tạo hồ sơ khám bệnh và đặt lịch khám thành công!\n"
                f"- Khoa: {medical_record['recommendations']['specialties'][0]['specialty']}\n"
                f"- Số thứ tự: {queue_number}\n\n"
                "Bạn có thể xem chi tiết hồ sơ bệnh án và thông tin đặt khám trong phần thông tin chi tiết."
            )
            
            self.conversation_state = "completed"
            return response
        else:
            self.conversation_state = "completed"
            return (
                "Cảm ơn bạn đã tham khảo. "
                "Nếu cần hỗ trợ thêm, hãy quay lại khi cần nhé!"
            )

    def _get_queue_number(self, specialty: str) -> int:
        """Get queue number for a specialty"""
        # This should be replaced with actual queue management system
        return random.randint(1, 100)

    def get_response(self, user_input: str) -> Dict:
        """Get chatbot response based on conversation state"""
        try:
            if self.conversation_state == "greeting":
                response = self._collect_basic_info(user_input)
            elif self.conversation_state == "symptoms":
                response = self._process_symptoms(user_input)
            elif self.conversation_state == "medical_history":
                response = self._collect_medical_history(user_input)
            elif self.conversation_state == "confirm":
                response = self._handle_booking(user_input)
            else:
                response = "Xin lỗi, tôi không hiểu. Bạn có thể nói rõ hơn được không?"

            return {
                "response": response,
                "state": self.conversation_state,
                "collected_info": self.collected_info if self.conversation_state == "completed" else None
            }

        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            return {
                "response": "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại sau.",
                "state": "error",
                "error": str(e)
            }

    def get_queue_status(self, specialty: str) -> Dict:
        """Get current queue status for a specialty"""
        # This should be replaced with actual queue management system
        return {
            "specialty": specialty,
            "current_number": random.randint(1, 100),
            "waiting_time": random.randint(10, 60)
        }