import torch
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
import random
from datetime import datetime

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
        """
        Initialize Medical Chatbot
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
            device: Device to use
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Load specialty mapping
        model_dir = Path(model_path).parent
        with open(model_dir / 'specialty_map.json', 'r') as f:
            self.specialty_map = json.load(f)
            
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        self.model = EnhancedMedicalPhoBERT.from_pretrained(
            model_path,
            num_specialties=len(self.specialty_map),
            num_symptoms=self.config['model']['num_symptoms'],
            num_treatments=self.config['model']['num_treatments']
        )
        self.model.to(device)
        self.model.eval()
        
        # Initialize components
        self.device = device
        self.data_collector = DataCollector()
        self.treatment_processor = TreatmentProcessor()
        
        # Initialize conversation state
        self.current_patient = None
        self.conversation_state = None
        self.collected_info = {}
        
        logger.info("Medical Chatbot initialized successfully")

    def start_conversation(self, patient_id: Optional[str] = None) -> str:
        """
        Start a new conversation
        
        Args:
            patient_id: Optional patient ID for returning patients
            
        Returns:
            str: Initial greeting message
        """
        # Generate patient ID if not provided
        if patient_id is None:
            patient_id = f"P{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.current_patient = patient_id
        self.conversation_state = "greeting"
        self.collected_info = {}
        
        # Check if returning patient
        patient_history = self.data_collector.get_patient_history(patient_id)
        if patient_history:
            return f"""
            Xin chào! Rất vui được gặp lại bạn. 
            Tôi có thể giúp gì cho bạn hôm nay?
            """
        else:
            return f"""
            Xin chào! Tôi là trợ lý y tế ảo. 
            Tôi sẽ giúp bạn tìm hiểu về tình trạng sức khỏe 
            và đề xuất chuyên khoa phù hợp. 
            Trước tiên, xin cho biết họ tên đầy đủ của bạn?
            """

    def _collect_basic_info(self, user_input: str) -> str:
        """Collect basic patient information"""
        if "name" not in self.collected_info:
            self.collected_info["name"] = user_input
            return "Xin cho biết tuổi của bạn?"
        elif "age" not in self.collected_info:
            try:
                age = int(user_input)
                self.collected_info["age"] = age
                return "Xin cho biết giới tính của bạn (Nam/Nữ/Khác)?"
            except:
                return "Xin lỗi, vui lòng nhập tuổi bằng số."
        elif "gender" not in self.collected_info:
            if user_input.lower() in ["nam", "nữ", "khác"]:
                self.collected_info["gender"] = user_input
                return "Vui lòng cho biết số điện thoại hoặc email để liên hệ?"
            else:
                return "Vui lòng chọn giới tính: Nam, Nữ hoặc Khác."
        elif "contact" not in self.collected_info:
            self.collected_info["contact"] = user_input
            self.conversation_state = "symptoms"
            return """
            Cảm ơn thông tin của bạn.
            Bây giờ, xin hãy mô tả các triệu chứng bạn đang gặp phải?
            """
            
    def _collect_symptoms(self, user_input: str) -> str:
        """Collect and process symptoms"""
        if "symptoms" not in self.collected_info:
            self.collected_info["symptoms"] = user_input
            
            # Process symptoms and get predictions
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
            
            # Get specialty predictions
            specialty_probs = torch.softmax(outputs["specialty_logits"], dim=-1)
            top_specs = torch.topk(
                specialty_probs[0],
                k=min(3, len(self.specialty_map))
            )
            
            # Save predictions
            self.collected_info["predicted_specialties"] = [
                {
                    "specialty": self.specialty_map[str(idx.item())],
                    "confidence": prob.item()
                }
                for idx, prob in zip(top_specs.indices, top_specs.values)
            ]
            
            self.conversation_state = "medical_history"
            return "Bạn có tiền sử bệnh lý nào không? (ví dụ: bệnh mãn tính, phẫu thuật...)"
            
    def _collect_medical_history(self, user_input: str) -> str:
        """Collect medical history"""
        if "medical_history" not in self.collected_info:
            self.collected_info["medical_history"] = user_input
            self.conversation_state = "confirm"
            
            # Get treatment recommendations
            treatments = self.treatment_processor.get_treatment_recommendation(
                symptoms=self.collected_info["symptoms"].split(),
                medical_history={"description": user_input}
            )
            
            self.collected_info["recommendations"] = treatments
            
            # Prepare confirmation message
            specs = self.collected_info["predicted_specialties"]
            primary_spec = specs[0]["specialty"]
            
            return f"""
            Dựa trên thông tin bạn cung cấp, tôi đề xuất bạn nên khám tại khoa {primary_spec}.
            Bạn có muốn đặt lịch khám không? (Có/Không)
            """

    def _handle_booking(self, user_input: str) -> str:
        """Handle appointment booking"""
        if user_input.lower() in ["có", "ok", "đồng ý"]:
            # Generate queue number
            queue_number = random.randint(1, 100)  # This should be replaced with actual queue system
            
            # Save patient data
            self.data_collector.create_patient_profile({
                "patient_id": self.current_patient,
                "info": self.collected_info,
                "queue_number": queue_number,
                "timestamp": datetime.now().isoformat()
            })
            
            primary_spec = self.collected_info["predicted_specialties"][0]["specialty"]
            
            return f"""
            Đã đặt lịch khám thành công!
            - Khoa: {primary_spec}
            - Số thứ tự: {queue_number}
            
            Xin vui lòng đến đúng khoa để được khám.
            Cảm ơn bạn đã sử dụng dịch vụ!
            """
        else:
            return """
            Cảm ơn bạn đã tham khảo.
            Nếu cần hỗ trợ thêm, hãy quay lại khi cần nhé!
            """

    def get_response(self, user_input: str) -> str:
        """
        Get chatbot response based on user input
        
        Args:
            user_input: User's message
            
        Returns:
            str: Chatbot's response
        """
        try:
            if self.conversation_state == "greeting":
                return self._collect_basic_info(user_input)
                
            elif self.conversation_state == "symptoms":
                return self._collect_symptoms(user_input)
                
            elif self.conversation_state == "medical_history":
                return self._collect_medical_history(user_input)
                
            elif self.conversation_state == "confirm":
                return self._handle_booking(user_input)
                
            else:
                return "Xin lỗi, tôi không hiểu. Bạn có thể nói rõ hơn được không?"
                
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            return "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại sau."

    def get_queue_status(self, specialty: str) -> Dict:
        """
        Get current queue status for a specialty
        
        Args:
            specialty: Medical specialty
            
        Returns:
            Dict: Queue status information
        """
        # This should be replaced with actual queue management system
        return {
            "specialty": specialty,
            "current_number": random.randint(1, 100),
            "waiting_time": random.randint(10, 60)
        }

    def save_conversation(self):
        """Save conversation data"""
        if self.current_patient and self.collected_info:
            self.data_collector.collect_patient_response(
                self.current_patient,
                self.collected_info,
                "conversation"
            )