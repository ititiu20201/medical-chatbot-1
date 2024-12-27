# src/api/routes/symptom_analysis.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class SymptomAnalysisRequest(BaseModel):
    symptoms: str
    patientInfo: Optional[Dict] = None

class SymptomAnalysisResponse(BaseModel):
    recommendedSpecialty: str
    needsMoreInfo: bool = False
    followUpQuestion: Optional[str] = None
    confidence: float
    recommendations: Optional[List[str]] = None

@router.post("/analyze-symptoms", response_model=SymptomAnalysisResponse)
async def analyze_symptoms(request: SymptomAnalysisRequest):
    try:
        logger.info(f"Analyzing symptoms: {request.symptoms}")
        
        # Clean and normalize the symptoms text
        symptoms = request.symptoms.lower().strip()
        
        # Example symptom-to-specialty mapping
        specialty_mapping = {
            'đau đầu': 'Thần kinh',
            'đau dạ dày': 'Tiêu hóa',
            'đau ngực': 'Tim mạch',
            'ho': 'Hô hấp',
            'sốt': 'Nội tổng quát',
        }

        # Find matching specialty
        matched_specialty = None
        for symptom, specialty in specialty_mapping.items():
            if symptom in symptoms:
                matched_specialty = specialty
                break

        if matched_specialty is None:
            return SymptomAnalysisResponse(
                recommendedSpecialty="Đang phân tích",
                needsMoreInfo=True,
                followUpQuestion="Vui lòng mô tả chi tiết hơn về triệu chứng của bạn. Ví dụ: vị trí đau, mức độ đau, thời gian kéo dài?",
                confidence=0.0
            )

        return SymptomAnalysisResponse(
            recommendedSpecialty=matched_specialty,
            needsMoreInfo=False,
            confidence=0.8,
            recommendations=[
                f"Khám {matched_specialty}",
                "Chuẩn bị các xét nghiệm cơ bản",
                "Mang theo các kết quả xét nghiệm trước đây (nếu có)"
            ]
        )

    except Exception as e:
        logger.error(f"Error analyzing symptoms: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi khi phân tích triệu chứng")