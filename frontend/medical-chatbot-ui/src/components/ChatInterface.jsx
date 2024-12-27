import React, { useState, useRef, useEffect } from 'react';
import { Send, Clock, User, Bot } from 'lucide-react';

const MedicalChat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentState, setCurrentState] = useState('initial');
  const [patientInfo, setPatientInfo] = useState({});
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const messagesEndRef = useRef(null);

  const conversationStates = {
    initial: 'initial',
    personalInfo: 'personal_info',
    symptoms: 'symptoms',
    medicalHistory: 'medical_history',
    recommendations: 'recommendations',
    completed: 'completed'
  };

  const personalInfoQuestions = [
    "Xin chào! Tôi là trợ lý y tế ảo. Xin cho biết họ tên đầy đủ của bạn?",
    "Bạn bao nhiêu tuổi?",
    "Xin cho biết giới tính của bạn (Nam/Nữ/Khác)?",
    "Vui lòng cung cấp số điện thoại hoặc email để liên hệ?"
  ];

  const symptomQuestions = [
    "Bạn đang gặp phải những triệu chứng gì?",
    "Triệu chứng này đã kéo dài bao lâu?",
    "Mức độ khó chịu của triệu chứng từ 1-10?",
    "Triệu chứng có thường xuyên xuất hiện không?"
  ];

  useEffect(() => {
    startConversation();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const startConversation = () => {
    setMessages([{
      type: 'bot',
      content: personalInfoQuestions[0],
      timestamp: new Date()
    }]);
    setCurrentState(conversationStates.personalInfo);
  };

  const addBotMessage = (content, additionalInfo = {}) => {
    setMessages(prev => [...prev, {
      type: 'bot',
      content,
      timestamp: new Date(),
      ...additionalInfo
    }]);
  };

  const addUserMessage = (content) => {
    setMessages(prev => [...prev, {
      type: 'user',
      content,
      timestamp: new Date()
    }]);
  };

  const validateAge = (age) => {
    const parsedAge = parseInt(age);
    return {
      isValid: !isNaN(parsedAge) && parsedAge > 0 && parsedAge <= 150,
      message: "Vui lòng nhập tuổi hợp lệ bằng số (0-150)."
    };
  };

  const validatePhoneNumber = (phone) => {
    const phoneRegex = /^(0[0-9]{9})$/;
    return {
      isValid: phoneRegex.test(phone),
      message: "Vui lòng nhập số điện thoại hợp lệ (10 số, bắt đầu bằng số 0)."
    };
  };

  const processPersonalInfo = (input) => {
    const updatedInfo = { ...patientInfo };
    
    switch (currentQuestionIndex) {
      case 0:  // Name
        if (input.trim().length < 2) {
          addBotMessage("Vui lòng nhập họ tên đầy đủ của bạn.");
          return false;
        }
        updatedInfo.name = input;
        break;

      case 1:  // Age
        const ageValidation = validateAge(input);
        if (!ageValidation.isValid) {
          addBotMessage(ageValidation.message);
          return false;
        }
        updatedInfo.age = parseInt(input);
        break;

      case 2:  // Gender
        const gender = input.toLowerCase();
        if (!['nam', 'nữ', 'khác'].includes(gender)) {
          addBotMessage("Vui lòng chọn giới tính: Nam, Nữ hoặc Khác.");
          return false;
        }
        updatedInfo.gender = gender;
        break;

      case 3:  // Contact
        if (input.includes('@')) {
          if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(input)) {
            addBotMessage("Vui lòng nhập địa chỉ email hợp lệ.");
            return false;
          }
        } else {
          const phoneValidation = validatePhoneNumber(input);
          if (!phoneValidation.isValid) {
            addBotMessage(phoneValidation.message);
            return false;
          }
        }
        updatedInfo.contact = input;
        break;

      default:
        return false;
    }

    setPatientInfo(updatedInfo);
    return true;
  };

  const processSymptoms = async (input) => {
    if (!input || input.trim().length < 2) {
      addBotMessage("Vui lòng mô tả chi tiết triệu chứng của bạn.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/analyze-symptoms', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          symptoms: input,
          patientInfo
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.needsMoreInfo) {
        addBotMessage(data.followUpQuestion);
        return;
      }

      let recommendationText = `Dựa trên triệu chứng của bạn, tôi đề xuất:
1. Khám tại khoa ${data.recommendedSpecialty}
${data.recommendations ? '\n' + data.recommendations.map((rec, i) => `${i + 2}. ${rec}`).join('\n') : ''}

Bạn có tiền sử bệnh lý nào cần lưu ý không?`;

      addBotMessage(recommendationText);
      setCurrentState(conversationStates.medicalHistory);
      
    } catch (error) {
      console.error('Error processing symptoms:', error);
      addBotMessage("Xin lỗi, có lỗi xảy ra khi phân tích triệu chứng. Vui lòng thử lại sau.");
    } finally {
      setLoading(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    setLoading(true);
    addUserMessage(input);
    const userInput = input.trim();
    setInput('');

    try {
      switch (currentState) {
        case conversationStates.personalInfo:
          if (processPersonalInfo(userInput)) {
            if (currentQuestionIndex < personalInfoQuestions.length - 1) {
              setCurrentQuestionIndex(prev => prev + 1);
              addBotMessage(personalInfoQuestions[currentQuestionIndex + 1]);
            } else {
              setCurrentState(conversationStates.symptoms);
              addBotMessage(symptomQuestions[0]);
              setCurrentQuestionIndex(0);
            }
          }
          break;

        case conversationStates.symptoms:
          await processSymptoms(userInput);
          break;

        case conversationStates.medicalHistory:
          const response = await fetch('/api/generate-recommendations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              patientInfo,
              medicalHistory: userInput
            })
          });
          
          const recommendations = await response.json();
          
          addBotMessage("Dựa trên thông tin của bạn, đây là các đề xuất của tôi:", {
            recommendations
          });
          setCurrentState(conversationStates.completed);
          break;

        default:
          addBotMessage("Xin lỗi, tôi không hiểu. Vui lòng thử lại.");
      }
    } catch (error) {
      addBotMessage("Xin lỗi, có lỗi xảy ra. Vui lòng thử lại.");
    }

    setLoading(false);
  };

  const handleRestart = () => {
    setMessages([]);
    setPatientInfo({});
    setCurrentQuestionIndex(0);
    setCurrentState(conversationStates.initial);
    startConversation();
  };

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto p-4">
      <div className="flex-grow overflow-hidden flex flex-col bg-white rounded-lg shadow">
        <div className="flex-grow overflow-y-auto p-4 space-y-4">
          {messages.map((message, idx) => (
            <div
              key={idx}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex items-start space-x-2 max-w-[80%] ${
                message.type === 'user' ? 'flex-row-reverse' : 'flex-row'
              }`}>
                {message.type === 'user' ? (
                  <User className="w-6 h-6 text-blue-500" />
                ) : (
                  <Bot className="w-6 h-6 text-green-500" />
                )}
                <div className={`rounded-lg p-3 ${
                  message.type === 'user' 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-100 text-gray-900'
                }`}>
                  <p className="text-sm whitespace-pre-line">{message.content}</p>
                  {message.recommendations && (
                    <div className="mt-2 text-xs">
                      <p className="font-semibold">Đề xuất khám bệnh:</p>
                      <p>Chuyên khoa: {message.recommendations.specialty}</p>
                      <p>Mức độ ưu tiên: {message.recommendations.urgency}</p>
                      <p>Thời gian chờ dự kiến: {message.recommendations.estimatedWaitTime}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <div className="p-4 border-t">
          <div className="flex space-x-2">
            {currentState === conversationStates.completed ? (
              <button
                onClick={handleRestart}
                className="w-full p-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
              >
                Bắt đầu cuộc tư vấn mới
              </button>
            ) : (
              <>
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                  placeholder="Nhập câu trả lời của bạn..."
                  className="flex-grow p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={loading}
                />
                <button
                  onClick={handleSend}
                  disabled={loading || !input.trim()}
                  className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? (
                    <Clock className="w-5 h-5 animate-spin" />
                  ) : (
                    <Send className="w-5 h-5" />
                  )}
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MedicalChat;