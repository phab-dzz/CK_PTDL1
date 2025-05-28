import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.preprocessing import DrugDataPreprocessor
from models.decision_tree import DrugDecisionTree  
from models.model_evaluation import ModelEvaluator
from api.prediction_service import DrugPredictionService
from config.model_config import MODEL_CONFIG

def main():
    """Chạy toàn bộ pipeline dự báo thuốc"""
    print("=== DRUG PREDICTION MODEL ===")
    
    # 1. Tiền xử lý dữ liệu
    print("1. Đang tiền xử lý dữ liệu...")
    preprocessor = DrugDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data()
    
    # 2. Huấn luyện mô hình
    print("2. Đang huấn luyện mô hình...")
    model = DrugDecisionTree(MODEL_CONFIG)
    model.train(X_train, y_train)
    
    # 3. Đánh giá mô hình
    print("3. Đang đánh giá mô hình...")
    evaluator = ModelEvaluator()
    evaluator.evaluate_model(model.model, X_test, y_test)
    
    # 4. Lưu mô hình
    print("4. Đang lưu mô hình...")
    model.save_model()
    
    # 5. Khởi tạo service dự báo
    print("5. Khởi tạo service dự báo...")
    prediction_service = DrugPredictionService()
    
    # 6. Test dự báo
    print("6. Test dự báo...")
    sample_patient = {
        'Age': 32,
        'Blood_Pressure': 'HIGH',
        'Cholesterol': 'NORMAL',
        'Sodium_to_Potassium': 13,
        'Sex': 'Female'
    }
    
    predicted_drug = prediction_service.predict(sample_patient)
    print(f"Thuốc được dự báo: {predicted_drug}")
    
    print("=== HOÀN THÀNH ===")

if __name__ == "__main__":
    main()