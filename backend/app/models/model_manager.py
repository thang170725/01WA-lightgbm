import os
import joblib

# [Ghi chú mới] Lớp quản lý việc lưu (save) và tải (load) mô hình học máy kèm kết quả đánh giá (.pkl)
class ModelManager:
    # [Input] :
    #   - model_dir (str): Đường dẫn thư mục dùng để lưu trữ các file model (mặc định "models")
    # [Output]:
    #   - None (Khởi tạo thuộc tính self.model_dir và tạo thư mục nếu chưa có)
    def __init__(self,
        model_dir="models"
    ):
        self.model_dir = model_dir

        # tạo folder nếu chưa có
        os.makedirs(self.model_dir, exist_ok=True)

    # ====== save model =======
    # [Input] :
    #   - model (object)  : Đối tượng mô hình đã huấn luyện (LightGBM hoặc Ridge)
    #   - evaluate (dict) : Dictionary chứa các chỉ số đánh giá (mae, relative_mae, mape, rmse, r2)
    #   - save_path (str) : Tên file `.pkl` sẽ lưu trong thư mục `model_dir`
    # [Output]:
    #   - None (Lưu dictionary {'model': model, 'evaluate': evaluate} xuống file .pkl)
    def save_model(self,
        model,
        evaluate: dict,
        save_path="lightgbm_model.pkl"
    ):
        path = os.path.join(self.model_dir, save_path)

        # [Ghi chú mới] Đóng gói cả đối tượng model và các chỉ số evaluate vào chung 1 file .pkl
        joblib.dump({
            'model': model,
            'evaluate': evaluate
        }, path)

        print(f"\nModel saved to: {path}")

    # load model
    # [Input] :
    #   - name (str): Tên file `.pkl` cần nạp từ thư mục `model_dir`
    # [Output]:
    #   - dict: Dictionary gồm 2 phần tử {'model': đối tượng model, 'evaluate': dict các chỉ số đánh giá}
    def load_model(self,
        name="lightgbm_model.pkl"
    ):
        path = os.path.join(self.model_dir, name)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        print(f"\nLoading model from: {path}")

        return joblib.load(path)
