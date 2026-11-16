import os
import joblib

# Lớp quản lý lưu trữ (save) và nạp (load) mô hình học máy kèm kết quả đánh giá (.pkl)
class ModelManager:
    def __init__(self,
        model_dir="models"  # Input: đường dẫn thư mục chứa các file model (.pkl)
    ):
        # Input : model_dir (str) - Đường dẫn thư mục lưu/nạp model (mặc định "models")
        # Output: None - Khởi tạo đối tượng ModelManager và tự động tạo thư mục nếu chưa tồn tại
        if not os.path.exists(model_dir) and os.path.exists(os.path.join("backend", model_dir)):
            model_dir = os.path.join("backend", model_dir)
        self.model_dir = model_dir

        # tạo folder nếu chưa có
        os.makedirs(self.model_dir, exist_ok=True)

    # ====== save model =======
    def save_model(self,
        model,                      # Input 1: đối tượng mô hình đã huấn luyện (LightGBM / Ridge)
        evaluate: dict,             # Input 2: dictionary chứa các chỉ số đánh giá (mae, relative_mae, mape, rmse, r2)
        save_path="lightgbm_model.pkl"  # Input 3: tên file .pkl cần lưu
    ):
        # Input :
        #   - model (object)  : Mô hình học máy đã train xong
        #   - evaluate (dict) : Bộ chỉ số đánh giá mô hình trên tập Test
        #   - save_path (str) : Tên file lưu trong thư mục self.model_dir
        # Output:
        #   - path (str)      : Đường dẫn file .pkl đã lưu thành công trên ổ cứng
        path = os.path.join(self.model_dir, save_path)

        # Đóng gói cả model và kết quả evaluate vào chung 1 dictionary rồi lưu bằng joblib
        joblib.dump({
            'model': model,
            'evaluate': evaluate
        }, path)

        print(f"\nModel saved to: {path}")
        return path

    # load model
    def load_model(self,
        name="lightgbm_model.pkl"   # Input: tên file model (.pkl) cần tải lên
    ):
        # Input :
        #   - name (str): Tên file .pkl trong thư mục self.model_dir (vd: "lightgbm_model_v2.pkl", "linear_v2.pkl")
        # Output:
        #   - dict      : Dictionary gồm {'model': đối tượng model, 'evaluate': dict các chỉ số đánh giá}
        path = os.path.join(self.model_dir, name)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        print(f"\nLoading model from: {path}")

        return joblib.load(path)
