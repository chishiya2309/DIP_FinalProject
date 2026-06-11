import os
import yaml

class EnhancementConfigManager:
    """Quản lý cấu hình của module Video Enhancement.
    
    Hỗ trợ đọc từ file YAML mặc định, cập nhật tham số tại runtime,
    và ghi cấu hình mới xuống file.
    """
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Lấy đường dẫn tuyệt đối đến configs/enhancement.yaml tương đối so với file này
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.config_path = os.path.join(base_dir, "configs", "enhancement.yaml")
        else:
            self.config_path = config_path
            
        self.config = self._get_default_config()
        self.load()

    def _get_default_config(self) -> dict:
        """Cấu hình mặc định phòng trường hợp không đọc được file."""
        return {
            "enhancement": {
                "auto_detect": True,
                "sharpening": {
                    "enabled": True,
                    "method": "unsharp_mask",
                    "strength": 1.5,
                    "blur_kernel": 5,
                    "sigma": 1.0,
                    "sharpness_threshold": 100.0
                },
                "denoising": {
                    "enabled": True,
                    "method": "bilateral",
                    "noise_threshold": 500.0,
                    "bilateral_d": 5,
                    "sigma_color": 75,
                    "sigma_space": 75,
                    "nlmeans_h": 10,
                    "median_kernel": 3
                },
                "backlight": {
                    "enabled": True,
                    "auto_detect_threshold": 0.3,
                    "method": "gamma",
                    "gamma": 1.5,
                    "retinex_sigma": 30.0
                },
                "white_balance": {
                    "enabled": True,
                    "method": "gray_world",
                    "cast_threshold": 20
                },
                "clahe": {
                    "clip_limit": 2.0,
                    "tile_grid_size": [8, 8]
                }
            }
        }

    def load(self):
        """Đọc cấu hình từ file YAML."""
        if not os.path.exists(self.config_path):
            self.save()  # Tạo file mặc định nếu chưa có
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded and "enhancement" in loaded:
                    # Merge để tránh thiếu key mới
                    self._merge_config(self.config, loaded)
        except Exception as e:
            print(f"[EnhancementConfigManager] Lỗi đọc file cấu hình: {e}. Sử dụng cấu hình mặc định.")

    def _merge_config(self, base: dict, loaded: dict):
        """Merge đệ quy loaded dict vào base dict."""
        for k, v in loaded.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._merge_config(base[k], v)
            else:
                base[k] = v

    def save(self):
        """Lưu cấu hình hiện tại xuống file YAML."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"[EnhancementConfigManager] Lỗi lưu file cấu hình: {e}")

    def get_section(self, section_name: str) -> dict:
        """Lấy một section cấu hình cụ thể (e.g. 'sharpening', 'denoising')."""
        return self.config["enhancement"].get(section_name, {})

    def update_param(self, section_name: str, key: str, value):
        """Cập nhật giá trị cấu hình tại runtime."""
        if section_name in self.config["enhancement"]:
            self.config["enhancement"][section_name][key] = value
        elif section_name == "enhancement":
            self.config["enhancement"][key] = value

    @property
    def auto_detect(self) -> bool:
        return self.config["enhancement"].get("auto_detect", True)

    @auto_detect.setter
    def auto_detect(self, value: bool):
        self.config["enhancement"]["auto_detect"] = value
