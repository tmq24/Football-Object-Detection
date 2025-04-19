from ultralytics import YOLO

class ModelProvider:
    _instance = None
    _player_detector = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelProvider, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_player_detector(cls, model_path='weights/best.pt'):
        """Singleton to get player detector model"""
        if cls._player_detector is None:
            print("Loading player detector model...")
            cls._player_detector = YOLO(model_path)
            cls._player_detector.conf = 0.3
        return cls._player_detector
