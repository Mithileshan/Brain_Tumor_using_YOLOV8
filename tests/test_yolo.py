"""
Unit tests for YOLO package
Phase 6: Quality assurance
"""

import pytest
from src.bt_yolo.config import Config


class TestYOLOConfig:
    """Test YOLO configuration"""
    
    def test_config_creation(self):
        """Test config creation"""
        config = Config(data_yaml="data.yaml", epochs=100)
        assert config.data_yaml == "data.yaml"
        assert config.epochs == 100
    
    def test_config_dict(self):
        """Test config to dict"""
        config = Config()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
    
    def test_config_validation_epochs(self):
        """Test config validation"""
        config = Config(epochs=-1)
        with pytest.raises(ValueError):
            config.validate()
    
    def test_config_validation_imgsz(self):
        """Test imgsz must be divisible by 32"""
        config = Config(imgsz=641)  # Not divisible by 32
        with pytest.raises(ValueError):
            config.validate()
    
    def test_config_valid_imgsz(self):
        """Test valid imgsz"""
        config = Config(imgsz=640)
        config.validate()  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
