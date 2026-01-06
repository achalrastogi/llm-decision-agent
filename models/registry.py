"""
Model registry loader and utilities
"""

import yaml
from pathlib import Path
from typing import Dict, List
from .schemas import ModelInfo


class ModelRegistry:
    """Loads and manages the static model registry"""
    
    def __init__(self, registry_path: str = "models/model_registry.yaml"):
        self.registry_path = Path(registry_path)
        self._models: Dict[str, ModelInfo] = {}
        self.load_registry()
    
    def load_registry(self) -> None:
        """Load models from YAML registry file"""
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            for model_id, model_data in data['models'].items():
                self._models[model_id] = ModelInfo(**model_data)
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Model registry not found at {self.registry_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in model registry: {e}")
        except Exception as e:
            raise ValueError(f"Error loading model registry: {e}")
    
    def get_all_models(self) -> Dict[str, ModelInfo]:
        """Get all models from registry"""
        return self._models.copy()
    
    def get_model(self, model_id: str) -> ModelInfo:
        """Get specific model by ID"""
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self._models[model_id]
    
    def get_models_by_provider(self, provider: str) -> Dict[str, ModelInfo]:
        """Get all models from a specific provider"""
        return {
            model_id: model_info 
            for model_id, model_info in self._models.items()
            if model_info.provider.lower() == provider.lower()
        }
    
    def get_model_names(self) -> List[str]:
        """Get list of all model names"""
        return list(self._models.keys())
    
    def get_providers(self) -> List[str]:
        """Get list of all unique providers"""
        return list(set(model.provider for model in self._models.values()))
    
    def reload_registry(self) -> None:
        """Reload the registry from file"""
        self._models.clear()
        self.load_registry()