import pytest
from quota_aware_llm_router.config_loader import ConfigLoader
from quota_aware_llm_router.routing.model_aliases import ModelAliasManager
import os

@pytest.fixture
def temp_config_file(tmp_path):
    config_content = """
providers:
  - id: test-provider
    type: ollama_local
    enabled: true
    priority: 1
    models: ["model-a"]

model_aliases:
  test-model:
    candidates:
      - provider: test-provider
        model: model-a
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return str(config_path)

def test_config_loading(temp_config_file):
    loader = ConfigLoader(temp_config_file)
    config = loader.load()
    assert "providers" in config
    assert config["providers"][0]["id"] == "test-provider"

def test_model_alias_resolution(temp_config_file):
    loader = ConfigLoader(temp_config_file)
    loader.load()
    alias_manager = ModelAliasManager(loader.get_model_aliases())
    
    candidates = alias_manager.resolve("test-model")
    assert len(candidates) == 1
    assert candidates[0].provider == "test-provider"
    assert candidates[0].model == "model-a"

def test_unknown_alias(temp_config_file):
    loader = ConfigLoader(temp_config_file)
    loader.load()
    alias_manager = ModelAliasManager(loader.get_model_aliases())
    
    candidates = alias_manager.resolve("unknown")
    assert len(candidates) == 0
