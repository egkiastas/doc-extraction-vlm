import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    seed: int = 0
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    ollama_num_ctx: int = 4096
    vlm_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    vlm_adapter: str = ""
    vlm_max_pixels: int = 1_310_720
    vlm_max_new_tokens: int = 1024
    donut_model: str = "naver-clova-ix/donut-base-finetuned-cord-v2"
    api_base_url: str = ""
    api_model: str = ""
    api_key: str = ""
    api_usd_per_mtok_in: float = 0.0
    api_usd_per_mtok_out: float = 0.0
    cost_usd_per_hour_mps: float = 0.0
    cost_usd_per_hour_cuda: float = 0.0
    cost_usd_per_hour_cpu: float = 0.0
    service_extractor: str = "ocr_llm"
    service_replay_file: str = "tests/fixtures/recorded.jsonl"
    service_db_path: str = "data/review.db"
    service_corrections_dir: str = "data/corrections"
    service_confidence_file: str = "results/confidence_model.json"


def load_config(path: str | Path = ROOT / "config.toml") -> Config:
    """Flatten `[section] key` into `section_key`, then let env vars override."""
    raw = tomllib.loads(Path(path).read_text())
    flat: dict = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update({f"{key}_{k}": v for k, v in value.items()})
        else:
            flat[key] = value
    cfg = Config()
    for f in fields(Config):
        value = os.environ.get(f.name.upper(), flat.get(f.name, getattr(cfg, f.name)))
        setattr(cfg, f.name, type(getattr(cfg, f.name))(value))
    return cfg
