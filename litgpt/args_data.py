from dataclasses import dataclass
from typing import List, Optional
import yaml


@dataclass
class P2PConfig:
    hf_tokenizer: str = "checkpoints/meta-llama/Meta-Llama-3.1-8B"
    raw_dir: Optional[str] = None
    output_dir: str = "outputs/p2p_tok_default"
    template_fn: str = "default_text"
    num_proc_pull: int = 32
    num_raw_shards: int = 32
    max_retries: int = 50
    retry_patience: int = 30
    limiter: Optional[int] = None
    target_shard_num: int = 4
    max_wait_time: int = 60 * 60 * 6
    wait_interval: int = 30
    num_proc: Optional[int] = None
    target_block_size: int = 2048
    pack_setting: str = "no_pack"
    num_proc_map_to_block: Optional[int] = None
    shuffle_filenames: bool = True
    train_split_pct: float = 0.99
    resume: bool = True
    log_tokenization_progress: bool = True
    rm_cache: bool = True

    @classmethod
    def from_yaml(cls, file_path: str) -> "P2PConfig":
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return cls(**data)


@dataclass
class DataSource:
    address: str
    subset: str = "default"
    split: str = "train"
    template_fn: Optional[str] = None
    license: Optional[str] = None
    citation: Optional[str] = None
    machine_generated: bool = False
    requires_software_heritage_aws_download: Optional[bool] = False
    weight: float = 1.0
    category: str = "generic-web"
    every_token_is_sacred: bool = False

    def fill_empty_citation(self):
        if not self.citation:
            self.citation = f"https://huggingface.co/datasets/{self.address}"

    def fill_empty_license(self):
        if not self.license:
            self.license = "other"


@dataclass
class DataSources:
    sources: dict[str, DataSource]

    @classmethod
    def from_yaml(cls, file_path: str) -> "DataSources":
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        sources = {
            name: DataSource(**source_data) for name, source_data in data.items()
        }
        for source in sources.values():
            source.fill_empty_citation()
            source.fill_empty_license()
        return cls(sources=sources)