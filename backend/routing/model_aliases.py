from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class Candidate(BaseModel):
    provider: str
    model: str

class ModelAliasManager:
    def __init__(self, aliases: Dict[str, Any]):
        self.aliases = aliases

    def resolve(self, alias_name: str) -> List[Candidate]:
        """Resolve a model alias to a list of potential provider candidates."""
        alias_cfg = self.aliases.get(alias_name)
        if not alias_cfg:
            return []
        
        candidates_raw = alias_cfg.get("candidates", [])
        candidates = []
        for c in candidates_raw:
            candidates.append(Candidate(provider=c["provider"], model=c["model"]))
        return candidates

    def get_all_aliases(self) -> List[str]:
        return list(self.aliases.keys())
