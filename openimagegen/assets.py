
"""
@file        assets.py
@author      ParisNeo
@description Secure management of LoRA, GGUF, and custom diffusion asset files.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("OpenImageGen.assets")

ALLOWED_EXTENSIONS = {".safetensors", ".gguf", ".ckpt", ".pt", ".bin"}

class AssetManager:
    """Stores user-supplied model files without exposing arbitrary host paths."""

    def __init__(self, root: Path, max_bytes: int = 32 * 1024**3) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def _validate_name(self, name: str) -> str:
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError("Asset name must be a plain filename.")
        if any(character in name for character in '<>:"/\\|?*'):
            raise ValueError("Asset name contains invalid filename characters.")
        if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported asset extension. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        return name

    def path_for(self, name: str, must_exist: bool = True) -> Path:
        name = self._validate_name(name)
        path = (self.root / name).resolve()
        if path.parent != self.root:
            raise ValueError("Asset path escapes the asset directory.")
        if must_exist and (not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS):
            raise FileNotFoundError(name)
        return path

    def list_assets(self) -> List[Dict[str, Any]]:
        assets = []
        for path in sorted(self.root.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            stat = path.stat()
            assets.append({
                "name": path.name,
                "kind": "gguf" if path.suffix.lower() == ".gguf" else "lora",
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
            })
        return assets

    def save(self, name: str, source: Any, size: int) -> Dict[str, Any]:
        name = self._validate_name(name)
        if size > self.max_bytes:
            raise ValueError(f"Asset exceeds the {self.max_bytes}-byte limit.")

        target = self.path_for(name, must_exist=False)
        temporary = self.root / f".{name}.{hashlib.sha256(name.encode()).hexdigest()[:12]}.upload"
        written = 0
        try:
            with temporary.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    written += len(chunk)
                    if written > self.max_bytes:
                        raise ValueError("Asset exceeds the configured upload limit.")
                    output.write(chunk)
            if written == 0:
                raise ValueError("Asset is empty.")
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

        logger.info(f"Stored model asset '{name}' ({written} bytes).")
        return next(asset for asset in self.list_assets() if asset["name"] == name)

    def delete(self, name: str) -> None:
        self.path_for(name).unlink()

    def resolve_source(
        self,
        source: str,
        model_cache_dir: Path,
        weight_name: Optional[str] = None,
    ) -> str:
        """Resolve a local asset name/path or download a Hugging Face weight file.

        Arbitrary URLs are deliberately rejected. Hugging Face repositories use
        the trusted huggingface_hub downloader and its standard cache.
        """
        candidate = Path(source).expanduser()
        if candidate.is_absolute() or candidate.parent != Path("."):
            path = candidate.resolve()
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                return str(path)
            if candidate.parent == Path("."):
                raise FileNotFoundError(source)

        try:
            return str(self.path_for(source))
        except (FileNotFoundError, ValueError):
            pass

        if source.startswith(("http://", "https://")):
            raise ValueError(
                "Direct URLs are not accepted. Upload the file or use a Hugging Face repository."
            )
        if source.count("/") != 1:
            raise FileNotFoundError(f"Unknown model asset '{source}'.")

        from huggingface_hub import hf_hub_download

        filename = weight_name
        if not filename:
            matches = [asset for asset in self.list_assets() if asset["name"] == source]
            if matches:
                return str(self.path_for(source))
            raise ValueError("weight_name is required when downloading a Hugging Face asset.")
        return hf_hub_download(
            repo_id=source,
            filename=filename,
            cache_dir=str(model_cache_dir),
        )
