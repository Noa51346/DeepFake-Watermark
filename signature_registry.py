"""
VeriFrame — Signature Registry

Stores the mapping between signature names (public) and
encryption keys (secret). This allows verification by
signature name without knowing the encryption key.

Format: JSON file with {signature_name: encryption_key}
"""

import json
import os

REGISTRY_FILE = "signatures.json"


def _load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_registry(registry: dict):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def register_signature(name: str, key: str):
    """Save a signature name → encryption key mapping."""
    registry = _load_registry()
    registry[name] = key
    _save_registry(registry)


def get_key_for_signature(name: str) -> str:
    """Look up the encryption key for a given signature name."""
    registry = _load_registry()
    return registry.get(name, "")


def list_signatures() -> list:
    """Return all registered signature names."""
    registry = _load_registry()
    return sorted(registry.keys())


def signature_exists(name: str) -> bool:
    """Check if a signature name is already registered."""
    registry = _load_registry()
    return name in registry
