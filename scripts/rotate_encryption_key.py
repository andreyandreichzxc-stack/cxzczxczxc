#!/usr/bin/env python3
"""Rotate ENCRYPTION_KEY for API keys in the database.

Usage:
    1. Set OLD_ENCRYPTION_KEY and NEW_ENCRYPTION_KEY in .env (or pass as args)
    2. python scripts/rotate_encryption_key.py
    3. Update ENCRYPTION_KEY in .env to the new key
    4. Restart the bot

This script:
- Generates a new Fernet key if NEW_ENCRYPTION_KEY is not set
- Reads all api_keys and llm_key_slots from the database
- Decrypts each key with the old Fernet key
- Re-encrypts with the new Fernet key
- Updates the database in-place
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptography.fernet import Fernet


async def rotate(
    old_key: str,
    new_key: str,
    db_url: str,
) -> None:
    from sqlalchemy import select, update
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

    old_fernet = Fernet(old_key.encode())
    new_fernet = Fernet(new_key.encode())

    engine = create_async_engine(db_url)
    total_rotated = 0
    errors = 0

    async with AsyncSession(engine) as session:
        # Rotate api_keys.key_enc
        from src.db.models._auth import ApiKey, LlmKeySlot

        for model_cls, label in [(ApiKey, "api_keys"), (LlmKeySlot, "llm_key_slots")]:
            rows = (await session.execute(select(model_cls))).scalars().all()
            for row in rows:
                try:
                    plaintext = old_fernet.decrypt(row.key_enc.encode()).decode()
                    row.key_enc = new_fernet.encrypt(plaintext.encode()).decode()
                    total_rotated += 1
                except Exception as e:
                    errors += 1
                    print(f"  WARN: {label} id={row.id}: {e}")

        await session.commit()

    await engine.dispose()
    print(f"\nDone: {total_rotated} keys rotated, {errors} errors")
    if errors:
        print("WARNING: Some keys could not be rotated. Check logs above.")
    else:
        print("SUCCESS: All keys rotated. Update ENCRYPTION_KEY in .env and restart.")


def main() -> None:
    import os
    from pathlib import Path

    # Try to load .env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    old_key = os.environ.get("OLD_ENCRYPTION_KEY", "")
    new_key = os.environ.get("NEW_ENCRYPTION_KEY", "")
    db_url = os.environ.get("DATABASE_URL", "")

    if not old_key:
        print("ERROR: Set OLD_ENCRYPTION_KEY (current key from .env)")
        sys.exit(1)
    if not db_url:
        print("ERROR: Set DATABASE_URL")
        sys.exit(1)

    if not new_key:
        new_key = Fernet.generate_key().decode()
        print(f"Generated new key: {new_key}")
        print("SAVE THIS KEY — you'll need it for ENCRYPTION_KEY in .env")

    print(f"Rotating keys in {db_url}...")
    asyncio.run(rotate(old_key, new_key, db_url))


if __name__ == "__main__":
    main()
