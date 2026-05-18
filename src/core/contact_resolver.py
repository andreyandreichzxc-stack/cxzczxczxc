from dataclasses import dataclass

from rapidfuzz import fuzz, process
from telethon import TelegramClient

from src.db.models import Contact, User
from src.db.repo import list_contacts
from src.db.session import get_session
from src.userbot.dialogs import sync_dialogs


@dataclass
class ContactCandidate:
    peer_id: int
    display_name: str
    username: str | None
    peer_kind: str
    score: int

    def label(self) -> str:
        if self.username:
            return f"{self.display_name} (@{self.username})"
        return self.display_name


def _searchable(c: Contact) -> str:
    parts = [c.display_name]
    if c.username:
        parts.append("@" + c.username)
    if c.phone:
        parts.append(c.phone)
    return " | ".join(parts)


async def resolve(
    client: TelegramClient,
    user: User,
    query: str,
    *,
    limit: int = 5,
    min_score: int = 55,
    kinds: tuple[str, ...] = ("user",),
    include_bots: bool = False,
) -> list[ContactCandidate]:
    # По умолчанию ищем только людей. Для каналов/групп kinds расширять явно.
    async with get_session() as session:
        contacts = await list_contacts(session, user, kinds=kinds, include_bots=include_bots)

    if not contacts:
        await sync_dialogs(client, user)
        async with get_session() as session:
            contacts = await list_contacts(session, user, kinds=kinds, include_bots=include_bots)

    if not contacts:
        return []

    choices = {c.peer_id: _searchable(c) for c in contacts}
    raw = process.extract(
        query,
        choices,
        scorer=fuzz.WRatio,
        limit=limit,
    )

    by_id = {c.peer_id: c for c in contacts}
    results: list[ContactCandidate] = []
    for _, score, peer_id in raw:
        if score < min_score:
            continue
        c = by_id[peer_id]
        results.append(
            ContactCandidate(
                peer_id=c.peer_id,
                display_name=c.display_name,
                username=c.username,
                peer_kind=c.peer_kind,
                score=int(score),
            )
        )
    return results
