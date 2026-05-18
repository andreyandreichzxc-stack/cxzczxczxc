from cryptography.fernet import Fernet, InvalidToken

from src.config import settings


_fernet = Fernet(settings.encryption_key.encode())


def encrypt(plaintext: str) -> str:
    return _fernet.encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    try:
        return _fernet.decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:
        raise ValueError("Не удалось расшифровать: неверный ключ или повреждённые данные") from exc
