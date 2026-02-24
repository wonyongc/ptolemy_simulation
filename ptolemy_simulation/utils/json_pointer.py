"""Minimal JSON pointer setter/getter utilities."""

from __future__ import annotations

from typing import Any, MutableMapping, MutableSequence


class JsonPointerError(ValueError):
    """Raised when a JSON pointer operation fails."""


def _decode_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _tokens(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise JsonPointerError(f"JSON pointer must start with '/': {pointer}")
    return [_decode_token(t) for t in pointer.lstrip("/").split("/")]


def set_pointer(document: Any, pointer: str, value: Any) -> None:
    tokens = _tokens(pointer)
    if not tokens:
        raise JsonPointerError("Cannot assign to document root with empty pointer")

    cursor = document
    for idx, token in enumerate(tokens[:-1]):
        next_token = tokens[idx + 1]
        if isinstance(cursor, MutableMapping):
            if token not in cursor:
                cursor[token] = [] if next_token.isdigit() else {}
            cursor = cursor[token]
        elif isinstance(cursor, MutableSequence):
            if not token.isdigit():
                raise JsonPointerError(f"Expected numeric index at token '{token}'")
            index = int(token)
            if index >= len(cursor):
                raise JsonPointerError(f"List index out of range: {token}")
            cursor = cursor[index]
        else:
            raise JsonPointerError(f"Pointer traversal hit non-container at token '{token}'")

    last = tokens[-1]
    if isinstance(cursor, MutableMapping):
        cursor[last] = value
        return
    if isinstance(cursor, MutableSequence):
        if not last.isdigit():
            raise JsonPointerError(f"Expected numeric final token for list assignment: {last}")
        index = int(last)
        if index >= len(cursor):
            raise JsonPointerError(f"List index out of range: {last}")
        cursor[index] = value
        return
    raise JsonPointerError("Pointer target parent is not a container")
