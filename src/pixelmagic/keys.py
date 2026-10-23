"""Where an image service's API key comes from: the environment, then the keychain.

This is the one place every face (command line, web page, MCP server) and the
Gemini and OpenAI services get a key from. The environment wins, so a person who
already exports `GEMINI_API_KEY` sees no change; a key pasted on the Services
page is saved in the operating system's keychain through `keyring`, never in the
project folder.

A key is never logged, never put in an exception message and never shown in a
repr: `FoundKey.key` is the only place it lives, and it is kept out of `repr`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

LOGGER = logging.getLogger(__name__)

#: The keychain entry's service name; the entry's user name is the service name.
KEYCHAIN_SERVICE = "pixelmagic"

#: The variables each service reads its key from. Owned here so the services
#: import them from this module and not the other way round.
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

#: The second variable the Gemini SDK honours, read after `GEMINI_API_KEY`.
GEMINI_FALLBACK_ENV = "GOOGLE_API_KEY"

#: Where a found key came from.
FROM_ENVIRONMENT = "environment"
FROM_KEYCHAIN = "keychain"

#: Per key-taking service, the environment variables it reads, in order.
ENV_VARIABLES: dict[str, tuple[str, ...]] = {
    "gemini": (GEMINI_API_KEY_ENV, GEMINI_FALLBACK_ENV),
    "openai": (OPENAI_API_KEY_ENV,),
}


class KeyStoreError(ValueError):
    """A key could not be saved or removed, in a sentence a person can read.

    Never carries the key. A `ValueError`, so a face that already turns those
    into a sentence at its boundary needs nothing new.
    """


@dataclass(frozen=True)
class FoundKey:
    """A key and where it came from."""

    key: str = field(repr=False)
    #: `FROM_ENVIRONMENT` or `FROM_KEYCHAIN`.
    source: str
    #: The environment variable it was read from; None for the keychain.
    variable: str | None = None


def is_usable(key: str) -> bool:
    """Whether a key can travel in a request header: printable ASCII, no space or line break.

    The one place this rule is written; the saved key and the environment's are held to it.
    """
    return key.isascii() and key.isprintable() and not any(c.isspace() for c in key)


def refused_variable(service: str) -> str | None:
    """The environment variable that is set to a key `is_usable` refuses, if any.

    Lets the Services page say why a service with a variable set still has no key.
    """
    for variable in ENV_VARIABLES.get(service, ()):
        value = os.environ.get(variable, "").strip()
        if value and not is_usable(value):
            return variable
    return None


def _require_key_service(service: str) -> None:
    """Refuse a service that takes no key (codex, fake) or does not exist."""
    if service not in ENV_VARIABLES:
        raise KeyStoreError(f"the {service} service does not take an API key")


def find_key(service: str) -> FoundKey | None:
    """The service's key: the environment first, then the keychain; None if neither.

    An empty or whitespace-only variable counts as not set, and so does one
    that `is_usable` refuses (it is skipped with a warning naming the variable). A keychain that
    cannot be used counts as holding nothing, with a warning naming the service.
    """
    for variable in ENV_VARIABLES.get(service, ()):
        value = os.environ.get(variable, "").strip()
        if value and not is_usable(value):
            # Never the value: an unusable key makes the HTTP library quote it in its errors.
            LOGGER.warning(
                "the environment variable %s holds a key with a space, line break or "
                "unusual character inside it, so it is not used",
                variable,
            )
        elif value:
            return FoundKey(value, FROM_ENVIRONMENT, variable)
    if service not in ENV_VARIABLES:
        return None
    try:
        stored = keyring.get_password(KEYCHAIN_SERVICE, service)
    except KeyringError as error:
        LOGGER.warning(
            "the keychain could not be read for the %s service (%s)",
            service,
            type(error).__name__,
        )
        return None
    if stored and stored.strip():
        return FoundKey(stored.strip(), FROM_KEYCHAIN)
    return None


def save_key(service: str, key: str) -> None:
    """Save a key in the keychain, replacing any earlier one.

    Surrounding whitespace is stripped; an empty key is refused. Raises
    `KeyStoreError` for a service that takes no key, a key with a space, line
    break or non-ASCII character inside it, or a keychain that fails.
    """
    _require_key_service(service)
    cleaned = key.strip()
    if not cleaned:
        raise KeyStoreError("there is no key to save: paste the key first")
    if not is_usable(cleaned):
        raise KeyStoreError(
            "the key has a space, line break or unusual character inside it; paste it again"
        )
    try:
        keyring.set_password(KEYCHAIN_SERVICE, service, cleaned)
    except KeyringError as error:
        LOGGER.warning(
            "the keychain could not save a key for the %s service (%s)",
            service,
            type(error).__name__,
        )
        raise KeyStoreError(
            "the key could not be saved: this computer's keychain is not available"
        ) from None


def remove_key(service: str) -> None:
    """Remove the service's key from the keychain; nothing saved is not an error.

    Raises `KeyStoreError` for a service that takes no key or a keychain that fails.
    """
    _require_key_service(service)
    try:
        keyring.delete_password(KEYCHAIN_SERVICE, service)
    except PasswordDeleteError:
        return
    except KeyringError as error:
        LOGGER.warning(
            "the keychain could not remove the key for the %s service (%s)",
            service,
            type(error).__name__,
        )
        raise KeyStoreError(
            "the key could not be removed: this computer's keychain is not available"
        ) from None
