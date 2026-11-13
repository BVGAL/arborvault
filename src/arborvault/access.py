"""Document authorization context, bound only by authenticated server code."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    subject: str = ""
    manage_acl: bool = False


ANONYMOUS = Principal()
CURRENT_PRINCIPAL = ContextVar("arborvault_principal", default=ANONYMOUS)


def initial_acl(visibility="namespace"):
    subject = CURRENT_PRINCIPAL.get().subject
    members = ["*"] if visibility == "namespace" else [subject]
    return {"owner": subject, **{op: members.copy() for op in ("read", "write", "delete")}}


def allowed(acl, operation="read"):
    principal = CURRENT_PRINCIPAL.get()
    if acl["owner"] and acl["owner"] == principal.subject:
        return True
    if operation == "manage":
        return principal.manage_acl
    return "*" in acl[operation] or principal.subject in acl[operation]


class AuthorizedStore:
    """Bind an actor inside each synchronous worker call; never accept request headers as identity."""

    def __init__(self, store, principal):
        self.store, self.principal = store, principal

    @contextmanager
    def snapshot(self, *args):
        token = CURRENT_PRINCIPAL.set(self.principal)
        try:
            with self.store.snapshot(*args) as snapshot:
                yield snapshot
        finally:
            CURRENT_PRINCIPAL.reset(token)

    def __getattr__(self, name):
        method = getattr(self.store, name)

        def bound(*args, **kwargs):
            token = CURRENT_PRINCIPAL.set(self.principal)
            try:
                return method(*args, **kwargs)
            finally:
                CURRENT_PRINCIPAL.reset(token)

        return bound


@dataclass
class Scope:
    tenant: str
    namespace: str
    store: AuthorizedStore

    def __iter__(self):
        return iter((self.tenant, self.namespace))
