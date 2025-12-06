from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict, Field

OrderEntry = Tuple[str, str, int, int]


def _append_message_sequence(
    base: List[AnyMessage], additions: List[AnyMessage]
) -> List[AnyMessage]:
    """Append message sequences, merging consecutive entries of the same type."""

    merged = deepcopy(base or [])
    for message in additions or []:
        if len(merged) == 0:
            merged.append(message)
        elif merged[-1].type == message.type:
            merged[-1].content += f"\n\n{message.content}"
        else:
            merged.append(message)

    return merged


class MessageStore(BaseModel):
    """Structured collection that tracks LangGraph node/call messages in order."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: Dict[str, Dict[str, List[AnyMessage]]] = Field(default_factory=dict)
    order: List[OrderEntry] = Field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return bool(self.data)

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def pop(self, key: str, default=None):
        sentinel = object()
        value = self.data.pop(key, sentinel)
        if value is sentinel:
            return default
        self.order = [entry for entry in self.order if entry[0] != key]
        return value

    def clear_node(self, key: str) -> None:
        self.data.pop(key, None)
        self.order = [entry for entry in self.order if entry[0] != key]

    def _append_entry(self, node: str, call: str, messages: List[AnyMessage]) -> None:
        if not messages:
            return
        node_bucket = self.data.setdefault(node, {})
        existing_seq = node_bucket.get(call, [])
        start_idx = len(existing_seq)
        node_bucket[call] = _append_message_sequence(existing_seq, messages)
        end_idx = len(node_bucket[call])
        if end_idx > start_idx:
            self.order.append((node, call, start_idx, end_idx))

    def append(
        self, node: str, call: str, messages: List[AnyMessage]
    ) -> "MessageStore":
        store = self.copy()
        store._append_entry(node, call, messages)
        return store

    def copy(self) -> "MessageStore":
        return MessageStore(
            data=deepcopy(self.data),
            order=list(self.order),
        )

    @classmethod
    def ensure(cls, value: "MessageStore | dict | None") -> "MessageStore":
        if isinstance(value, MessageStore):
            return value
        if value is None:
            return MessageStore()
        if isinstance(value, dict):
            order = value.get("__order__") or []
            data = {
                key: deepcopy(bucket)
                for key, bucket in value.items()
                if key != "__order__"
            }
            return MessageStore(data=data, order=list(order))
        raise TypeError(f"Unsupported message store payload: {type(value)!r}")

    def merge_with(self, other: "MessageStore | dict | None") -> "MessageStore":
        merged = self.copy()
        merged._merge_inplace(other)
        return merged

    def _merge_inplace(self, other: "MessageStore | dict | None") -> None:
        other_store = MessageStore.ensure(other)
        if not other_store:
            return

        if other_store.order:
            for node, call, start, end in other_store.order:
                call_messages = other_store.data.get(node, {}).get(call, [])
                batch = call_messages[start:end]
                merged_batch = deepcopy(batch)
                self._append_entry(node, call, merged_batch)
            return

        for node, call_bucket in other_store.data.items():
            for call, call_messages in call_bucket.items():
                self._append_entry(node, call, deepcopy(call_messages))

    def flatten(self) -> List[AnyMessage]:
        if self.order:
            flattened: List[AnyMessage] = []
            for node, call, start, end in self.order:
                call_messages = self.data.get(node, {}).get(call, [])
                flattened = _append_message_sequence(
                    flattened, call_messages[start:end]
                )
            return flattened

        flattened: List[AnyMessage] = []
        for call_bucket in self.data.values():
            for call_messages in call_bucket.values():
                flattened = _append_message_sequence(flattened, call_messages or [])
        return flattened


def make_message_entry(
    node: str, call: str, messages: List[AnyMessage]
) -> MessageStore:
    """Create a store containing a single node/call batch."""

    store = MessageStore()
    store._append_entry(node, call, deepcopy(messages or []))
    return store


def merge_message_histories(
    existing: MessageStore | dict | None, new: MessageStore | dict | None
) -> MessageStore:
    """Reducer used by LangGraph to accumulate message batches."""

    base = MessageStore.ensure(existing)
    return base.merge_with(new)


def flatten_message_history(store: MessageStore | dict | None) -> List[AnyMessage]:
    """Flatten the ordered history for presentation."""

    message_store = MessageStore.ensure(store)
    return message_store.flatten()


def prepare_llm_messages(
    history: MessageStore | dict | None,
    new_messages: List[AnyMessage],
    additional: MessageStore | dict | None = None,
) -> List[AnyMessage]:
    """
    Prepare LLM inputs by combining historical context with the latest prompt.
    """

    base_store = MessageStore.ensure(history)
    if additional:
        base_store = base_store.merge_with(additional)

    flattened = base_store.flatten()

    # Always put system messages first so they are included in every LLM call,
    # even if ordering metadata is missing or stale.
    system_buckets = base_store.data.get("system", {})
    if system_buckets:
        system_messages: List[AnyMessage] = []
        for call_messages in system_buckets.values():
            system_messages = _append_message_sequence(
                system_messages, deepcopy(call_messages or [])
            )
        non_system_messages = [msg for msg in flattened if msg.type != "system"]
        flattened = system_messages + non_system_messages

    return _append_message_sequence(flattened, new_messages)
