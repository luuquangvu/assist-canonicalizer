"""Deterministic to-do entity for the managed benchmark home."""

from __future__ import annotations

from typing import override

from homeassistant.components.todo import (
    TodoItem,
    TodoListEntity,
)
from homeassistant.components.todo.const import TodoItemStatus, TodoListEntityFeature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from . import DOMAIN


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Add the benchmark to-do list."""
    entity = BenchmarkTodoList()
    hass.data[DOMAIN]["todo_entity"] = entity
    async_add_entities((entity,))


class BenchmarkTodoList(TodoListEntity):
    """In-memory list implementing the generic list intent operations."""

    _attr_should_poll = False
    _attr_supported_features = (
        TodoListEntityFeature.CREATE_TODO_ITEM
        | TodoListEntityFeature.DELETE_TODO_ITEM
        | TodoListEntityFeature.UPDATE_TODO_ITEM
    )

    def __init__(self) -> None:
        """Initialize an empty deterministic list."""
        self._attr_name = "Benchmark Shopping List"
        self._attr_unique_id = "benchmark_todo_list"
        self._items: list[TodoItem] = []
        self._attr_todo_items = self._items
        self._next_uid = 1

    def reset_items(self, item: str | None) -> None:
        """Reset the list, optionally seeding one actionable item."""
        self._next_uid = 1
        self._items.clear()
        if item is not None:
            self._items.append(
                TodoItem(
                    summary=item,
                    uid=self._new_uid(),
                    status=TodoItemStatus.NEEDS_ACTION,
                )
            )
        self.async_write_ha_state()

    def _new_uid(self) -> str:
        """Return a stable monotonically increasing item identifier."""
        uid = f"benchmark-item-{self._next_uid}"
        self._next_uid += 1
        return uid

    @override
    async def async_create_todo_item(self, item: TodoItem) -> None:
        """Add an item."""
        self._items.append(
            TodoItem(
                summary=item.summary,
                uid=item.uid or self._new_uid(),
                status=item.status or TodoItemStatus.NEEDS_ACTION,
            )
        )
        self.async_write_ha_state()

    @override
    async def async_update_todo_item(self, item: TodoItem) -> None:
        """Update an existing item by uid."""
        self._items[:] = [
            item if existing.uid == item.uid else existing for existing in self._items
        ]
        self.async_write_ha_state()

    @override
    async def async_delete_todo_items(self, uids: list[str]) -> None:
        """Delete matching items."""
        deleted = set(uids)
        self._items[:] = [item for item in self._items if item.uid not in deleted]
        self.async_write_ha_state()
