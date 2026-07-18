"""Deterministic media-player entities for the managed benchmark home."""

from __future__ import annotations

from typing import Any, override

from homeassistant.components.media_player import (
    MediaPlayerDeviceClass,
    MediaPlayerEntity,
)
from homeassistant.components.media_player.browse_media import (
    BrowseMedia,
    SearchMedia,
    SearchMediaQuery,
)
from homeassistant.components.media_player.const import (
    MediaClass,
    MediaPlayerEntityFeature,
    MediaPlayerState,
    MediaType,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

SUPPORTED_FEATURES = (
    MediaPlayerEntityFeature.PAUSE
    | MediaPlayerEntityFeature.VOLUME_SET
    | MediaPlayerEntityFeature.VOLUME_MUTE
    | MediaPlayerEntityFeature.PREVIOUS_TRACK
    | MediaPlayerEntityFeature.NEXT_TRACK
    | MediaPlayerEntityFeature.TURN_ON
    | MediaPlayerEntityFeature.TURN_OFF
    | MediaPlayerEntityFeature.PLAY
    | MediaPlayerEntityFeature.PLAY_MEDIA
    | MediaPlayerEntityFeature.SEARCH_MEDIA
)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Add the benchmark media players."""
    async_add_entities(
        (
            BenchmarkMediaPlayer(
                "Living Room Television",
                "media_living_room_tv",
                MediaPlayerDeviceClass.TV,
            ),
            BenchmarkMediaPlayer(
                "Kitchen Speaker",
                "media_kitchen_speaker",
                MediaPlayerDeviceClass.SPEAKER,
            ),
            BenchmarkMediaPlayer(
                "Bedroom Speaker",
                "media_bedroom_speaker",
                MediaPlayerDeviceClass.SPEAKER,
            ),
        )
    )


class BenchmarkMediaPlayer(MediaPlayerEntity):
    """In-memory media player implementing the intent-related controls."""

    _attr_should_poll = False
    _attr_supported_features = SUPPORTED_FEATURES

    def __init__(
        self,
        name: str,
        unique_id: str,
        device_class: MediaPlayerDeviceClass,
    ) -> None:
        """Initialize an idle media player."""
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._attr_device_class = device_class
        self._attr_state = MediaPlayerState.OFF
        self._attr_volume_level = 0.35
        self._attr_is_volume_muted = False

    @override
    def turn_on(self) -> None:
        """Turn on the player."""
        self._attr_state = MediaPlayerState.IDLE
        self.schedule_update_ha_state()

    @override
    def turn_off(self) -> None:
        """Turn off the player."""
        self._attr_state = MediaPlayerState.OFF
        self.schedule_update_ha_state()

    @override
    def media_play(self) -> None:
        """Start playback."""
        self._attr_state = MediaPlayerState.PLAYING
        self.schedule_update_ha_state()

    @override
    def play_media(
        self,
        media_type: MediaType | str,
        media_id: str,
        **kwargs: Any,
    ) -> None:
        """Play a deterministic in-memory media search result."""
        self._attr_media_content_type = media_type
        self._attr_media_content_id = media_id
        self._attr_state = MediaPlayerState.PLAYING
        self.schedule_update_ha_state()

    @override
    async def async_search_media(self, query: SearchMediaQuery) -> SearchMedia:
        """Return one deterministic playable result for every search query."""
        result = BrowseMedia(
            media_class=MediaClass.MUSIC,
            media_content_id=f"benchmark://{query.search_query}",
            media_content_type=MediaType.MUSIC,
            title=query.search_query,
            can_play=True,
            can_expand=False,
        )
        return SearchMedia(result=[result])

    @override
    def media_pause(self) -> None:
        """Pause playback."""
        self._attr_state = MediaPlayerState.PAUSED
        self.schedule_update_ha_state()

    @override
    def media_previous_track(self) -> None:
        """Acknowledge a previous-track request without external I/O."""
        self.schedule_update_ha_state()

    @override
    def media_next_track(self) -> None:
        """Acknowledge a next-track request without external I/O."""
        self.schedule_update_ha_state()

    @override
    def mute_volume(self, mute: bool) -> None:
        """Set the muted state."""
        self._attr_is_volume_muted = mute
        self.schedule_update_ha_state()

    @override
    def set_volume_level(self, volume: float) -> None:
        """Set the volume level."""
        self._attr_volume_level = volume
        self.schedule_update_ha_state()
