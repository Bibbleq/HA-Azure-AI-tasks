"""Config flow for Azure AI Tasks integration."""
from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)
from homeassistant.core import callback, HomeAssistant

from .const import (
    CONF_API_KEY,
    CONF_ENDPOINT,
    CONF_CHAT_MODEL,
    CONF_IMAGE_MODEL,
    CONF_USE_RESPONSES_API,
    CONF_ENABLE_WEB_SEARCH,
    DEFAULT_NAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_IMAGE_MODEL,
    DOMAIN,
    CHAT_MODELS,
    IMAGE_MODELS
)

_LOGGER = logging.getLogger(__name__)

# Placeholder options used to explicitly select "no model" in the pickers.
NONE_CHAT_LABEL = "[None - leave empty to disable chat]"
NONE_IMAGE_LABEL = "[None - leave empty to disable images]"

# Legacy data-plane deployments listing - the only deployment-name source that
# works with just the endpoint + api-key (the official listing API lives on the
# ARM management plane and needs Entra credentials).
DEPLOYMENTS_API_VERSION = "2023-03-15-preview"
LIST_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)
# A v1 /models result much larger than any realistic deployment count means we
# got the base-model catalogue instead of deployments - discard it.
MAX_PLAUSIBLE_DEPLOYMENTS = 50


STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_NAME, default=DEFAULT_NAME): str,
        vol.Required(CONF_ENDPOINT): str,
        vol.Required(CONF_API_KEY): str,
    }
)


def _clean_model(value: Any) -> str:
    """Normalize a model picker value ('[None...' placeholders become empty)."""
    value = str(value or "").strip()
    if value.startswith("[None"):
        return ""
    return value


def _model_selector(models: list[str], none_label: str, current: str = "") -> SelectSelector:
    """Build a model dropdown that also accepts typed custom values."""
    options = [none_label, *models]
    if current and current not in options:
        options.append(current)
    return SelectSelector(
        SelectSelectorConfig(
            options=options,
            custom_value=True,
            mode=SelectSelectorMode.DROPDOWN,
        )
    )


async def _async_list_deployments(
    hass: HomeAssistant, endpoint: str, api_key: str
) -> list[str]:
    """Best-effort fetch of the model deployment names on an Azure resource.

    Returns an empty list on any failure - the pickers accept custom values,
    so listing is a convenience, never a requirement.
    """
    if not endpoint or not api_key:
        return []

    session = async_get_clientsession(hass)
    endpoint = endpoint.rstrip("/")
    headers = {"api-key": api_key}

    # Preferred: legacy data-plane deployments list (real deployment names).
    try:
        async with session.get(
            f"{endpoint}/openai/deployments",
            headers=headers,
            params={"api-version": DEPLOYMENTS_API_VERSION},
            timeout=LIST_REQUEST_TIMEOUT,
        ) as response:
            if response.status == 200:
                result = await response.json()
                names = sorted(
                    {item.get("id") for item in result.get("data", []) if item.get("id")}
                )
                if names:
                    _LOGGER.debug("Found %d deployments via deployments API", len(names))
                    return names
            else:
                _LOGGER.debug(
                    "Deployments listing returned status %s", response.status
                )
    except Exception as err:  # pylint: disable=broad-except
        _LOGGER.debug("Deployments listing failed: %s", err)

    # Fallback: v1 models list. On some resources this returns deployments; on
    # others it returns the full base-model catalogue, which we discard.
    try:
        async with session.get(
            f"{endpoint}/openai/v1/models",
            headers=headers,
            timeout=LIST_REQUEST_TIMEOUT,
        ) as response:
            if response.status == 200:
                result = await response.json()
                names = sorted(
                    {item.get("id") for item in result.get("data", []) if item.get("id")}
                )
                if names and len(names) <= MAX_PLAUSIBLE_DEPLOYMENTS:
                    _LOGGER.debug("Found %d models via v1 models API", len(names))
                    return names
                if names:
                    _LOGGER.debug(
                        "v1 models API returned %d entries - looks like the base-model"
                        " catalogue, ignoring",
                        len(names),
                    )
            else:
                _LOGGER.debug("v1 models listing returned status %s", response.status)
    except Exception as err:  # pylint: disable=broad-except
        _LOGGER.debug("v1 models listing failed: %s", err)

    return []


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Azure AI Tasks."""

    VERSION = 2  # Increment version to trigger migration

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._user_data: dict[str, Any] = {}
        self._available_models: list[str] = []

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Create the options flow."""
        return OptionsFlowHandler(config_entry)

    async def async_step_import(self, import_data: dict[str, Any]) -> FlowResult:
        """Handle migration from version 1 to 2."""
        # Clean up deprecated models during migration
        if "chat_model" in import_data:
            if import_data["chat_model"] == "gpt-35-turbo":
                import_data["chat_model"] = ""  # Remove deprecated model

        # Create new entry with migrated data
        return self.async_create_entry(
            title=import_data.get("name", "Azure AI Tasks"),
            data=import_data
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step: connection details."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                await self._test_credentials(
                    user_input[CONF_ENDPOINT], user_input[CONF_API_KEY]
                )
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Credential validation failed")
                errors["base"] = "cannot_connect"
            else:
                self._user_data = user_input
                self._available_models = await _async_list_deployments(
                    self.hass, user_input[CONF_ENDPOINT], user_input[CONF_API_KEY]
                )
                return await self.async_step_models()

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    async def async_step_models(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the second step: model selection and API options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            chat_model = _clean_model(user_input.get(CONF_CHAT_MODEL, ""))
            image_model = _clean_model(user_input.get(CONF_IMAGE_MODEL, ""))

            if not chat_model and not image_model:
                errors["base"] = "no_models_configured"
            else:
                data = {
                    **self._user_data,
                    CONF_CHAT_MODEL: chat_model,
                    CONF_IMAGE_MODEL: image_model,
                    CONF_USE_RESPONSES_API: user_input.get(CONF_USE_RESPONSES_API, False),
                    CONF_ENABLE_WEB_SEARCH: user_input.get(CONF_ENABLE_WEB_SEARCH, False),
                }
                return self.async_create_entry(
                    title=self._user_data.get(CONF_NAME, DEFAULT_NAME), data=data
                )

        schema = vol.Schema(
            {
                vol.Optional(CONF_CHAT_MODEL, default=NONE_CHAT_LABEL): _model_selector(
                    self._available_models, NONE_CHAT_LABEL
                ),
                vol.Optional(CONF_IMAGE_MODEL, default=NONE_IMAGE_LABEL): _model_selector(
                    self._available_models, NONE_IMAGE_LABEL
                ),
                vol.Optional(CONF_USE_RESPONSES_API, default=False): bool,
                vol.Optional(CONF_ENABLE_WEB_SEARCH, default=False): bool,
            }
        )
        return self.async_show_form(
            step_id="models", data_schema=schema, errors=errors
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Reconfigure an existing entry (endpoint, API key, name, models).

        Lets the user update the connection details in place - e.g. when the
        Azure endpoint or API key changes - without removing and re-adding the
        integration.
        """
        reconfigure_entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}

        if user_input is not None:
            chat_model = _clean_model(user_input.get(CONF_CHAT_MODEL, ""))
            image_model = _clean_model(user_input.get(CONF_IMAGE_MODEL, ""))

            if not chat_model and not image_model:
                errors["base"] = "no_models_configured"
            else:
                try:
                    await self._test_credentials(
                        user_input[CONF_ENDPOINT], user_input[CONF_API_KEY]
                    )
                except Exception:  # pylint: disable=broad-except
                    _LOGGER.exception("Reconfigure credential validation failed")
                    errors["base"] = "cannot_connect"
                else:
                    return self.async_update_reload_and_abort(
                        reconfigure_entry,
                        title=user_input.get(CONF_NAME, DEFAULT_NAME),
                        data_updates={
                            CONF_NAME: user_input.get(CONF_NAME, DEFAULT_NAME),
                            CONF_ENDPOINT: user_input[CONF_ENDPOINT],
                            CONF_API_KEY: user_input[CONF_API_KEY],
                            CONF_CHAT_MODEL: chat_model,
                            CONF_IMAGE_MODEL: image_model,
                            CONF_USE_RESPONSES_API: user_input.get(CONF_USE_RESPONSES_API, False),
                            CONF_ENABLE_WEB_SEARCH: user_input.get(CONF_ENABLE_WEB_SEARCH, False),
                        },
                        # Models now live in data; clear any stale options copy so
                        # they can't shadow the reconfigured values.
                        options={},
                    )

        # Pre-fill from the current entry (options take precedence over data, as
        # the options flow may have updated the model selections).
        current = {**reconfigure_entry.data, **reconfigure_entry.options}
        if user_input is not None:
            current = {**current, **user_input}

        # Offer the deployments on the currently stored endpoint/key. If the
        # user is changing credentials the list may be stale, but the pickers
        # accept typed values so nothing is blocked.
        available_models = await _async_list_deployments(
            self.hass,
            reconfigure_entry.data.get(CONF_ENDPOINT, ""),
            reconfigure_entry.data.get(CONF_API_KEY, ""),
        )

        current_chat = _clean_model(current.get(CONF_CHAT_MODEL, ""))
        current_image = _clean_model(current.get(CONF_IMAGE_MODEL, ""))

        schema = vol.Schema(
            {
                vol.Required(CONF_NAME, default=current.get(CONF_NAME, DEFAULT_NAME)): str,
                vol.Required(CONF_ENDPOINT, default=current.get(CONF_ENDPOINT, "")): str,
                vol.Required(CONF_API_KEY, default=current.get(CONF_API_KEY, "")): str,
                vol.Optional(
                    CONF_CHAT_MODEL, default=current_chat or NONE_CHAT_LABEL
                ): _model_selector(available_models, NONE_CHAT_LABEL, current_chat),
                vol.Optional(
                    CONF_IMAGE_MODEL, default=current_image or NONE_IMAGE_LABEL
                ): _model_selector(available_models, NONE_IMAGE_LABEL, current_image),
                vol.Optional(CONF_USE_RESPONSES_API, default=bool(current.get(CONF_USE_RESPONSES_API, False))): bool,
                vol.Optional(CONF_ENABLE_WEB_SEARCH, default=bool(current.get(CONF_ENABLE_WEB_SEARCH, False))): bool,
            }
        )
        return self.async_show_form(
            step_id="reconfigure", data_schema=schema, errors=errors
        )

    async def _test_credentials(self, endpoint: str, api_key: str) -> bool:
        """Test if we can authenticate with the host."""
        session = async_get_clientsession(self.hass)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Basic connectivity test to the endpoint
        async with session.get(endpoint, headers=headers) as response:
            if response.status == 401:
                raise Exception("Invalid API key")
            elif response.status >= 400:
                raise Exception("Cannot connect to Azure AI endpoint")

        return True


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for Azure AI Tasks."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle options flow."""
        if user_input is not None:
            _LOGGER.info("Options flow received input: %s", user_input)

            chat_model = _clean_model(user_input.get(CONF_CHAT_MODEL, ""))
            image_model = _clean_model(user_input.get(CONF_IMAGE_MODEL, ""))

            _LOGGER.info("Final values - chat: '%s', image: '%s'", chat_model, image_model)

            # Check if both are empty
            if not chat_model and not image_model:
                _LOGGER.warning("No models configured, showing error")
                errors = {"base": "no_models_configured"}
                return self.async_show_form(
                    step_id="init",
                    data_schema=await self._get_options_schema(),
                    errors=errors,
                )

            # Create the final data
            final_data = {
                CONF_CHAT_MODEL: chat_model,
                CONF_IMAGE_MODEL: image_model,
                CONF_USE_RESPONSES_API: user_input.get(CONF_USE_RESPONSES_API, False),
                CONF_ENABLE_WEB_SEARCH: user_input.get(CONF_ENABLE_WEB_SEARCH, False),
            }

            _LOGGER.info("Saving configuration: %s", final_data)
            return self.async_create_entry(title="", data=final_data)

        _LOGGER.info("Showing options form with current config")
        return self.async_show_form(
            step_id="init",
            data_schema=await self._get_options_schema(),
        )

    async def _get_options_schema(self) -> vol.Schema:
        """Get the options schema."""
        # Get current values from options first, then data, then defaults
        current_chat_model = _clean_model(
            self._config_entry.options.get(CONF_CHAT_MODEL)
            or self._config_entry.data.get(CONF_CHAT_MODEL, "")
        )
        current_image_model = _clean_model(
            self._config_entry.options.get(CONF_IMAGE_MODEL)
            or self._config_entry.data.get(CONF_IMAGE_MODEL, "")
        )

        _LOGGER.info("Schema defaults - chat: '%s', image: '%s'",
                     current_chat_model, current_image_model)

        available_models = await _async_list_deployments(
            self.hass,
            self._config_entry.data.get(CONF_ENDPOINT, ""),
            self._config_entry.data.get(CONF_API_KEY, ""),
        )

        if CONF_USE_RESPONSES_API in self._config_entry.options:
            current_responses = bool(self._config_entry.options[CONF_USE_RESPONSES_API])
        else:
            current_responses = bool(self._config_entry.data.get(CONF_USE_RESPONSES_API, False))

        if CONF_ENABLE_WEB_SEARCH in self._config_entry.options:
            current_web_search = bool(self._config_entry.options[CONF_ENABLE_WEB_SEARCH])
        else:
            current_web_search = bool(self._config_entry.data.get(CONF_ENABLE_WEB_SEARCH, False))

        return vol.Schema(
            {
                vol.Optional(
                    CONF_CHAT_MODEL, default=current_chat_model or NONE_CHAT_LABEL
                ): _model_selector(available_models, NONE_CHAT_LABEL, current_chat_model),
                vol.Optional(
                    CONF_IMAGE_MODEL, default=current_image_model or NONE_IMAGE_LABEL
                ): _model_selector(available_models, NONE_IMAGE_LABEL, current_image_model),
                vol.Optional(CONF_USE_RESPONSES_API, default=current_responses): bool,
                vol.Optional(CONF_ENABLE_WEB_SEARCH, default=current_web_search): bool,
            }
        )
