# Copilot instructions — HA-Azure-AI-tasks

> Canonical standards live in the `dev-standards` repo on SOUNDWAVE/Gitea.
> Read by Copilot chat **and** inline suggestions. For full HA build conventions,
> see the `build-ha-component` skill in dev-standards.

## What this repo is

A **Home Assistant custom component** providing an **AI Task** entity backed by
Azure OpenAI. Domain: `azure_ai_tasks`.

## Repo shape

- `custom_components/azure_ai_tasks/` — `manifest.json`, `__init__.py`,
  `config_flow.py`, `const.py`, `ai_task.py` (the AI Task platform, large),
  `strings.json`.
- `hacs.json`, `info.md`, `.github/workflows/` (`release.yaml` + `validate.yaml`).

## Conventions

- Bump `manifest.json` **version** every release (semver); `domain` matches the
  folder name.
- Test: `hassfest` + HACS validation, then `pytest` with
  `pytest-homeassistant-custom-component`.
- Deploy/test via the published release artifact into TEST1/TEST2, not host
  file-copy. Backup + auto-rollback.
- The **Azure OpenAI endpoint + API key are user config** (entered in the config
  flow) — never commit them, deployment names, or resource URLs tied to a private
  tenant.

## Never

- Don't commit Azure keys/endpoints, HA tokens, or deploy keys — Gitea Actions
  secrets only.
