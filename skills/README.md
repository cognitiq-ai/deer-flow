# DeerFlow Skills Repository

This repository contains DeerFlow-compatible skills intended to be used from a separate DeerFlow deployment.

The main skill in this repo is:

- `custom/curriculum-discovery/` — a backward-designed curriculum discovery skill that turns a normalized planning brief into a parser-stable markdown learning progression report

## Repository Layout

DeerFlow expects `public/` and/or `custom/` directories containing skill folders with a `SKILL.md`:

```text
deerflow-skills/
├── README.md
├── public/
└── custom/
    └── curriculum-discovery/
        ├── SKILL.md
        ├── references/
        └── templates/
```

## Simplified deployment guide

Prerequisite: in your DeerFlow checkout you have a valid `config.yaml` and `extensions_config.json` (e.g. `make config`, or copy from `config.example.yaml` / `extensions_config.example.json`). Running `make up` from the DeerFlow repo root also seeds missing files via `scripts/deploy.sh`.

### Quick setup (3 steps)

1. **Mount your skills repo** in `docker/docker-compose.yaml` on both `gateway` and `langgraph`:

```yaml
# Under each service:
volumes:
  - /path/to/your/deerflow-skills:/app/skills:ro
environment:
  - DEER_FLOW_HOST_SKILLS_PATH=/path/to/your/deerflow-skills
```

Use the **same** host directory for the volume source and for `DEER_FLOW_HOST_SKILLS_PATH`. The stock compose sets `DEER_FLOW_HOST_SKILLS_PATH` from `DEER_FLOW_REPO_ROOT`; override that line so the host Docker daemon (DooD) resolves the same tree you mounted at `/app/skills`.

2. **Enable the skill** in DeerFlow’s `extensions_config.json`. The JSON key is the skill’s `name` from `SKILL.md` frontmatter (not only the folder name):

```json
{
  "mcpServers": {},
  "skills": {
    "curriculum-discovery": { "enabled": true }
  }
}
```

If a custom skill has **no** entry under `skills`, DeerFlow defaults it to enabled; an explicit entry is still useful for documentation and toggling.

3. **Start DeerFlow** from the **DeerFlow** repository root:

```bash
make up
```

Or:

```bash
docker compose -f docker/docker-compose.yaml up --build
```

If you use raw `docker compose`, ensure `DEER_FLOW_CONFIG_PATH` and `DEER_FLOW_EXTENSIONS_CONFIG_PATH` point at your `config.yaml` and `extensions_config.json` (or use a `.env` file consistent with `docker/docker-compose.yaml`). `make up` sets these via `scripts/deploy.sh`.

### Critical path alignment

These must stay consistent:

| Role | Path |
|------|------|
| Host directory you bind-mount | `/path/to/your/deerflow-skills` (same value as `DEER_FLOW_HOST_SKILLS_PATH`) |
| Skills directory inside gateway/langgraph containers | `/app/skills` |
| Skills path inside the sandbox (from `config.yaml`) | `/mnt/skills` (default `skills.container_path`) |

### Verify

After services are up, list skills through nginx (default UI port):

```bash
curl http://localhost:2026/api/skills
```

You should see `curriculum-discovery` in the response. To hit the gateway directly:

```bash
curl http://localhost:8001/api/skills
```

### Expected behavior

When paths are aligned, DeerFlow discovers `custom/curriculum-discovery/SKILL.md`, the app reads skills from `/app/skills`, and sandbox tooling resolves resources under `/mnt/skills/custom/curriculum-discovery/...`.

### Simpler case

If this repo already lives on the host at `<deer-flow>/skills`, you do not need custom compose mounts: set `DEER_FLOW_REPO_ROOT` (as `scripts/deploy.sh` does) so the default `DEER_FLOW_HOST_SKILLS_PATH=${DEER_FLOW_REPO_ROOT}/skills` matches that directory.

### Troubleshooting

**Skill missing from `/api/skills`**

1. Bind mount targets `/app/skills` and `custom/curriculum-discovery/SKILL.md` exists.
2. `SKILL.md` frontmatter includes `name` and `description`; `name` matches the `extensions_config.json` key if you add one.
3. Restart `gateway` and `langgraph` after compose changes.

**Skill listed but sandbox cannot read files**

1. `DEER_FLOW_HOST_SKILLS_PATH` on the host matches the volume source path.
2. `skills.container_path` in `config.yaml` remains `/mnt/skills` unless you changed it everywhere consistently.

---
