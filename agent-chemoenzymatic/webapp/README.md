# `webapp/`

A local web UI for driving the agent and reviewing the routes it saves. The
client is a Vite/React app built on Ketcher for structure drawing; the server is a
FastAPI app that starts runs, streams their progress and serves saved sessions.

```bash
# server (from the harness root)
export AGENT_CONFIG=deployment.local.yaml
uvicorn webapp.server.main:app --port 8000

# client (dev server, proxies /api to :8000)
cd webapp/client && npm install && npm run dev
# or: npm run build   for a static bundle
```

`package-lock.json` is included, `node_modules` is not. The UI uses the same deployment YAML as the CLI. The selected UI preset sets the coordinator model and search budget. See the [deployment guide](../../README.md#configure-a-deployment).
