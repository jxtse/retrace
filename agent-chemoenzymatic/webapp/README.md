# `webapp/`

A local web UI for driving the agent and reviewing the routes it saves. The
client is a Vite/React app built on Ketcher for structure drawing; the server is a
FastAPI app that starts runs, streams their progress and serves saved sessions.

```bash
# server (from the harness root)
uvicorn webapp.server.main:app --port 8000

# client (dev server, proxies /api to :8000)
cd webapp/client && npm install && npm run dev
# or: npm run build   for a static bundle
```

`package-lock.json` is included, `node_modules` is not. The UI is a convenience
and needs the same endpoints as the CLI (`LLM_API_KEY`, `RETROPLANNER_HTTP_URL`,
`REACTION_RATER_URL`); see `.env.example`.
