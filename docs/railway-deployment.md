# Railway Deployment Runbook

This fork deploys Open Notebook from the `railway` branch to Railway behind a GitHub organization OAuth gateway.

## Branch Model

- `main` stays aligned with `lfnovo/open-notebook`.
- `railway` is the deployable branch and the repository default branch.
- Upstream changes are merged into `railway` through reviewed merge commits. Do not squash or rebase upstream sync PRs.
- Railway deploys only after CI succeeds on `railway`.

## Railway Topology

- `gateway` is the only public service. It runs OAuth2 Proxy and restricts access to the `Venture-Formations` GitHub organization.
- `open-notebook` is private, listens on port `8502`, and runs the upstream Dockerfile without a custom start command.
- `surrealdb` is private, listens on port `8000`, and stores data on `/mydata`.
  It builds from `Dockerfile.surrealdb`, which pins the tested SurrealDB image digest and runs as root so Railway's mounted volume is writable.
- `open-notebook` stores uploads and local checkpoints on `/app/data`.

Do not create public domains or TCP proxies for `open-notebook` or `surrealdb`.

## Required Railway Variables

`surrealdb`:

- `PORT=8000`
- `SURREAL_USER=root`
- `SURREAL_PASS=<sealed secret>`
- `SURREAL_SYNC_DATA=true`

`open-notebook`:

- `PORT=8502`
- `HOSTNAME=0.0.0.0`
- `INTERNAL_API_URL=http://localhost:5055`
- `API_URL=https://${{gateway.RAILWAY_PUBLIC_DOMAIN}}`
- `CORS_ORIGINS=https://${{gateway.RAILWAY_PUBLIC_DOMAIN}}`
- `SURREAL_URL=ws://${{surrealdb.RAILWAY_PRIVATE_DOMAIN}}:8000/rpc`
- `SURREAL_USER=${{surrealdb.SURREAL_USER}}`
- `SURREAL_PASSWORD=${{surrealdb.SURREAL_PASS}}`
- `SURREAL_NAMESPACE=open_notebook`
- `SURREAL_DATABASE=open_notebook`
- `OPEN_NOTEBOOK_ENCRYPTION_KEY=<sealed secret>`
- `OPEN_NOTEBOOK_PASSWORD=<sealed secret>`

`gateway`:

- `PORT=4180`
- `OAUTH2_PROXY_HTTP_ADDRESS=0.0.0.0:4180`
- `OAUTH2_PROXY_UPSTREAMS=http://open-notebook.railway.internal:8502`
- `OAUTH2_PROXY_PROVIDER=github`
- `OAUTH2_PROXY_GITHUB_ORG=Venture-Formations`
- `OAUTH2_PROXY_EMAIL_DOMAINS=*`
- `OAUTH2_PROXY_COOKIE_SECURE=true`
- `OAUTH2_PROXY_COOKIE_SECRET=<sealed secret>`
- `OAUTH2_PROXY_CLIENT_ID=<GitHub OAuth app client id>`
- `OAUTH2_PROXY_CLIENT_SECRET=<GitHub OAuth app client secret>`
- `OAUTH2_PROXY_REDIRECT_URL=https://${{gateway.RAILWAY_PUBLIC_DOMAIN}}/oauth2/callback`

## Recovery Rules

- Back up both Railway volumes: `/mydata` and `/app/data`.
- Keep the Open Notebook encryption key outside Railway. Losing it makes stored provider credentials undecryptable.
- Before merging upstream changes that touch `open_notebook/database/migrations`, confirm a fresh backup exists.
- Rollback after a migration requires restoring the app revision and both data stores together.
- Test restore into clean volumes before considering backups trustworthy.

## Smoke Tests

- Visiting the public Railway domain redirects to GitHub OAuth.
- Users outside `Venture-Formations` are rejected by the gateway.
- After GitHub OAuth, Open Notebook still requires `OPEN_NOTEBOOK_PASSWORD`.
- Direct public access to ports `5055` and `8000` is impossible.
- An unauthenticated request through the gateway to `/api/notebooks` is rejected.
- Uploads and provider credentials survive redeploys.
