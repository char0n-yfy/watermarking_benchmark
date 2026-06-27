FROM node:22-slim AS deps
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-workspace.yaml /app/
COPY apps/web/package.json /app/apps/web/package.json
RUN pnpm install --filter @wm-bench/web --frozen-lockfile=false

FROM node:22-slim AS builder
WORKDIR /app
RUN corepack enable
COPY --from=deps /app/node_modules /app/node_modules
COPY --from=deps /app/apps/web/node_modules /app/apps/web/node_modules
COPY package.json pnpm-workspace.yaml /app/
COPY apps/web /app/apps/web
RUN pnpm --filter @wm-bench/web build

FROM node:22-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/apps/web/.next/standalone ./
COPY --from=builder /app/apps/web/.next/static ./apps/web/.next/static
COPY --from=builder /app/apps/web/public ./apps/web/public
EXPOSE 3000
CMD ["node", "apps/web/server.js"]
