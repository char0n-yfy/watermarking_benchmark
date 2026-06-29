FROM node:22-slim AS deps
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-workspace.yaml /app/
COPY apps/web/package.json /app/apps/web/package.json
RUN pnpm install --filter @wm-bench/web --frozen-lockfile=false

FROM node:22-slim AS builder
WORKDIR /app
RUN corepack enable
ARG NEXT_PUBLIC_API_BASE_URL=/api
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}
COPY --from=deps /app/node_modules /app/node_modules
COPY --from=deps /app/apps/web/node_modules /app/apps/web/node_modules
COPY package.json pnpm-workspace.yaml /app/
COPY apps/web /app/apps/web
RUN pnpm --filter @wm-bench/web build

FROM nginx:1.27-alpine AS runner
COPY infra/docker/web.nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/apps/web/out /usr/share/nginx/html
EXPOSE 3000
