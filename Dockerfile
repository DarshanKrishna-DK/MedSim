# Stage 1: Build Frontend
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci || npm install

COPY tsconfig*.json vite.config.ts index.html ./
COPY public/ ./public/
COPY src/ ./src/

RUN npm run build

# Stage 2: Python Backend + Runtime
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app/backend

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
COPY --from=builder /app/dist /app/dist
COPY --from=builder /app/src/data/radiologyCases.json /app/src/data/radiologyCases.json

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips=* --ws-ping-interval 20 --ws-ping-timeout 20"]
