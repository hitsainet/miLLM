# miLLM Admin UI Dockerfile
# Multi-stage build for optimized production image

# =============================================================================
# Build Stage
# =============================================================================
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Where the in-app Help and the sidebar's book icon point (src/config/manual.ts). Empty keeps the
# default, the published manual; set it to point at a self-hosted copy of manual/.
ARG VITE_MANUAL_BASE_URL=

# Build the application
RUN npm run build

# =============================================================================
# Production Stage - Nginx
# =============================================================================
FROM nginx:alpine AS production

# Update Alpine packages to fix known CVEs
RUN apk upgrade --no-cache

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
