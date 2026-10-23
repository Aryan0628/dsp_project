# Web UI image. The API address is baked in at build time (NEXT_PUBLIC_* vars are inlined
# into the browser bundle), so pass it as a build argument:
#   docker build --build-arg NEXT_PUBLIC_API_URL=https://api.example.com -t braintumor-ui frontend/
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG NEXT_PUBLIC_API_URL=http://localhost:3000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production PORT=5713 HOSTNAME=0.0.0.0
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 5713
CMD ["node", "server.js"]
