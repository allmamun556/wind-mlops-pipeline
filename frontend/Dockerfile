# Multi-stage build: compile the Vite/React app, then serve the static
# bundle with Nginx (also reverse-proxies /api to the FastAPI backend so
# the frontend can be built without baking a backend hostname into the JS).
FROM node:20-slim AS build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
# Same-origin /api path — see nginx.conf — so no backend URL needs to be
# known at build time.
ENV VITE_API_URL=/api
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
HEALTHCHECK CMD wget -qO- http://127.0.0.1:80/ || exit 1
