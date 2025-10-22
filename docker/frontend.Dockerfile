# Frontend Dockerfile placeholder
FROM node:18
WORKDIR /app
COPY frontend/package.json ./
RUN npm install
COPY frontend/src ./src
CMD ["npm", "run", "dev"]
