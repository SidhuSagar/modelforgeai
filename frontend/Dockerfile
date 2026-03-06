# Frontend Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend code
COPY . .

# Expose port
EXPOSE 5173

# Run dev server (for production, use: npm run build && npm run preview)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

