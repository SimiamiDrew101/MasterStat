# Use Node.js 20 alpine (multi-platform compatible)
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application files
COPY . .

# Expose port
EXPOSE 5173

# Start development server with host flag for Docker
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
