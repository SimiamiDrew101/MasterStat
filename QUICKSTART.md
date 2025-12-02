# MasterStat Quick Start Guide

Get MasterStat up and running in under 5 minutes!

## 🚀 Option 1: Docker (Recommended)

The easiest way to run MasterStat with both frontend and backend.

### Prerequisites
- Docker Desktop ([Download](https://www.docker.com/products/docker-desktop))

### Steps

1. **Start Docker Desktop** and wait for it to fully launch

2. **Open Terminal** and navigate to the project:
   ```bash
   cd /path/to/MasterStat
   ```

3. **Launch MasterStat:**
   ```bash
   ./start.sh
   ```

   Or use Docker Compose directly:
   ```bash
   docker-compose up --build
   ```

4. **Open your browser:**
   - Frontend: **http://localhost:5173**
   - Backend API: **http://localhost:8000**
   - API Docs: **http://localhost:8000/docs**

5. **Stop the application:**
   - Press `Ctrl+C` in Terminal, or run:
   ```bash
   docker-compose down
   ```

---

## 💻 Option 2: Local Development (Frontend Only)

Run just the frontend without Docker. Note: Backend features won't be available.

### Prerequisites
- Node.js v16+ ([Download](https://nodejs.org/))

### Steps

1. **Navigate to frontend directory:**
   ```bash
   cd /path/to/MasterStat/frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser:**
   - Frontend: **http://localhost:5173**

---

## ✅ Try Your First Analysis

### Example 1: Two-Sample t-Test

1. Navigate to **Hypothesis Testing** from the home page
2. Select **t-Test** from the dropdown
3. Enter Sample 1: `12.5, 13.1, 11.8, 14.2, 12.9`
4. Enter Sample 2: `10.2, 11.5, 10.8, 11.9, 10.1`
5. Click **Run Analysis**
6. View results: t-statistic, p-value, confidence intervals

### Example 2: Factorial Design

1. Navigate to **Experiment Wizard**
2. Select goal: **Screening**
3. Enter 5 factors with names
4. Generate design matrix
5. Export to Excel, PDF, or CSV

### Example 3: Response Surface

1. Navigate to **Response Surface**
2. Choose **Central Composite Design**
3. Enter 3 factors
4. View 3D surface plots and contour plots
5. Perform optimization

---

## 🐛 Troubleshooting

### Port Already in Use

If ports 5173 or 8000 are already in use:

```bash
# Check what's using the ports
lsof -ti:5173
lsof -ti:8000

# Stop Docker containers
docker-compose down

# Try again
./start.sh
```

### Container Build Fails

```bash
# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### Can't Access in Browser

- Ensure Docker containers are running (check Terminal output)
- Try `http://localhost:5173` (not `https://`)
- Clear browser cache
- Check Docker Desktop - both containers should show as running

---

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore all 12 statistical modules
- Check API documentation at http://localhost:8000/docs
- Support the project at [Ko-fi](https://ko-fi.com/MasterStat)

---

## 💡 Tips

- **Hot-reload enabled:** Changes to code automatically refresh in Docker
- **API Documentation:** Interactive Swagger UI at `/docs` endpoint
- **Multiple methods:** npm, Docker, or Docker with start script
- **Backend optional:** Frontend works standalone, but backend enables full analysis features

---

**Need help?** Open an issue on [GitHub](https://github.com/SimiamiDrew101/MasterStat/issues)
