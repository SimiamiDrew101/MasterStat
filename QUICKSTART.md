# MasterStat Quick Start Guide

Get MasterStat up and running in under 5 minutes!

## 🖥️ Option 1: Electron Desktop App (Easiest!)

Run MasterStat as a native desktop application - one command to launch everything!

### Prerequisites
- Node.js v16+ ([Download](https://nodejs.org/))
- Python 3.11+ ([Download](https://python.org/downloads/))

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SimiamiDrew101/MasterStat.git
   cd MasterStat
   ```

2. **Install dependencies:**
   ```bash
   npm install
   cd backend && pip install -r requirements.txt && cd ..
   ```

3. **Launch the app:**
   ```bash
   npm run electron
   ```

4. **That's it!** The app opens in a native window with the backend automatically started.

**What you get:**
- 🚀 One-click launch - everything starts automatically
- 💻 Native desktop window - no browser needed
- 🔒 Fully offline - all processing happens locally
- ⚡ Fast startup - optimized for desktop

**To close:** Just close the window or press `Ctrl+C` in the terminal.

---

## 🐳 Option 2: Docker

Run MasterStat in containers with both frontend and backend.

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

## 💻 Option 3: Local Development (Full Stack)

Run backend and frontend separately for development work.

### Prerequisites
- Node.js v16+ ([Download](https://nodejs.org/))
- Python 3.11+ ([Download](https://python.org/downloads/))

### Steps

1. **Start the backend:**
   ```bash
   cd /path/to/MasterStat/backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

2. **In a new terminal, start the frontend:**
   ```bash
   cd /path/to/MasterStat/frontend
   npm install
   npm run dev
   ```

3. **Open your browser:**
   - Frontend: **http://localhost:5173**
   - Backend API: **http://localhost:8000**
   - API Docs: **http://localhost:8000/docs**

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

### Electron App Won't Start

If the Electron app fails to launch:

```bash
# Close any existing instances
pkill -f electron
pkill -f uvicorn

# Check Python is installed
python3 --version  # Should be 3.11+

# Reinstall dependencies
npm install
cd backend && pip install -r requirements.txt && cd ..

# Try again
npm run electron
```

### Port Already in Use

If ports 5173 or 8000 are already in use:

```bash
# Check what's using the ports
lsof -ti:5173
lsof -ti:8000

# Close Electron or other instances
pkill -f electron
pkill -f uvicorn

# Stop Docker containers (if using Docker)
docker-compose down

# Try again
npm run electron  # or ./start.sh for Docker
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

- **Try the Electron app** - Easiest way to get started!
- Read the full [README.md](README.md) for detailed documentation
- Explore all 14 statistical modules
- Check API documentation at http://localhost:8000/docs
- Support the project at [Ko-fi](https://ko-fi.com/MasterStat)

---

## 💡 Tips

- **Electron is easiest:** One command (`npm run electron`) launches everything
- **Docker for isolation:** Best for production or avoiding local Python setup
- **Local dev for coding:** Hot-reload enabled for both frontend and backend
- **API Documentation:** Interactive Swagger UI at http://localhost:8000/docs
- **Cross-platform:** Electron app works on macOS, Windows, and Linux
- **Fully offline:** All analysis runs locally, no internet required

---

**Need help?** Open an issue on [GitHub](https://github.com/SimiamiDrew101/MasterStat/issues)
