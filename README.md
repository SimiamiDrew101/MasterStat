# MasterStat - Statistical Analysis Tool

A comprehensive statistical analysis toolkit covering experimental design, ANOVA, and advanced statistical methods. Built with FastAPI (Python) backend and React + Tailwind CSS frontend, fully containerized with Docker for easy deployment on macOS (M1/M2).

## Features

### Statistical Methods Covered

1. **Hypothesis Testing**
   - t-tests (one-sample, two-sample, paired)
   - F-tests for variance equality
   - Z-tests
   - One-sided and two-sided tests
   - Confidence intervals

2. **ANOVA**
   - Single-factor ANOVA
   - Two-way ANOVA with interactions
   - Post-hoc tests (Tukey HSD)

3. **Factorial Designs**
   - Full factorial designs (2^k, 3^k)
   - Main effects and interactions
   - Fractional factorial designs
   - Alias structure and resolution

4. **Block Designs**
   - Randomized Complete Block Design (RCBD)
   - Latin Square designs
   - Blocking efficiency analysis

5. **Mixed Models**
   - Split-plot designs
   - Nested designs
   - Variance component estimation
   - Expected Mean Squares (EMS)

6. **Response Surface Methodology**
   - Central Composite Designs (CCD)
   - Second-order model fitting
   - Steepest ascent/descent
   - Lack-of-fit testing
   - Curvature detection

## Tech Stack

### Backend
- **FastAPI**: Modern Python web framework
- **scipy**: Statistical tests and distributions
- **statsmodels**: ANOVA and regression models
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **pyDOE2**: Design of Experiments

### Frontend
- **React 18**: UI framework
- **Vite**: Build tool
- **Tailwind CSS**: Styling
- **Recharts**: Data visualization
- **Axios**: API calls

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration

## Getting Started

### Prerequisites

- Docker Desktop for Mac (with Apple Silicon support)
- macOS with M1/M2/M3 processor

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd /Users/nj/Desktop/MasterStat2
   ```

2. **Build and start the containers**
   ```bash
   docker-compose up --build
   ```

   This will:
   - Build the Python backend with all statistical libraries
   - Build the React frontend with Tailwind CSS
   - Start both services
   - Make the app available at http://localhost:5173

3. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Development Mode

The containers are configured for hot-reloading:
- Backend changes in `backend/app/` are automatically reloaded
- Frontend changes in `frontend/src/` trigger instant updates

### Stopping the Application

```bash
docker-compose down
```

## Project Structure

```
MasterStat2/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── api/
│   │   │   ├── hypothesis_testing.py
│   │   │   ├── anova.py
│   │   │   ├── factorial.py
│   │   │   ├── block_designs.py
│   │   │   ├── mixed_models.py
│   │   │   └── rsm.py
│   │   ├── models/              # Data models
│   │   └── utils/               # Utility functions
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main application
│   │   ├── components/
│   │   │   └── ResultCard.jsx   # Results display
│   │   └── pages/
│   │       ├── Home.jsx
│   │       ├── HypothesisTesting.jsx
│   │       ├── ANOVA.jsx
│   │       ├── FactorialDesigns.jsx
│   │       ├── BlockDesigns.jsx
│   │       ├── MixedModels.jsx
│   │       └── RSM.jsx
│   ├── package.json
│   ├── tailwind.config.js
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Usage Examples

### Hypothesis Testing (t-test)

1. Navigate to "Hypothesis Testing" from the sidebar
2. Enter Sample 1 data: `12.5, 13.1, 11.8, 14.2, 12.9`
3. Enter Sample 2 data: `10.2, 11.5, 10.8, 11.9, 10.1`
4. Select test type and parameters
5. Click "Run Analysis"
6. View comprehensive results including:
   - Test statistic
   - p-value
   - Confidence intervals
   - Sample statistics

### ANOVA

1. Navigate to "ANOVA" from the sidebar
2. Add groups and enter data for each
3. Set significance level
4. Click "Run ANOVA"
5. View ANOVA table with F-statistics and p-values

## API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Example API Call

```bash
curl -X POST "http://localhost:8000/api/hypothesis/t-test" \
  -H "Content-Type: application/json" \
  -d '{
    "sample1": [12.5, 13.1, 11.8, 14.2, 12.9],
    "sample2": [10.2, 11.5, 10.8, 11.9, 10.1],
    "alternative": "two-sided",
    "alpha": 0.05,
    "paired": false
  }'
```

## Future Enhancements

- CSV/Excel file upload for data input
- Export results to PDF/Excel
- More visualization options
- Additional post-hoc tests
- Power analysis tools
- Sample size calculators

## Troubleshooting

### Port Already in Use

If ports 8000 or 5173 are already in use:

```bash
# Stop all containers
docker-compose down

# Check for processes using the ports
lsof -ti:8000
lsof -ti:5173

# Kill the processes if needed
kill -9 $(lsof -ti:8000)
kill -9 $(lsof -ti:5173)
```

### Container Build Issues

```bash
# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

## License

This project is for educational and research purposes.

## Support

For issues or questions, please check:
- API documentation: http://localhost:8000/docs
- subject_list.md for covered statistical topics
