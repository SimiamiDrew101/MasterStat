# Quick Start Guide

## Step 1: Start Docker Desktop

1. Open **Docker Desktop** application on your Mac
2. Wait for Docker to fully start (the whale icon in the menu bar should be steady)

## Step 2: Launch MasterStat

Open Terminal and run:

```bash
cd /Users/nj/Desktop/MasterStat2
./start.sh
```

Or use Docker Compose directly:

```bash
cd /Users/nj/Desktop/MasterStat2
docker-compose up --build
```

## Step 3: Open in Browser

Once you see "Application startup complete", open Safari and navigate to:

**http://localhost:5173**

## What You'll See

- Beautiful gradient interface with statistical analysis tools
- Sidebar navigation with 6 major statistical method categories
- Interactive forms for data entry
- Real-time analysis results with visualizations

## Try Your First Analysis

### Example: Two-Sample t-Test

1. Click **"Hypothesis Testing"** in the sidebar
2. Select **"t-Test"** from the dropdown
3. Enter Sample 1: `12.5, 13.1, 11.8, 14.2, 12.9, 13.5`
4. Enter Sample 2: `10.2, 11.5, 10.8, 11.9, 10.1, 11.2`
5. Keep default settings (α = 0.05, two-sided)
6. Click **"Run Analysis"**
7. View comprehensive results including:
   - t-statistic
   - p-value
   - Confidence intervals
   - Sample statistics
   - Decision (Reject H₀ or not)

### Example: One-Way ANOVA

1. Click **"ANOVA"** in the sidebar
2. Enter data for Group A: `23, 25, 22, 24, 26`
3. Enter data for Group B: `30, 32, 29, 31, 33`
4. Enter data for Group C: `18, 20, 19, 21, 17`
5. Click **"Run ANOVA"**
6. View ANOVA table with F-statistic and p-values

## Stopping the Application

Press `Ctrl+C` in the Terminal window, or run:

```bash
docker-compose down
```

## Troubleshooting

### Issue: Port already in use

```bash
# Stop all containers
docker-compose down

# Try starting again
./start.sh
```

### Issue: Container won't build

```bash
# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### Issue: Can't access on Safari

- Make sure both containers are running (check Terminal output)
- Try http://localhost:5173 instead of https://
- Clear Safari cache if needed
- Check if port 5173 is accessible: `lsof -i:5173`

## API Documentation

You can also access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Next Steps

- Explore all 6 statistical method categories
- Try different test types and parameters
- Check the ANOVA tables and interpretations
- View the raw JSON results for deeper insights

## Need Help?

- Check README.md for full documentation
- View subject_list.md to see all covered topics
- API docs available at http://localhost:8000/docs
