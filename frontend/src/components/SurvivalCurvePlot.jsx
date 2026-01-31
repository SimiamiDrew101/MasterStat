import Plot from 'react-plotly.js'
import { getPlotlyConfig } from '../utils/plotlyConfig'

const SurvivalCurvePlot = ({
  data,
  title = 'Kaplan-Meier Survival Curve',
  showConfidenceIntervals = true,
  height = 450
}) => {
  if (!data) return null

  const traces = []

  // Handle single group (overall) or multiple groups
  if (data.overall) {
    // Single group analysis
    const curve = data.overall.survival_curve
    if (curve && curve.times && curve.survival) {
      // Main survival curve (step function)
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: curve.times,
        y: curve.survival,
        name: 'Survival',
        line: { shape: 'hv', color: '#3b82f6', width: 2 },
        hovertemplate: 'Time: %{x:.2f}<br>Survival: %{y:.3f}<extra></extra>'
      })

      // Confidence interval bands
      if (showConfidenceIntervals && curve.ci_lower && curve.ci_upper) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: [...curve.times, ...curve.times.slice().reverse()],
          y: [...curve.ci_upper, ...curve.ci_lower.slice().reverse()],
          fill: 'toself',
          fillcolor: 'rgba(59, 130, 246, 0.2)',
          line: { color: 'transparent' },
          name: 'Confidence Interval',
          showlegend: true,
          hoverinfo: 'skip'
        })
      }

      // Add median survival time marker if available
      if (data.overall.median_survival_time) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: [0, data.overall.median_survival_time, data.overall.median_survival_time],
          y: [0.5, 0.5, 0],
          line: { color: '#ef4444', width: 1, dash: 'dash' },
          name: `Median: ${data.overall.median_survival_time.toFixed(2)}`,
          hovertemplate: 'Median Survival Time: %{x:.2f}<extra></extra>'
        })
      }
    }
  } else if (data.groups) {
    // Multiple groups
    const colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899']
    const groupNames = Object.keys(data.groups)

    groupNames.forEach((groupName, idx) => {
      const group = data.groups[groupName]
      const curve = group.survival_curve
      const color = colors[idx % colors.length]

      if (curve && curve.times && curve.survival) {
        // Main survival curve
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: curve.times,
          y: curve.survival,
          name: `${groupName} (n=${group.n_observations})`,
          line: { shape: 'hv', color: color, width: 2 },
          hovertemplate: `${groupName}<br>Time: %{x:.2f}<br>Survival: %{y:.3f}<extra></extra>`
        })

        // Confidence interval bands
        if (showConfidenceIntervals && curve.ci_lower && curve.ci_upper) {
          traces.push({
            type: 'scatter',
            mode: 'lines',
            x: [...curve.times, ...curve.times.slice().reverse()],
            y: [...curve.ci_upper, ...curve.ci_lower.slice().reverse()],
            fill: 'toself',
            fillcolor: color.replace(')', ', 0.15)').replace('rgb', 'rgba'),
            line: { color: 'transparent' },
            showlegend: false,
            hoverinfo: 'skip'
          })
        }
      }
    })
  }

  const layout = {
    title: {
      text: title,
      font: { color: '#e2e8f0', size: 16 }
    },
    paper_bgcolor: '#1e293b',
    plot_bgcolor: '#0f172a',
    font: { color: '#e2e8f0' },
    xaxis: {
      title: 'Time',
      gridcolor: '#475569',
      zerolinecolor: '#475569',
      tickfont: { color: '#e2e8f0' }
    },
    yaxis: {
      title: 'Survival Probability',
      range: [0, 1.05],
      gridcolor: '#475569',
      zerolinecolor: '#475569',
      tickfont: { color: '#e2e8f0' }
    },
    legend: {
      x: 0.7,
      y: 0.95,
      bgcolor: 'rgba(30, 41, 59, 0.8)',
      bordercolor: '#475569',
      font: { color: '#e2e8f0' }
    },
    margin: { t: 50, r: 30, b: 50, l: 60 },
    height: height
  }

  return (
    <Plot
      data={traces}
      layout={layout}
      config={getPlotlyConfig()}
      style={{ width: '100%' }}
    />
  )
}

export default SurvivalCurvePlot
