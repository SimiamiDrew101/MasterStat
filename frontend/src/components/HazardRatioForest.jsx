import Plot from 'react-plotly.js'
import { getPlotlyConfig } from '../utils/plotlyConfig'

const HazardRatioForest = ({
  data,
  title = 'Hazard Ratio Forest Plot',
  height = 400
}) => {
  if (!data || !data.length) return null

  // Sort by hazard ratio for better visualization
  const sortedData = [...data].reverse()

  const traces = []

  // Error bars for confidence intervals
  traces.push({
    type: 'scatter',
    mode: 'markers',
    x: sortedData.map(d => d.hr),
    y: sortedData.map(d => d.covariate),
    error_x: {
      type: 'data',
      symmetric: false,
      array: sortedData.map(d => d.hr_upper - d.hr),
      arrayminus: sortedData.map(d => d.hr - d.hr_lower),
      color: '#94a3b8',
      thickness: 2,
      width: 6
    },
    marker: {
      size: 12,
      color: sortedData.map(d => d.p_value < 0.05 ? '#3b82f6' : '#64748b'),
      symbol: 'diamond'
    },
    name: 'Hazard Ratio',
    hovertemplate: '%{y}<br>HR: %{x:.3f}<br>CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>p: %{customdata[2]:.4f}<extra></extra>',
    customdata: sortedData.map(d => [d.hr_lower, d.hr_upper, d.p_value])
  })

  // Reference line at HR = 1
  traces.push({
    type: 'scatter',
    mode: 'lines',
    x: [1, 1],
    y: [-0.5, sortedData.length - 0.5],
    line: { color: '#ef4444', width: 2, dash: 'dash' },
    showlegend: false,
    hoverinfo: 'skip'
  })

  // Calculate x-axis range based on data
  const allValues = sortedData.flatMap(d => [d.hr_lower, d.hr_upper])
  const minX = Math.max(0.1, Math.min(...allValues) * 0.8)
  const maxX = Math.max(...allValues) * 1.2

  const layout = {
    title: {
      text: title,
      font: { color: '#e2e8f0', size: 16 }
    },
    paper_bgcolor: '#1e293b',
    plot_bgcolor: '#0f172a',
    font: { color: '#e2e8f0' },
    xaxis: {
      title: 'Hazard Ratio (log scale)',
      type: 'log',
      gridcolor: '#475569',
      zerolinecolor: '#475569',
      tickfont: { color: '#e2e8f0' },
      range: [Math.log10(minX), Math.log10(maxX)]
    },
    yaxis: {
      title: '',
      gridcolor: '#475569',
      tickfont: { color: '#e2e8f0' },
      automargin: true
    },
    margin: { t: 50, r: 30, b: 60, l: 150 },
    height: Math.max(height, 80 + sortedData.length * 40),
    annotations: [
      {
        x: 0.1,
        y: -0.12,
        xref: 'paper',
        yref: 'paper',
        text: '← Lower Risk',
        showarrow: false,
        font: { color: '#22c55e', size: 11 }
      },
      {
        x: 0.9,
        y: -0.12,
        xref: 'paper',
        yref: 'paper',
        text: 'Higher Risk →',
        showarrow: false,
        font: { color: '#ef4444', size: 11 }
      }
    ],
    shapes: [
      // Shaded region for HR < 1 (protective)
      {
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: minX,
        x1: 1,
        y0: 0,
        y1: 1,
        fillcolor: 'rgba(34, 197, 94, 0.05)',
        line: { width: 0 }
      },
      // Shaded region for HR > 1 (risk)
      {
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: 1,
        x1: maxX,
        y0: 0,
        y1: 1,
        fillcolor: 'rgba(239, 68, 68, 0.05)',
        line: { width: 0 }
      }
    ]
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

export default HazardRatioForest
