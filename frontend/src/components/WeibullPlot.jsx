import Plot from 'react-plotly.js'
import { getPlotlyConfig } from '../utils/plotlyConfig'

const WeibullPlot = ({
  data,
  title = 'Weibull Probability Plot',
  height = 400
}) => {
  if (!data || !data.probability_plot) return null

  const { probability_plot, parameters } = data
  const { x, y, times, probabilities } = probability_plot

  const traces = []

  // Data points
  traces.push({
    type: 'scatter',
    mode: 'markers',
    x: x,
    y: y,
    name: 'Observed Failures',
    marker: {
      size: 10,
      color: '#3b82f6',
      symbol: 'circle'
    },
    hovertemplate: 'ln(Time): %{x:.3f}<br>ln(-ln(1-F)): %{y:.3f}<br>Time: %{customdata[0]:.2f}<br>F(t): %{customdata[1]:.3f}<extra></extra>',
    customdata: times.map((t, i) => [t, probabilities[i]])
  })

  // Fitted line based on Weibull parameters
  if (parameters && parameters.shape && parameters.scale) {
    const beta = parameters.shape  // shape
    const eta = parameters.scale   // scale (lambda)

    // Weibull line: y = beta * x - beta * ln(eta)
    const xMin = Math.min(...x)
    const xMax = Math.max(...x)
    const xRange = [xMin - 0.5, xMax + 0.5]
    const yFitted = xRange.map(xi => beta * xi - beta * Math.log(eta))

    traces.push({
      type: 'scatter',
      mode: 'lines',
      x: xRange,
      y: yFitted,
      name: `Fitted (β=${beta.toFixed(2)}, η=${eta.toFixed(2)})`,
      line: { color: '#ef4444', width: 2 }
    })
  }

  // Y-axis labels for probability values
  const probLabels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
  const yTickVals = probLabels.map(p => Math.log(-Math.log(1 - p)))
  const yTickText = probLabels.map(p => `${(p * 100).toFixed(0)}%`)

  const layout = {
    title: {
      text: title,
      font: { color: '#e2e8f0', size: 16 }
    },
    paper_bgcolor: '#1e293b',
    plot_bgcolor: '#0f172a',
    font: { color: '#e2e8f0' },
    xaxis: {
      title: 'ln(Time)',
      gridcolor: '#475569',
      zerolinecolor: '#475569',
      tickfont: { color: '#e2e8f0' }
    },
    yaxis: {
      title: 'Cumulative Failure Probability',
      gridcolor: '#475569',
      zerolinecolor: '#475569',
      tickfont: { color: '#e2e8f0' },
      tickvals: yTickVals,
      ticktext: yTickText
    },
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(30, 41, 59, 0.8)',
      bordercolor: '#475569',
      font: { color: '#e2e8f0' }
    },
    margin: { t: 50, r: 30, b: 50, l: 70 },
    height: height,
    annotations: parameters ? [
      {
        x: 0.98,
        y: 0.02,
        xref: 'paper',
        yref: 'paper',
        text: `Shape (β): ${parameters.shape?.toFixed(3)}<br>Scale (η): ${parameters.scale?.toFixed(3)}<br>MTTF: ${parameters.mttf?.toFixed(2)}`,
        showarrow: false,
        font: { color: '#e2e8f0', size: 11 },
        bgcolor: 'rgba(30, 41, 59, 0.8)',
        bordercolor: '#475569',
        borderwidth: 1,
        borderpad: 6,
        align: 'right'
      }
    ] : []
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

export default WeibullPlot
