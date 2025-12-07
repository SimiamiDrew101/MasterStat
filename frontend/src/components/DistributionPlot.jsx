import Plot from 'react-plotly.js'
import { Activity } from 'lucide-react'
import { getPlotlyConfig } from '../utils/plotlyConfig'

const DistributionPlot = ({ distributionData, pValue, testType = "t-test", alpha = 0.05 }) => {
  if (!distributionData) return null

  const { x, y, test_statistic, critical_lower, critical_upper, df } = distributionData

  // Create traces for the plot
  const traces = []

  // Main distribution curve
  traces.push({
    type: 'scatter',
    mode: 'lines',
    x: x,
    y: y,
    fill: 'tozeroy',
    fillcolor: 'rgba(59, 130, 246, 0.1)',
    line: {
      color: '#3b82f6',
      width: 2
    },
    name: `${testType === "t-test" ? "t" : "z"}-distribution${df ? ` (df=${df})` : ""}`,
    hovertemplate: '%{x:.2f}, %{y:.4f}<extra></extra>'
  })

  // Critical region shading (left tail)
  if (critical_lower !== null && critical_lower !== undefined) {
    const criticalLeftX = x.filter(val => val <= critical_lower)
    const criticalLeftY = criticalLeftX.map(val => {
      const idx = x.findIndex(v => v >= val)
      return y[idx] || 0
    })

    traces.push({
      type: 'scatter',
      mode: 'lines',
      x: criticalLeftX,
      y: criticalLeftY,
      fill: 'tozeroy',
      fillcolor: 'rgba(239, 68, 68, 0.3)',
      line: {
        color: 'rgba(239, 68, 68, 0)',
        width: 0
      },
      name: `Critical region (α${critical_upper !== null ? '/2' : ''})`,
      showlegend: true,
      hoverinfo: 'skip'
    })
  }

  // Critical region shading (right tail)
  if (critical_upper !== null && critical_upper !== undefined) {
    const criticalRightX = x.filter(val => val >= critical_upper)
    const criticalRightY = criticalRightX.map(val => {
      const idx = x.findIndex(v => v >= val)
      return y[idx] || 0
    })

    traces.push({
      type: 'scatter',
      mode: 'lines',
      x: criticalRightX,
      y: criticalRightY,
      fill: 'tozeroy',
      fillcolor: 'rgba(239, 68, 68, 0.3)',
      line: {
        color: 'rgba(239, 68, 68, 0)',
        width: 0
      },
      name: critical_lower !== null ? '' : `Critical region (α)`,
      showlegend: critical_lower === null,
      hoverinfo: 'skip'
    })
  }

  // P-value shading (area corresponding to p-value)
  if (test_statistic !== null && test_statistic !== undefined) {
    // For two-sided test, shade both tails
    if (critical_lower !== null && critical_upper !== null) {
      // Left tail p-value area
      const pLeftX = x.filter(val => val <= -Math.abs(test_statistic))
      const pLeftY = pLeftX.map(val => {
        const idx = x.findIndex(v => v >= val)
        return y[idx] || 0
      })

      if (pLeftX.length > 0) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: pLeftX,
          y: pLeftY,
          fill: 'tozeroy',
          fillcolor: 'rgba(234, 179, 8, 0.4)',
          line: {
            color: 'rgba(234, 179, 8, 0)',
            width: 0
          },
          name: `p-value area (${pValue.toFixed(4)})`,
          showlegend: true,
          hoverinfo: 'skip'
        })
      }

      // Right tail p-value area
      const pRightX = x.filter(val => val >= Math.abs(test_statistic))
      const pRightY = pRightX.map(val => {
        const idx = x.findIndex(v => v >= val)
        return y[idx] || 0
      })

      if (pRightX.length > 0) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: pRightX,
          y: pRightY,
          fill: 'tozeroy',
          fillcolor: 'rgba(234, 179, 8, 0.4)',
          line: {
            color: 'rgba(234, 179, 8, 0)',
            width: 0
          },
          name: '',
          showlegend: false,
          hoverinfo: 'skip'
        })
      }
    } else if (critical_upper !== null) {
      // Right-tailed test
      const pRightX = x.filter(val => val >= test_statistic)
      const pRightY = pRightX.map(val => {
        const idx = x.findIndex(v => v >= val)
        return y[idx] || 0
      })

      if (pRightX.length > 0) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: pRightX,
          y: pRightY,
          fill: 'tozeroy',
          fillcolor: 'rgba(234, 179, 8, 0.4)',
          line: {
            color: 'rgba(234, 179, 8, 0)',
            width: 0
          },
          name: `p-value area (${pValue.toFixed(4)})`,
          showlegend: true,
          hoverinfo: 'skip'
        })
      }
    } else if (critical_lower !== null) {
      // Left-tailed test
      const pLeftX = x.filter(val => val <= test_statistic)
      const pLeftY = pLeftX.map(val => {
        const idx = x.findIndex(v => v >= val)
        return y[idx] || 0
      })

      if (pLeftX.length > 0) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: pLeftX,
          y: pLeftY,
          fill: 'tozeroy',
          fillcolor: 'rgba(234, 179, 8, 0.4)',
          line: {
            color: 'rgba(234, 179, 8, 0)',
            width: 0
          },
          name: `p-value area (${pValue.toFixed(4)})`,
          showlegend: true,
          hoverinfo: 'skip'
        })
      }
    }
  }

  // Test statistic line
  if (test_statistic !== null && test_statistic !== undefined) {
    const maxY = Math.max(...y)
    traces.push({
      type: 'scatter',
      mode: 'lines',
      x: [test_statistic, test_statistic],
      y: [0, maxY],
      line: {
        color: pValue < alpha ? '#ef4444' : '#22c55e',
        width: 3,
        dash: 'dash'
      },
      name: `Test statistic (${test_statistic.toFixed(3)})`,
      hovertemplate: `Test statistic: ${test_statistic.toFixed(4)}<extra></extra>`
    })
  }

  // Critical value lines
  if (critical_lower !== null && critical_lower !== undefined) {
    const maxY = Math.max(...y)
    traces.push({
      type: 'scatter',
      mode: 'lines',
      x: [critical_lower, critical_lower],
      y: [0, maxY],
      line: {
        color: '#ef4444',
        width: 2,
        dash: 'dot'
      },
      name: `Critical value (${critical_lower.toFixed(3)})`,
      hovertemplate: `Lower critical: ${critical_lower.toFixed(4)}<extra></extra>`
    })
  }

  if (critical_upper !== null && critical_upper !== undefined) {
    const maxY = Math.max(...y)
    traces.push({
      type: 'scatter',
      mode: 'lines',
      x: [critical_upper, critical_upper],
      y: [0, maxY],
      line: {
        color: '#ef4444',
        width: 2,
        dash: 'dot'
      },
      name: critical_lower !== null ? '' : `Critical value (${critical_upper.toFixed(3)})`,
      showlegend: critical_lower === null,
      hovertemplate: `Upper critical: ${critical_upper.toFixed(4)}<extra></extra>`
    })
  }

  const layout = {
    title: {
      text: `${testType === "t-test" ? "t" : "z"}-Distribution with Test Statistic and Critical Regions`,
      font: {
        size: 16,
        color: '#f1f5f9'
      }
    },
    xaxis: {
      title: `${testType === "t-test" ? "t" : "z"}-value`,
      gridcolor: '#475569',
      color: '#e2e8f0',
      zeroline: true,
      zerolinecolor: '#64748b',
      zerolinewidth: 1
    },
    yaxis: {
      title: 'Probability Density',
      gridcolor: '#475569',
      color: '#e2e8f0'
    },
    paper_bgcolor: '#334155',
    plot_bgcolor: '#1e293b',
    font: {
      color: '#e2e8f0'
    },
    margin: { l: 60, r: 40, b: 60, t: 80 },
    height: 450,
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(30, 41, 59, 0.8)',
      bordercolor: '#475569',
      borderwidth: 1
    }
  }

  const config = getPlotlyConfig('distribution-plot', {
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
  })

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="w-5 h-5 text-blue-400" />
        <h3 className="text-xl font-bold text-gray-100">Distribution Visualization</h3>
      </div>

      <Plot
        data={traces}
        layout={layout}
        config={config}
        style={{ width: '100%' }}
      />

      <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
        <div className="bg-blue-900/20 rounded-lg p-3 border border-blue-700/30">
          <p className="text-blue-200 font-semibold mb-1">Blue Curve</p>
          <p className="text-gray-300">The {testType === "t-test" ? "t" : "z"}-distribution under the null hypothesis</p>
        </div>
        <div className="bg-red-900/20 rounded-lg p-3 border border-red-700/30">
          <p className="text-red-200 font-semibold mb-1">Red Shaded Area</p>
          <p className="text-gray-300">Critical regions (α = {alpha}) where we reject H₀</p>
        </div>
        <div className="bg-yellow-900/20 rounded-lg p-3 border border-yellow-700/30">
          <p className="text-yellow-200 font-semibold mb-1">Yellow Shaded Area</p>
          <p className="text-gray-300">p-value area representing the probability of observing this result</p>
        </div>
      </div>
    </div>
  )
}

export default DistributionPlot
