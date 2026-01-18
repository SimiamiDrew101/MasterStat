import Plot from 'react-plotly.js'
import { getPlotlyConfig } from '../utils/plotlyConfig'

/**
 * PowerCurvePlot component for visualizing statistical power vs sample size
 * Shows power curves for different effect sizes with reference lines
 * Used in experiment planning and design recommendation
 */
const PowerCurvePlot = ({
  currentN = null,
  currentPower = null,
  effectSize = 0.5,
  alpha = 0.05,
  minN = 5,
  maxN = 100
}) => {
  // Effect sizes to display (small, medium, large)
  const effectSizes = [
    { value: 0.2, label: 'Small (d=0.2)', color: '#3b82f6' },
    { value: 0.5, label: 'Medium (d=0.5)', color: '#10b981' },
    { value: 0.8, label: 'Large (d=0.8)', color: '#f59e0b' }
  ]

  // Calculate power using approximate formula for t-test
  // Power = 1 - β = Φ(|δ|√(n/2) - z_{1-α/2})
  // where δ is Cohen's d, Φ is standard normal CDF
  const calculatePower = (n, d, alpha) => {
    // Simplified power calculation for two-sample t-test
    // Using non-central t-distribution approximation
    const ncp = d * Math.sqrt(n / 2) // Non-centrality parameter
    const tCrit = 1.96 // Approximate critical value for α = 0.05 (two-tailed)

    // Standard normal CDF approximation
    const phi = (x) => {
      const t = 1 / (1 + 0.2316419 * Math.abs(x))
      const d = 0.3989423 * Math.exp(-x * x / 2)
      const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
      return x > 0 ? 1 - p : p
    }

    const power = phi(ncp - tCrit) + phi(-ncp - tCrit)
    return Math.max(0, Math.min(1, power)) // Clamp between 0 and 1
  }

  // Generate sample sizes
  const sampleSizes = []
  for (let n = minN; n <= maxN; n += Math.max(1, Math.floor((maxN - minN) / 50))) {
    sampleSizes.push(n)
  }

  // Generate power curves for each effect size
  const traces = effectSizes.map(({ value, label, color }) => {
    const powers = sampleSizes.map(n => calculatePower(n, value, alpha))

    return {
      type: 'scatter',
      mode: 'lines',
      x: sampleSizes,
      y: powers,
      name: label,
      line: {
        color: color,
        width: 3
      },
      hovertemplate: `<b>${label}</b><br>` +
        'Sample Size: %{x}<br>' +
        'Power: %{y:.3f}<br>' +
        '<extra></extra>'
    }
  })

  // Add reference lines
  const referenceLines = [
    {
      type: 'scatter',
      mode: 'lines',
      x: [minN, maxN],
      y: [0.80, 0.80],
      name: 'Target Power (0.80)',
      line: {
        color: '#ef4444',
        width: 2,
        dash: 'dash'
      },
      hoverinfo: 'skip',
      showlegend: true
    },
    {
      type: 'scatter',
      mode: 'lines',
      x: [minN, maxN],
      y: [0.90, 0.90],
      name: 'High Power (0.90)',
      line: {
        color: '#ef4444',
        width: 2,
        dash: 'dot'
      },
      hoverinfo: 'skip',
      showlegend: true
    }
  ]

  // Add current design marker if provided
  if (currentN !== null && currentPower !== null) {
    traces.push({
      type: 'scatter',
      mode: 'markers',
      x: [currentN],
      y: [currentPower],
      name: 'Current Design',
      marker: {
        size: 16,
        color: '#8b5cf6',
        symbol: 'diamond',
        line: {
          color: '#f1f5f9',
          width: 2
        }
      },
      hovertemplate: '<b>Current Design</b><br>' +
        `Sample Size: ${currentN}<br>` +
        `Power: ${currentPower.toFixed(3)}<br>` +
        '<extra></extra>'
    })
  }

  const layout = {
    title: {
      text: `Statistical Power vs Sample Size (α = ${alpha})`,
      font: {
        size: 18,
        color: '#f1f5f9'
      }
    },
    xaxis: {
      title: 'Sample Size (per group)',
      gridcolor: '#475569',
      zerolinecolor: '#64748b',
      color: '#e2e8f0',
      range: [minN, maxN]
    },
    yaxis: {
      title: 'Statistical Power (1 - β)',
      gridcolor: '#475569',
      zerolinecolor: '#64748b',
      color: '#e2e8f0',
      range: [0, 1],
      tickformat: '.0%'
    },
    paper_bgcolor: '#334155',
    plot_bgcolor: '#1e293b',
    font: {
      color: '#e2e8f0'
    },
    margin: {
      l: 70,
      r: 60,
      b: 70,
      t: 80
    },
    showlegend: true,
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(51, 65, 85, 0.9)',
      bordercolor: '#475569',
      borderwidth: 1,
      font: {
        size: 12
      }
    },
    hovermode: 'closest'
  }

  const config = getPlotlyConfig('power-curve-plot', {
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
  })

  const plotData = [...traces, ...referenceLines]

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="mb-4">
        <h4 className="text-gray-100 font-semibold text-lg">Power Analysis</h4>
        <p className="text-gray-400 text-sm mt-1">
          Statistical power to detect effects of different sizes
        </p>
      </div>

      <div className="flex justify-center bg-slate-800/50 rounded-lg p-4">
        <Plot
          data={plotData}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '500px' }}
          useResizeHandler={true}
        />
      </div>

      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <div className="text-gray-300 text-sm space-y-2">
          <p>
            <strong className="text-gray-100">Interpretation:</strong> Power curves show the probability
            of detecting a true effect (if it exists) for different sample sizes and effect magnitudes.
          </p>

          <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="bg-slate-700/50 rounded p-3">
              <h5 className="text-gray-100 font-medium mb-2">Cohen's d Effect Sizes:</h5>
              <ul className="space-y-1 text-xs">
                <li className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                  <span>Small (d = 0.2): Subtle difference</span>
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                  <span>Medium (d = 0.5): Moderate difference</span>
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-amber-500"></div>
                  <span>Large (d = 0.8): Substantial difference</span>
                </li>
              </ul>
            </div>

            <div className="bg-slate-700/50 rounded p-3">
              <h5 className="text-gray-100 font-medium mb-2">Power Guidelines:</h5>
              <ul className="space-y-1 text-xs">
                <li className="flex items-center gap-2">
                  <div className="w-3 h-1 bg-red-500"></div>
                  <span>0.80 (Conventional minimum)</span>
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-3 h-1 bg-red-500" style={{ borderBottom: '1px dotted' }}></div>
                  <span>0.90 (Preferred for critical studies)</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-gray-400">Higher power = Lower risk of Type II error (β)</span>
                </li>
              </ul>
            </div>
          </div>

          {currentN && currentPower && (
            <div className="mt-3 bg-purple-900/20 border border-purple-700/50 rounded p-3">
              <p className="text-purple-200">
                <strong>Current Design:</strong> With {currentN} samples per group and
                effect size d = {effectSize}, your design has approximately{' '}
                <strong className="text-purple-100">{(currentPower * 100).toFixed(1)}% power</strong>
                {currentPower >= 0.80
                  ? ' (sufficient for most studies)'
                  : ' (consider increasing sample size for higher power)'}
              </p>
            </div>
          )}

          <div className="mt-3 text-xs text-gray-400">
            <strong>Note:</strong> Power calculations assume a two-sample t-test with equal variances.
            Actual power may vary with the specific analysis method and design complexity.
          </div>
        </div>
      </div>
    </div>
  )
}

export default PowerCurvePlot
