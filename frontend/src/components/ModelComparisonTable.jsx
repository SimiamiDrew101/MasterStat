import { BarChart3, TrendingDown, Award, Info } from 'lucide-react'
import { useState } from 'react'

const ModelComparisonTable = ({ modelFit, modelName = "Current Model" }) => {
  const [showDetails, setShowDetails] = useState(false)

  if (!modelFit || modelFit.error) {
    return null
  }

  const { aic, bic, caic, adj_bic, log_likelihood, n_parameters, n_observations, aic_per_obs, bic_per_obs } = modelFit

  // Determine model quality based on per-observation metrics
  const getQuality = () => {
    if (!aic_per_obs || !bic_per_obs) return 'unknown'
    const avgMetric = (aic_per_obs + bic_per_obs) / 2
    if (avgMetric < 2) return 'excellent'
    if (avgMetric < 5) return 'good'
    if (avgMetric < 10) return 'moderate'
    return 'poor'
  }

  const quality = getQuality()
  const qualityColors = {
    excellent: { bg: 'bg-green-500/10', border: 'border-green-500/50', text: 'text-green-400' },
    good: { bg: 'bg-blue-500/10', border: 'border-blue-500/50', text: 'text-blue-400' },
    moderate: { bg: 'bg-yellow-500/10', border: 'border-yellow-500/50', text: 'text-yellow-400' },
    poor: { bg: 'bg-red-500/10', border: 'border-red-500/50', text: 'text-red-400' },
    unknown: { bg: 'bg-slate-500/10', border: 'border-slate-500/50', text: 'text-slate-400' }
  }

  const colors = qualityColors[quality]

  // Visual bar for metrics (relative scale)
  const MetricBar = ({ value, label, color = 'bg-blue-500', showValue = true }) => {
    // Normalize to percentage for visual display (arbitrary scale for aesthetics)
    const maxDisplay = Math.max(aic || 0, bic || 0, caic || 0, adj_bic || 0)
    const percentage = maxDisplay > 0 ? (value / maxDisplay) * 100 : 0

    return (
      <div className="flex items-center space-x-3">
        <div className="w-24 text-xs text-gray-400 text-right">{label}</div>
        <div className="flex-1 flex items-center space-x-2">
          <div className="flex-1 bg-slate-700 rounded-full h-3 overflow-hidden">
            <div
              className={`${color} h-full transition-all duration-500 ease-out`}
              style={{ width: `${Math.min(100, percentage)}%` }}
            />
          </div>
          {showValue && (
            <div className="w-20 text-sm font-mono text-gray-200 text-right">
              {value.toFixed(2)}
            </div>
          )}
        </div>
      </div>
    )
  }

  // Radar chart for model fit metrics (SVG-based)
  const RadarChart = () => {
    if (!aic || !bic || !caic || !adj_bic) return null

    // Normalize metrics to 0-100 scale for visualization
    const maxValue = Math.max(aic, bic, caic, adj_bic)
    const normalize = (val) => ((maxValue - val) / maxValue) * 80 + 10 // Invert (lower is better)

    const metrics = [
      { label: 'AIC', value: normalize(aic), angle: 0 },
      { label: 'BIC', value: normalize(bic), angle: 90 },
      { label: 'CAIC', value: normalize(caic), angle: 180 },
      { label: 'Adj BIC', value: normalize(adj_bic), angle: 270 }
    ]

    const getPoint = (value, angle) => {
      const rad = (angle * Math.PI) / 180
      const x = 100 + value * Math.cos(rad)
      const y = 100 + value * Math.sin(rad)
      return { x, y }
    }

    const points = metrics.map(m => getPoint(m.value, m.angle))
    const pathData = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z'

    return (
      <div className="flex justify-center my-6">
        <svg width="240" height="240" viewBox="0 0 200 200" className="drop-shadow-lg">
          {/* Background circles */}
          {[20, 40, 60, 80].map(r => (
            <circle
              key={r}
              cx="100"
              cy="100"
              r={r}
              fill="none"
              stroke="currentColor"
              strokeWidth="0.5"
              className="text-slate-600"
            />
          ))}

          {/* Axes */}
          {metrics.map((m, i) => {
            const endpoint = getPoint(90, m.angle)
            return (
              <line
                key={i}
                x1="100"
                y1="100"
                x2={endpoint.x}
                y2={endpoint.y}
                stroke="currentColor"
                strokeWidth="0.5"
                className="text-slate-600"
              />
            )
          })}

          {/* Data polygon */}
          <path
            d={pathData}
            fill="currentColor"
            fillOpacity="0.3"
            stroke="currentColor"
            strokeWidth="2"
            className="text-purple-400"
          />

          {/* Data points */}
          {points.map((p, i) => (
            <circle
              key={i}
              cx={p.x}
              cy={p.y}
              r="3"
              fill="currentColor"
              className="text-purple-400"
            />
          ))}

          {/* Labels */}
          {metrics.map((m, i) => {
            const labelPoint = getPoint(105, m.angle)
            return (
              <text
                key={i}
                x={labelPoint.x}
                y={labelPoint.y}
                fill="currentColor"
                fontSize="10"
                textAnchor="middle"
                className="text-gray-300 font-semibold"
              >
                {m.label}
              </text>
            )
          })}
        </svg>
      </div>
    )
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <BarChart3 className="w-6 h-6 text-indigo-400" />
          <h3 className="text-xl font-bold text-gray-100">Model Fit Metrics</h3>
        </div>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-gray-400 hover:text-gray-200 transition-colors"
        >
          <Info className="w-5 h-5" />
        </button>
      </div>

      <p className="text-gray-400 text-sm mb-6">
        Information criteria for model comparison. Lower values indicate better fit, balancing goodness-of-fit with model complexity.
      </p>

      {/* Overall quality indicator */}
      {quality !== 'unknown' && (
        <div className={`rounded-lg border ${colors.border} ${colors.bg} p-4 mb-6`}>
          <div className="flex items-center space-x-2">
            <Award className={`w-5 h-5 ${colors.text}`} />
            <h4 className={`font-semibold ${colors.text} capitalize`}>
              {quality} Model Fit
            </h4>
          </div>
          <p className="text-sm text-gray-300 mt-2">
            Based on per-observation information criteria (AIC/obs: {aic_per_obs?.toFixed(3)}, BIC/obs: {bic_per_obs?.toFixed(3)})
          </p>
        </div>
      )}

      {/* Radar chart visualization */}
      {aic && bic && caic && adj_bic && (
        <>
          <h4 className="text-md font-semibold text-gray-200 mb-3 flex items-center">
            <TrendingDown className="w-4 h-4 mr-2 text-purple-400" />
            Information Criteria Comparison
          </h4>
          <RadarChart />
        </>
      )}

      {/* Bar chart for metrics */}
      <div className="space-y-3 mt-6">
        {aic && <MetricBar value={aic} label="AIC" color="bg-blue-500" />}
        {bic && <MetricBar value={bic} label="BIC" color="bg-purple-500" />}
        {caic && <MetricBar value={caic} label="CAIC" color="bg-indigo-500" />}
        {adj_bic && <MetricBar value={adj_bic} label="Adj BIC" color="bg-violet-500" />}
      </div>

      {/* Model summary */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-slate-700/30 rounded-lg">
        <div className="text-center">
          <div className="text-xs text-gray-400 mb-1">Log-Likelihood</div>
          <div className="text-sm font-mono text-gray-200">
            {log_likelihood !== null && log_likelihood !== undefined ? log_likelihood.toFixed(2) : 'N/A'}
          </div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-400 mb-1">Parameters</div>
          <div className="text-sm font-mono text-gray-200">{n_parameters}</div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-400 mb-1">Observations</div>
          <div className="text-sm font-mono text-gray-200">{n_observations}</div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-400 mb-1">Complexity</div>
          <div className="text-sm font-mono text-gray-200">
            {((n_parameters / n_observations) * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Interpretation guide */}
      {showDetails && (
        <div className="mt-6 bg-slate-700/30 rounded-lg p-4">
          <h5 className="font-semibold text-gray-200 mb-3 flex items-center">
            <Info className="w-4 h-4 mr-2" />
            Interpretation Guidelines
          </h5>
          <div className="space-y-2 text-xs text-gray-400">
            <div>
              <span className="font-semibold text-blue-400">AIC (Akaike Information Criterion):</span>
              <span className="ml-1">Balances fit and complexity. Lower is better. Good for prediction.</span>
            </div>
            <div>
              <span className="font-semibold text-purple-400">BIC (Bayesian Information Criterion):</span>
              <span className="ml-1">Penalizes complexity more than AIC. Prefers simpler models. Good for explanation.</span>
            </div>
            <div>
              <span className="font-semibold text-indigo-400">CAIC (Consistent AIC):</span>
              <span className="ml-1">Additional correction for small sample sizes. More conservative than AIC.</span>
            </div>
            <div>
              <span className="font-semibold text-violet-400">Adjusted BIC:</span>
              <span className="ml-1">Modified BIC accounting for prior assumptions. Balances between AIC and BIC.</span>
            </div>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-600">
            <p className="text-xs text-gray-400">
              <strong>Model Comparison:</strong> When comparing models, prefer the one with lower information criteria.
              A difference of {'>'} 10 in AIC/BIC suggests substantial evidence for the better model.
              Consider multiple criteria rather than relying on just one.
            </p>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-600">
            <p className="text-xs text-gray-400">
              <strong>Complexity Ratio:</strong> The ratio of parameters to observations indicates model complexity.
              Values {'>'} 10% suggest the model may be overfitting. Consider simplifying if ratio is high.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default ModelComparisonTable
