import { useState } from 'react'
import Plot from 'react-plotly.js'
import { CheckCircle, Award, Info, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { getPlotlyConfig } from '../utils/plotlyConfig'

const LifeDistributionResults = ({
  results,
  comparison,
  bestDistribution,
  height = 400
}) => {
  const [selectedDist, setSelectedDist] = useState(bestDistribution || 'weibull')

  if (!results) return null

  // Build survival curve traces for all distributions
  const buildSurvivalPlot = () => {
    const traces = []
    const colors = {
      weibull: '#3b82f6',
      lognormal: '#22c55e',
      exponential: '#f59e0b',
      loglogistic: '#8b5cf6'
    }

    Object.entries(results).forEach(([distName, distData]) => {
      if (distData.error || !distData.survival_curve) return

      const { times, survival } = distData.survival_curve
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: times,
        y: survival,
        name: distName.charAt(0).toUpperCase() + distName.slice(1),
        line: {
          color: colors[distName] || '#94a3b8',
          width: distName === selectedDist ? 3 : 1.5,
          dash: distName === selectedDist ? 'solid' : 'dot'
        },
        opacity: distName === selectedDist ? 1 : 0.7
      })
    })

    return traces
  }

  const layout = {
    title: {
      text: 'Fitted Life Distributions',
      font: { color: '#e2e8f0', size: 14 }
    },
    paper_bgcolor: '#1e293b',
    plot_bgcolor: '#0f172a',
    font: { color: '#e2e8f0' },
    xaxis: {
      title: 'Time',
      gridcolor: '#475569',
      tickfont: { color: '#e2e8f0' }
    },
    yaxis: {
      title: 'Reliability R(t)',
      range: [0, 1.05],
      gridcolor: '#475569',
      tickfont: { color: '#e2e8f0' }
    },
    legend: {
      x: 0.7,
      y: 0.95,
      bgcolor: 'rgba(30, 41, 59, 0.8)',
      font: { color: '#e2e8f0' }
    },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    height: height
  }

  const getFailurePatternIcon = (interpretation) => {
    if (!interpretation) return <Minus className="w-4 h-4 text-gray-400" />
    if (interpretation.includes('Decreasing')) {
      return <TrendingDown className="w-4 h-4 text-green-400" />
    } else if (interpretation.includes('Increasing')) {
      return <TrendingUp className="w-4 h-4 text-red-400" />
    }
    return <Minus className="w-4 h-4 text-yellow-400" />
  }

  return (
    <div className="space-y-4">
      {/* Distribution Comparison Table */}
      {comparison && comparison.length > 0 && (
        <div className="bg-slate-700/50 rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
            <Award className="w-4 h-4 text-yellow-400" />
            Distribution Comparison (ranked by AIC)
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 border-b border-slate-600">
                  <th className="text-left py-2 px-3">Rank</th>
                  <th className="text-left py-2 px-3">Distribution</th>
                  <th className="text-right py-2 px-3">AIC</th>
                  <th className="text-right py-2 px-3">BIC</th>
                  <th className="text-right py-2 px-3">Log-Likelihood</th>
                  <th className="text-right py-2 px-3">Median Life</th>
                </tr>
              </thead>
              <tbody>
                {comparison.map((dist, idx) => (
                  <tr
                    key={dist.distribution}
                    className={`border-b border-slate-600/50 cursor-pointer hover:bg-slate-600/50 ${
                      dist.distribution === selectedDist ? 'bg-blue-900/30' : ''
                    }`}
                    onClick={() => setSelectedDist(dist.distribution)}
                  >
                    <td className="py-2 px-3">
                      {idx === 0 ? (
                        <span className="flex items-center gap-1">
                          <CheckCircle className="w-4 h-4 text-green-400" />
                          1
                        </span>
                      ) : (
                        idx + 1
                      )}
                    </td>
                    <td className="py-2 px-3 font-medium text-gray-200">
                      {dist.distribution.charAt(0).toUpperCase() + dist.distribution.slice(1)}
                    </td>
                    <td className="py-2 px-3 text-right font-mono">
                      {dist.aic?.toFixed(2) ?? '-'}
                    </td>
                    <td className="py-2 px-3 text-right font-mono">
                      {dist.bic?.toFixed(2) ?? '-'}
                    </td>
                    <td className="py-2 px-3 text-right font-mono">
                      {dist.log_likelihood?.toFixed(2) ?? '-'}
                    </td>
                    <td className="py-2 px-3 text-right font-mono">
                      {dist.median_survival?.toFixed(2) ?? '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Survival Curves Plot */}
      <div className="bg-slate-700/50 rounded-lg p-4">
        <Plot
          data={buildSurvivalPlot()}
          layout={layout}
          config={getPlotlyConfig()}
          style={{ width: '100%' }}
        />
      </div>

      {/* Selected Distribution Details */}
      {results[selectedDist] && !results[selectedDist].error && (
        <div className="bg-slate-700/50 rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
            <Info className="w-4 h-4 text-blue-400" />
            {selectedDist.charAt(0).toUpperCase() + selectedDist.slice(1)} Distribution Parameters
          </h4>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(results[selectedDist].parameters || {}).map(([param, value]) => {
              if (param === 'interpretation') return null
              return (
                <div key={param} className="bg-slate-800 rounded p-3">
                  <div className="text-xs text-gray-400 uppercase">{param}</div>
                  <div className="text-lg font-mono text-gray-100">
                    {typeof value === 'number' ? value.toFixed(4) : value ?? '-'}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Failure Pattern Interpretation (for Weibull) */}
          {results[selectedDist].parameters?.interpretation && (
            <div className="mt-4 flex items-center gap-2 text-sm text-gray-300 bg-slate-800 rounded p-3">
              {getFailurePatternIcon(results[selectedDist].parameters.interpretation)}
              <span>{results[selectedDist].parameters.interpretation}</span>
            </div>
          )}

          {/* Fit Statistics */}
          <div className="mt-4 grid grid-cols-3 gap-4">
            <div className="bg-slate-800 rounded p-3">
              <div className="text-xs text-gray-400">AIC</div>
              <div className="text-lg font-mono text-gray-100">
                {results[selectedDist].fit_statistics?.aic?.toFixed(2) ?? '-'}
              </div>
            </div>
            <div className="bg-slate-800 rounded p-3">
              <div className="text-xs text-gray-400">BIC</div>
              <div className="text-lg font-mono text-gray-100">
                {results[selectedDist].fit_statistics?.bic?.toFixed(2) ?? '-'}
              </div>
            </div>
            <div className="bg-slate-800 rounded p-3">
              <div className="text-xs text-gray-400">Median Survival</div>
              <div className="text-lg font-mono text-gray-100">
                {results[selectedDist].median_survival_time?.toFixed(2) ?? '-'}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default LifeDistributionResults
