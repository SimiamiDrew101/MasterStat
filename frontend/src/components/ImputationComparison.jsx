import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { GitCompare, TrendingUp, BarChart3, AlertCircle, CheckCircle2 } from 'lucide-react'
import { getPlotlyConfig, getPlotlyLayout } from '../utils/plotlyConfig'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * ImputationComparison - Component for comparing multiple imputation methods
 * Shows side-by-side comparison with metrics and visualizations
 */
const ImputationComparison = ({ data, columnName = 'Response', onSelectMethod }) => {
  const [comparisonResults, setComparisonResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedMethods, setSelectedMethods] = useState(['mean', 'median', 'knn', 'mice'])

  const availableMethods = [
    { value: 'mean', label: 'Mean', color: '#818cf8' },
    { value: 'median', label: 'Median', color: '#22d3ee' },
    { value: 'knn', label: 'KNN', color: '#34d399' },
    { value: 'mice', label: 'MICE', color: '#f472b6' },
    { value: 'linear', label: 'Linear Interp.', color: '#fb923c' },
    { value: 'locf', label: 'LOCF', color: '#a78bfa' }
  ]

  useEffect(() => {
    if (data && data.length > 0) {
      compareImputationMethods()
    }
  }, [data, selectedMethods])

  const compareImputationMethods = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post(`${API_URL}/api/imputation/compare`, {
        data: data,
        methods: selectedMethods,
        parameters: {
          knn: { knn_neighbors: 5 },
          mice: { mice_iterations: 10, mice_random_state: 42 }
        }
      })

      setComparisonResults(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to compare methods')
    } finally {
      setLoading(false)
    }
  }

  const toggleMethod = (method) => {
    if (selectedMethods.includes(method)) {
      if (selectedMethods.length > 1) {
        setSelectedMethods(selectedMethods.filter(m => m !== method))
      }
    } else {
      setSelectedMethods([...selectedMethods, method])
    }
  }

  // Create Q-Q plot data
  const createQQPlotData = () => {
    if (!comparisonResults || !comparisonResults.comparison) return []

    const traces = []

    // Get original observed values
    const observed = data.filter(v => v !== null && v !== undefined && !isNaN(v))
    observed.sort((a, b) => a - b)

    const n = observed.length
    const theoreticalQuantiles = observed.map((_, i) => {
      const p = (i + 0.5) / n
      // Calculate z-score for probability p (inverse normal CDF approximation)
      const z = p < 0.5
        ? -Math.sqrt(-2 * Math.log(p))
        : Math.sqrt(-2 * Math.log(1 - p))
      return z
    })

    // Original data trace
    traces.push({
      x: theoreticalQuantiles,
      y: observed,
      mode: 'markers',
      name: 'Original (Observed)',
      marker: { color: '#94a3b8', size: 6, opacity: 0.7 }
    })

    // Imputed data traces
    Object.entries(comparisonResults.comparison).forEach(([method, result]) => {
      if (!result.error && result.imputed_values_sample) {
        const methodInfo = availableMethods.find(m => m.value === method)
        const imputedSample = result.imputed_values_sample.slice().sort((a, b) => a - b)

        traces.push({
          x: theoreticalQuantiles.slice(0, imputedSample.length),
          y: imputedSample,
          mode: 'markers',
          name: methodInfo?.label || method,
          marker: { color: methodInfo?.color || '#888', size: 8, opacity: 0.8 }
        })
      }
    })

    // Add reference line
    if (observed.length > 0) {
      const minVal = Math.min(...observed)
      const maxVal = Math.max(...observed)
      traces.push({
        x: [Math.min(...theoreticalQuantiles), Math.max(...theoreticalQuantiles)],
        y: [minVal, maxVal],
        mode: 'lines',
        name: 'Reference',
        line: { color: '#64748b', width: 2, dash: 'dash' },
        showlegend: false
      })
    }

    return traces
  }

  // Create distribution comparison plot
  const createDistributionPlot = () => {
    if (!comparisonResults || !comparisonResults.comparison) return []

    const traces = []

    // Original data distribution
    const observed = data.filter(v => v !== null && v !== undefined && !isNaN(v))
    traces.push({
      x: observed,
      type: 'histogram',
      name: 'Original (Observed)',
      opacity: 0.5,
      marker: { color: '#94a3b8' },
      nbinsx: 20
    })

    return traces
  }

  // Create metrics comparison
  const createMetricsComparison = () => {
    if (!comparisonResults || !comparisonResults.comparison) return null

    const methods = Object.entries(comparisonResults.comparison)
      .filter(([_, result]) => !result.error)

    return (
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-600">
              <th className="text-left py-3 px-4 text-gray-300 font-semibold">Method</th>
              <th className="text-right py-3 px-4 text-gray-300 font-semibold">CV RMSE</th>
              <th className="text-right py-3 px-4 text-gray-300 font-semibold">Imputed Mean</th>
              <th className="text-right py-3 px-4 text-gray-300 font-semibold">Imputed Std</th>
              <th className="text-right py-3 px-4 text-gray-300 font-semibold">KS p-value</th>
              <th className="text-center py-3 px-4 text-gray-300 font-semibold">Distribution</th>
              <th className="text-center py-3 px-4 text-gray-300 font-semibold">Action</th>
            </tr>
          </thead>
          <tbody>
            {methods.map(([method, result]) => {
              const methodInfo = availableMethods.find(m => m.value === method)
              const preserved = result.distribution_preservation?.distribution_preserved

              return (
                <tr key={method} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: methodInfo?.color || '#888' }}
                      />
                      <span className="font-medium text-gray-200">{methodInfo?.label || method}</span>
                    </div>
                  </td>
                  <td className="text-right py-3 px-4 font-mono text-gray-300">
                    {result.cv_rmse !== null && result.cv_rmse !== undefined
                      ? result.cv_rmse.toFixed(4)
                      : <span className="text-gray-500">N/A</span>}
                  </td>
                  <td className="text-right py-3 px-4 font-mono text-gray-300">
                    {result.imputed_mean?.toFixed(3)}
                  </td>
                  <td className="text-right py-3 px-4 font-mono text-gray-300">
                    {result.imputed_std?.toFixed(3)}
                  </td>
                  <td className="text-right py-3 px-4 font-mono text-gray-300">
                    {result.distribution_preservation?.ks_pvalue?.toFixed(4)}
                  </td>
                  <td className="text-center py-3 px-4">
                    {preserved ? (
                      <div className="flex items-center justify-center gap-1 text-green-400">
                        <CheckCircle2 className="w-4 h-4" />
                        <span className="text-xs">Preserved</span>
                      </div>
                    ) : (
                      <div className="flex items-center justify-center gap-1 text-orange-400">
                        <AlertCircle className="w-4 h-4" />
                        <span className="text-xs">Changed</span>
                      </div>
                    )}
                  </td>
                  <td className="text-center py-3 px-4">
                    {onSelectMethod && (
                      <button
                        onClick={() => onSelectMethod(method)}
                        className="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white text-xs rounded transition"
                      >
                        Select
                      </button>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Method Selection */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-2 mb-4">
          <GitCompare className="w-5 h-5 text-indigo-400" />
          <h3 className="text-xl font-bold text-gray-100">Compare Imputation Methods</h3>
        </div>

        <div className="space-y-4">
          <div>
            <p className="text-sm text-gray-400 mb-3">Select methods to compare:</p>
            <div className="flex flex-wrap gap-2">
              {availableMethods.map(method => (
                <button
                  key={method.value}
                  onClick={() => toggleMethod(method.value)}
                  className={`px-4 py-2 rounded-lg border-2 transition ${
                    selectedMethods.includes(method.value)
                      ? 'border-indigo-500 bg-indigo-900/30 text-indigo-200'
                      : 'border-slate-600 bg-slate-700/30 text-gray-400 hover:border-slate-500'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: method.color }}
                    />
                    {method.label}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {loading && (
            <div className="bg-slate-700/30 rounded-lg p-4">
              <div className="flex items-center gap-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-400"></div>
                <p className="text-gray-300">Comparing imputation methods...</p>
              </div>
            </div>
          )}

          {error && (
            <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-red-400" />
                <p className="text-red-200">{error}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Recommendations */}
      {comparisonResults && comparisonResults.recommendations && (
        <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-5">
          <div className="flex items-start gap-3">
            <TrendingUp className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="font-semibold text-blue-200 mb-2">Recommendations</h4>
              <p className="text-sm text-blue-100">{comparisonResults.recommendations}</p>
              {comparisonResults.n_missing && (
                <div className="mt-3 text-sm text-blue-200">
                  <span className="font-semibold">Missing data:</span> {comparisonResults.n_missing} values
                  ({comparisonResults.percent_missing?.toFixed(1)}% of total)
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Metrics Comparison Table */}
      {comparisonResults && comparisonResults.comparison && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="w-5 h-5 text-indigo-400" />
            <h3 className="text-xl font-bold text-gray-100">Method Comparison</h3>
          </div>

          {createMetricsComparison()}

          <div className="mt-4 p-3 bg-slate-700/30 rounded-lg">
            <p className="text-xs text-gray-400">
              <strong>Interpretation:</strong> <strong>CV RMSE</strong> (Cross-Validation Root Mean Squared Error) measures imputation accuracy by hiding 20% of observed values and comparing predicted vs actual (lower is better).
              <strong>KS test</strong> compares distributions - p-value {'>'} 0.05 suggests the imputed data distribution is similar to the original.
              "Distribution Preserved" indicates the method maintains the original data characteristics.
            </p>
          </div>
        </div>
      )}

      {/* Q-Q Plot Comparison */}
      {comparisonResults && comparisonResults.comparison && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-gray-100 mb-4">Q-Q Plot Comparison</h3>

          <Plot
            data={createQQPlotData()}
            layout={{
              ...getPlotlyLayout('Quantile-Quantile Plot'),
              xaxis: {
                title: 'Theoretical Quantiles',
                color: '#cbd5e1',
                gridcolor: 'rgba(51, 65, 85, 0.5)',
                zerolinecolor: 'rgba(71, 85, 105, 0.7)'
              },
              yaxis: {
                title: columnName,
                color: '#cbd5e1',
                gridcolor: 'rgba(51, 65, 85, 0.5)',
                zerolinecolor: 'rgba(71, 85, 105, 0.7)'
              },
              showlegend: true,
              legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(30, 41, 59, 0.8)',
                bordercolor: 'rgba(71, 85, 105, 0.5)',
                borderwidth: 1
              },
              height: 500
            }}
            config={getPlotlyConfig()}
            style={{ width: '100%' }}
          />

          <div className="mt-4 p-3 bg-slate-700/30 rounded-lg">
            <p className="text-xs text-gray-400">
              <strong>Q-Q Plot:</strong> Points closer to the reference line indicate better distribution match.
              This helps assess how well each imputation method preserves the original data distribution.
            </p>
          </div>
        </div>
      )}

      {/* Distribution Comparison */}
      {comparisonResults && comparisonResults.comparison && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-gray-100 mb-4">Distribution Analysis</h3>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {Object.entries(comparisonResults.comparison)
              .filter(([_, result]) => !result.error)
              .map(([method, result]) => {
                const methodInfo = availableMethods.find(m => m.value === method)

                return (
                  <div key={method} className="bg-slate-700/30 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: methodInfo?.color || '#888' }}
                      />
                      <h4 className="font-semibold text-gray-200">{methodInfo?.label || method}</h4>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <span className="text-gray-400">Mean:</span>{' '}
                        <span className="text-gray-200 font-mono">{result.statistics?.imputed_mean?.toFixed(3)}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Std:</span>{' '}
                        <span className="text-gray-200 font-mono">{result.statistics?.imputed_std?.toFixed(3)}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Mean Diff:</span>{' '}
                        <span className="text-gray-200 font-mono">
                          {result.distribution_preservation?.mean_difference?.toFixed(3)}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-400">Var Ratio:</span>{' '}
                        <span className="text-gray-200 font-mono">
                          {result.distribution_preservation?.variance_ratio?.toFixed(3)}
                        </span>
                      </div>
                    </div>
                  </div>
                )
              })}
          </div>
        </div>
      )}
    </div>
  )
}

export default ImputationComparison
