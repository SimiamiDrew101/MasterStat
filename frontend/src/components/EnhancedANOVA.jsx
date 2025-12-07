import { BarChart3, AlertCircle, CheckCircle } from 'lucide-react'
import Plot from 'react-plotly.js'
import { getPlotlyConfig } from '../utils/plotlyConfig'

const EnhancedANOVA = ({ enhancedAnova, lackOfFitTest, alpha = 0.05 }) => {
  if (!enhancedAnova) return null

  const { Model: model, Residual: residual, Total: total, terms, lack_of_fit, pure_error } = enhancedAnova

  // Calculate percentage contributions
  const modelPct = ((model.sum_sq / total.sum_sq) * 100).toFixed(1)
  const residualPct = ((residual.sum_sq / total.sum_sq) * 100).toFixed(1)

  // Prepare data for contribution chart
  const termNames = Object.keys(terms || {})
  const termSS = termNames.map(name => terms[name].sum_sq)
  const termPct = termSS.map(ss => ((ss / total.sum_sq) * 100).toFixed(1))

  // Sort by SS for better visualization
  const sortedIndices = termSS
    .map((ss, idx) => ({ ss, idx }))
    .sort((a, b) => b.ss - a.ss)
    .map(item => item.idx)

  const sortedNames = sortedIndices.map(idx => termNames[idx])
  const sortedSS = sortedIndices.map(idx => termSS[idx])
  const sortedPct = sortedIndices.map(idx => termPct[idx])
  const sortedPValues = sortedIndices.map(idx => terms[termNames[idx]].p_value)

  // Bar chart of term contributions
  const barTrace = {
    type: 'bar',
    x: sortedNames,
    y: sortedSS,
    text: sortedPct.map((pct, idx) => `${pct}%<br>p=${sortedPValues[idx] || 'N/A'}`),
    textposition: 'auto',
    marker: {
      color: sortedPValues.map(p => p !== null && p < alpha ? '#22c55e' : '#64748b'),
      line: {
        color: '#1e293b',
        width: 1
      }
    },
    hovertemplate: '%{x}<br>SS: %{y:.4f}<br>%{text}<extra></extra>'
  }

  const barLayout = {
    title: {
      text: 'Sum of Squares by Term',
      font: {
        size: 16,
        color: '#f1f5f9'
      }
    },
    xaxis: {
      title: 'Model Terms',
      tickangle: -45,
      gridcolor: '#475569',
      color: '#e2e8f0'
    },
    yaxis: {
      title: 'Sum of Squares',
      gridcolor: '#475569',
      color: '#e2e8f0'
    },
    paper_bgcolor: '#334155',
    plot_bgcolor: '#1e293b',
    font: {
      color: '#e2e8f0'
    },
    margin: { l: 60, r: 40, b: 120, t: 60 },
    height: 400
  }

  const config = getPlotlyConfig('anova-contributions')

  return (
    <div className="space-y-6">
      {/* Overall ANOVA Summary */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="w-5 h-5 text-orange-400" />
          <h3 className="text-xl font-bold text-gray-100">ANOVA Table</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="bg-slate-700/70 border-b-2 border-slate-600">
                <th className="px-4 py-3 text-left text-gray-100 font-semibold">Source</th>
                <th className="px-4 py-3 text-right text-gray-100 font-semibold">DF</th>
                <th className="px-4 py-3 text-right text-gray-100 font-semibold">Sum of Squares</th>
                <th className="px-4 py-3 text-right text-gray-100 font-semibold">Mean Square</th>
                <th className="px-4 py-3 text-right text-gray-100 font-semibold">F-Value</th>
                <th className="px-4 py-3 text-right text-gray-100 font-semibold">p-value</th>
                <th className="px-4 py-3 text-center text-gray-100 font-semibold">Contribution</th>
              </tr>
            </thead>
            <tbody>
              {/* Model Row */}
              <tr className="border-b border-slate-700/50 bg-green-900/10 hover:bg-green-900/20">
                <td className="px-4 py-3 text-gray-100 font-semibold">Model</td>
                <td className="px-4 py-3 text-right text-gray-100 font-mono">{model.df}</td>
                <td className="px-4 py-3 text-right text-gray-100 font-mono">{model.sum_sq.toFixed(4)}</td>
                <td className="px-4 py-3 text-right text-gray-100 font-mono">{model.mean_sq.toFixed(4)}</td>
                <td className="px-4 py-3 text-right text-gray-100 font-mono">{model.F?.toFixed(4) || 'N/A'}</td>
                <td className="px-4 py-3 text-right font-mono">
                  <span className={model.p_value < alpha ? 'text-green-400' : 'text-gray-400'}>
                    {model.p_value < 0.0001 ? '<0.0001' : model.p_value?.toFixed(6)}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className="px-2 py-1 bg-green-900/50 text-green-200 rounded text-sm font-medium">
                    {modelPct}%
                  </span>
                </td>
              </tr>

              {/* Lack of Fit (if available) */}
              {lack_of_fit && (
                <tr className="border-b border-slate-700/50 hover:bg-slate-600/10">
                  <td className="px-4 py-3 text-gray-300 pl-8">└─ Lack of Fit</td>
                  <td className="px-4 py-3 text-right text-gray-300 font-mono">{lack_of_fit.df}</td>
                  <td className="px-4 py-3 text-right text-gray-300 font-mono">{lack_of_fit.ss.toFixed(4)}</td>
                  <td className="px-4 py-3 text-right text-gray-300 font-mono">{lack_of_fit.ms.toFixed(4)}</td>
                  <td className="px-4 py-3 text-right text-gray-300 font-mono">{lackOfFitTest?.f_statistic?.toFixed(4) || 'N/A'}</td>
                  <td className="px-4 py-3 text-right text-gray-300 font-mono">
                    {lackOfFitTest?.p_value ? (
                      <span className={lackOfFitTest.significant_lof ? 'text-red-400' : 'text-green-400'}>
                        {lackOfFitTest.p_value.toFixed(6)}
                      </span>
                    ) : 'N/A'}
                  </td>
                  <td className="px-4 py-3 text-center text-gray-400">-</td>
                </tr>
              )}

              {/* Pure Error (if available) */}
              {pure_error && (
                <tr className="border-b border-slate-700/50 hover:bg-slate-600/10">
                  <td className="px-4 py-3 text-gray-300 pl-8">└─ Pure Error</td>
                  <td className="px-4 py-3 text-right text-gray-300 font-mono">{pure_error.df}</td>
                  <td className="px-4 py-3 text-right text-gray-300 font-mono">{pure_error.ss.toFixed(4)}</td>
                  <td className="px-4 py-3 text-right text-gray-300 font-mono">{pure_error.ms.toFixed(4)}</td>
                  <td className="px-4 py-3 text-right text-gray-400">-</td>
                  <td className="px-4 py-3 text-right text-gray-400">-</td>
                  <td className="px-4 py-3 text-center text-gray-400">-</td>
                </tr>
              )}

              {/* Residual/Error Row */}
              {!lack_of_fit && (
                <tr className="border-b border-slate-700/50 bg-red-900/10 hover:bg-red-900/20">
                  <td className="px-4 py-3 text-gray-100 font-semibold">Residual Error</td>
                  <td className="px-4 py-3 text-right text-gray-100 font-mono">{residual.df}</td>
                  <td className="px-4 py-3 text-right text-gray-100 font-mono">{residual.sum_sq.toFixed(4)}</td>
                  <td className="px-4 py-3 text-right text-gray-100 font-mono">{residual.mean_sq.toFixed(4)}</td>
                  <td className="px-4 py-3 text-right text-gray-400">-</td>
                  <td className="px-4 py-3 text-right text-gray-400">-</td>
                  <td className="px-4 py-3 text-center">
                    <span className="px-2 py-1 bg-red-900/50 text-red-200 rounded text-sm font-medium">
                      {residualPct}%
                    </span>
                  </td>
                </tr>
              )}

              {/* Total Row */}
              <tr className="border-t-2 border-slate-600 bg-slate-700/50">
                <td className="px-4 py-3 text-gray-100 font-bold">Total</td>
                <td className="px-4 py-3 text-right text-gray-100 font-mono font-bold">{total.df}</td>
                <td className="px-4 py-3 text-right text-gray-100 font-mono font-bold">{total.sum_sq.toFixed(4)}</td>
                <td className="px-4 py-3 text-right text-gray-400">-</td>
                <td className="px-4 py-3 text-right text-gray-400">-</td>
                <td className="px-4 py-3 text-right text-gray-400">-</td>
                <td className="px-4 py-3 text-center">
                  <span className="px-2 py-1 bg-slate-600 text-gray-200 rounded text-sm font-medium">
                    100%
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Lack of Fit Interpretation */}
        {lackOfFitTest && (
          <div className={`mt-4 p-4 rounded-lg border ${lackOfFitTest.significant_lof ? 'bg-red-900/20 border-red-700/30' : 'bg-green-900/20 border-green-700/30'}`}>
            <div className="flex items-center gap-2 mb-2">
              {lackOfFitTest.significant_lof ? (
                <AlertCircle className="w-5 h-5 text-red-400" />
              ) : (
                <CheckCircle className="w-5 h-5 text-green-400" />
              )}
              <h4 className={`font-semibold ${lackOfFitTest.significant_lof ? 'text-red-200' : 'text-green-200'}`}>
                Lack of Fit Test
              </h4>
            </div>
            <p className={`text-sm ${lackOfFitTest.significant_lof ? 'text-red-100' : 'text-green-100'}`}>
              {lackOfFitTest.significant_lof
                ? 'Significant lack of fit detected (p < 0.05). The model may not adequately describe the data. Consider adding higher-order terms or checking for outliers.'
                : 'No significant lack of fit (p ≥ 0.05). The model adequately describes the data.'}
            </p>
          </div>
        )}
      </div>

      {/* Detailed Term Breakdown */}
      {terms && Object.keys(terms).length > 0 && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-xl font-bold text-gray-100 mb-4">Individual Term Analysis</h3>

          <div className="overflow-x-auto mb-4">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-slate-700/70 border-b border-slate-600">
                  <th className="px-4 py-2 text-left text-gray-100 font-semibold">Term</th>
                  <th className="px-4 py-2 text-right text-gray-100 font-semibold">DF</th>
                  <th className="px-4 py-2 text-right text-gray-100 font-semibold">Sum of Squares</th>
                  <th className="px-4 py-2 text-right text-gray-100 font-semibold">F-Value</th>
                  <th className="px-4 py-2 text-right text-gray-100 font-semibold">p-value</th>
                  <th className="px-4 py-2 text-center text-gray-100 font-semibold">Significance</th>
                  <th className="px-4 py-2 text-center text-gray-100 font-semibold">% Contribution</th>
                </tr>
              </thead>
              <tbody>
                {sortedNames.map((term, idx) => {
                  const termData = terms[term]
                  const pValue = termData.p_value
                  const isSignificant = pValue !== null && pValue < alpha

                  return (
                    <tr key={term} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                      <td className="px-4 py-2 text-gray-100 font-mono text-sm">{term}</td>
                      <td className="px-4 py-2 text-right text-gray-100 font-mono">{termData.df}</td>
                      <td className="px-4 py-2 text-right text-gray-100 font-mono">{termData.sum_sq.toFixed(4)}</td>
                      <td className="px-4 py-2 text-right text-gray-100 font-mono">{termData.F?.toFixed(4) || 'N/A'}</td>
                      <td className="px-4 py-2 text-right font-mono">
                        <span className={isSignificant ? 'text-green-400' : 'text-gray-400'}>
                          {pValue !== null ? (pValue < 0.0001 ? '<0.0001' : pValue.toFixed(6)) : 'N/A'}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-center">
                        {pValue !== null && (
                          <span className={`px-2 py-1 rounded text-xs font-medium ${isSignificant ? 'bg-green-900/50 text-green-200' : 'bg-slate-700 text-gray-400'}`}>
                            {isSignificant ? '***' : 'ns'}
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-2 text-center text-gray-300">{sortedPct[idx]}%</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Contribution Chart */}
          <div className="bg-slate-700/30 rounded-lg p-4">
            <Plot
              data={[barTrace]}
              layout={barLayout}
              config={config}
              style={{ width: '100%' }}
            />
            <p className="text-xs text-gray-400 mt-2">
              Green bars indicate statistically significant terms (p &lt; {alpha}). Bar height shows contribution to total variation.
            </p>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-700/30">
        <h4 className="text-blue-200 font-semibold mb-2">Interpretation Notes</h4>
        <ul className="space-y-1 text-sm text-blue-100">
          <li><strong>Model p-value &lt; 0.05:</strong> The model is statistically significant</li>
          <li><strong>High R² (Model %):</strong> Model explains large portion of variation</li>
          <li><strong>Significant terms:</strong> Have meaningful effect on response (p &lt; {alpha})</li>
          <li><strong>Lack of Fit test:</strong> Checks if model form is adequate (want p &gt; 0.05)</li>
        </ul>
      </div>
    </div>
  )
}

export default EnhancedANOVA
