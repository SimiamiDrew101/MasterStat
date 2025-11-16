import { TrendingUp, AlertTriangle, CheckCircle, Info } from 'lucide-react'
import Plot from 'react-plotly.js'

/**
 * AncovaResults component displays ANCOVA analysis results
 * Shows homogeneity of slopes test, adjusted means, and covariate effect
 */
const AncovaResults = ({ ancovaData, unadjustedMeans }) => {
  if (!ancovaData || ancovaData.error) {
    if (ancovaData?.error) {
      return (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200 font-medium">ANCOVA Error: {ancovaData.error}</p>
        </div>
      )
    }
    return null
  }

  // Prepare comparison data for adjusted vs unadjusted means
  const treatmentLabels = Object.keys(ancovaData.adjusted_treatment_means)
  const adjustedValues = Object.values(ancovaData.adjusted_treatment_means)
  const unadjustedValues = treatmentLabels.map(label => unadjustedMeans[label] || 0)

  const comparisonPlotData = [
    {
      type: 'bar',
      name: 'Unadjusted Means',
      x: treatmentLabels,
      y: unadjustedValues,
      marker: { color: 'rgba(99, 102, 241, 0.6)' },
      text: unadjustedValues.map(v => v.toFixed(2)),
      textposition: 'auto'
    },
    {
      type: 'bar',
      name: 'Adjusted Means (ANCOVA)',
      x: treatmentLabels,
      y: adjustedValues,
      marker: { color: 'rgba(16, 185, 129, 0.8)' },
      text: adjustedValues.map(v => v.toFixed(2)),
      textposition: 'auto'
    }
  ]

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50 space-y-6">
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp className="w-6 h-6 text-emerald-400" />
        <h3 className="text-xl font-bold text-gray-100">ANCOVA Results</h3>
        <span className="text-sm text-gray-400 ml-2">(Covariate: {ancovaData.covariate_name})</span>
      </div>

      {/* Homogeneity of Slopes Test */}
      <div className="bg-slate-700/30 rounded-lg p-5">
        <div className="flex items-center gap-2 mb-3">
          {ancovaData.slopes_homogeneous ? (
            <CheckCircle className="w-5 h-5 text-green-400" />
          ) : (
            <AlertTriangle className="w-5 h-5 text-orange-400" />
          )}
          <h4 className="font-semibold text-gray-100">Homogeneity of Slopes Test</h4>
        </div>

        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Test:</span>
            <span className="text-gray-200">Treatment × Covariate Interaction</span>
          </div>
          {ancovaData.interaction_p_value !== null && (
            <div className="flex justify-between">
              <span className="text-gray-400">p-value:</span>
              <span className={`font-mono font-semibold ${
                ancovaData.slopes_homogeneous ? 'text-green-400' : 'text-orange-400'
              }`}>
                {ancovaData.interaction_p_value.toFixed(6)}
              </span>
            </div>
          )}
          <div className="flex justify-between items-center mt-3 pt-3 border-t border-slate-600">
            <span className="text-gray-400">Assumption:</span>
            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
              ancovaData.slopes_homogeneous
                ? 'bg-green-500/20 text-green-400'
                : 'bg-orange-500/20 text-orange-400'
            }`}>
              {ancovaData.slopes_homogeneous ? 'Met' : 'Violated'}
            </span>
          </div>
        </div>

        {ancovaData.warning && (
          <div className="mt-4 bg-orange-900/20 border border-orange-700/50 rounded-lg p-3">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-5 h-5 text-orange-400 mt-0.5 flex-shrink-0" />
              <p className="text-sm text-orange-200">{ancovaData.warning}</p>
            </div>
          </div>
        )}

        <div className="mt-4 bg-slate-800/50 rounded p-3">
          <p className="text-xs text-gray-400">
            <strong className="text-gray-300">Note:</strong> The homogeneity of slopes assumption requires
            that the relationship between the covariate and response is the same across all treatment groups.
            If violated (p &lt; 0.05), treatment effects may depend on covariate level.
          </p>
        </div>
      </div>

      {/* Covariate Effect */}
      <div className="bg-slate-700/30 rounded-lg p-5">
        <h4 className="font-semibold text-gray-100 mb-3">Covariate Effect</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-800/50 rounded p-3">
            <div className="text-xs text-gray-400 mb-1">Covariate Mean</div>
            <div className="text-2xl font-bold text-cyan-400">{ancovaData.covariate_mean.toFixed(3)}</div>
          </div>
          <div className="bg-slate-800/50 rounded p-3">
            <div className="text-xs text-gray-400 mb-1">Regression Coefficient</div>
            <div className="text-2xl font-bold text-emerald-400">{ancovaData.covariate_coefficient.toFixed(4)}</div>
          </div>
        </div>
        <p className="text-xs text-gray-400 mt-3">
          The covariate explains variability in the response, increasing precision of treatment comparisons.
        </p>
      </div>

      {/* ANCOVA Table */}
      {ancovaData.ancova_table && (
        <div className="bg-slate-700/30 rounded-lg p-5">
          <h4 className="font-semibold text-gray-100 mb-3">ANCOVA Table</h4>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-slate-700/70">
                  <th className="px-4 py-2 text-left text-gray-100 font-semibold text-sm border-b-2 border-slate-600">
                    Source
                  </th>
                  <th className="px-4 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-slate-600">
                    Sum of Squares
                  </th>
                  <th className="px-4 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-slate-600">
                    df
                  </th>
                  <th className="px-4 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-slate-600">
                    F
                  </th>
                  <th className="px-4 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-slate-600">
                    p-value
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(ancovaData.ancova_table).map(([source, values], idx) => (
                  <tr key={idx} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                    <td className="px-4 py-2 text-gray-200 font-medium text-sm">{source}</td>
                    <td className="px-4 py-2 text-center text-gray-300 text-sm font-mono">
                      {values.sum_sq.toFixed(4)}
                    </td>
                    <td className="px-4 py-2 text-center text-gray-300 text-sm font-mono">{values.df}</td>
                    <td className="px-4 py-2 text-center text-gray-300 text-sm font-mono">
                      {values.F ? values.F.toFixed(4) : '-'}
                    </td>
                    <td className="px-4 py-2 text-center text-sm">
                      {values.p_value !== null ? (
                        <span className={`font-mono font-semibold ${
                          values.significant ? 'text-emerald-400' : 'text-gray-400'
                        }`}>
                          {values.p_value < 0.0001 ? '<0.0001' : values.p_value.toFixed(4)}
                        </span>
                      ) : (
                        <span className="text-gray-500">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Adjusted vs Unadjusted Means Comparison */}
      <div className="bg-slate-700/30 rounded-lg p-5">
        <div className="flex items-center gap-2 mb-4">
          <Info className="w-5 h-5 text-blue-400" />
          <h4 className="font-semibold text-gray-100">Adjusted vs Unadjusted Treatment Means</h4>
        </div>

        <Plot
          data={comparisonPlotData}
          layout={{
            paper_bgcolor: '#334155',
            plot_bgcolor: '#1e293b',
            font: { color: '#e2e8f0' },
            xaxis: {
              title: 'Treatment',
              gridcolor: '#475569',
              color: '#e2e8f0'
            },
            yaxis: {
              title: 'Mean Response',
              gridcolor: '#475569',
              color: '#e2e8f0'
            },
            barmode: 'group',
            showlegend: true,
            legend: {
              bgcolor: 'rgba(30, 41, 59, 0.8)',
              bordercolor: '#64748b',
              borderwidth: 1,
              x: 1,
              xanchor: 'right',
              y: 1
            },
            height: 400,
            margin: { l: 60, r: 40, b: 60, t: 40 }
          }}
          config={{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            toImageButtonOptions: {
              format: 'png',
              filename: `ancova-comparison-${new Date().toISOString().split('T')[0]}`,
              height: 400,
              width: 700,
              scale: 2
            }
          }}
          style={{ width: '100%' }}
        />

        <div className="mt-4 bg-blue-900/20 border border-blue-700/30 rounded-lg p-3">
          <p className="text-sm text-blue-100">
            <strong>Adjusted means</strong> control for the effect of the covariate ({ancovaData.covariate_name}),
            providing more precise treatment comparisons. Differences between adjusted and unadjusted means indicate
            how much the covariate was influencing the raw treatment means.
          </p>
        </div>
      </div>

      {/* Model Quality */}
      <div className="bg-gradient-to-r from-emerald-900/20 to-blue-900/20 rounded-lg p-4 border border-emerald-700/30">
        <h5 className="font-semibold text-emerald-200 mb-2">Model Quality</h5>
        <div className="flex items-center gap-4">
          <div>
            <span className="text-gray-400 text-sm">ANCOVA R²:</span>
            <span className="text-emerald-400 font-bold text-lg ml-2">
              {ancovaData.model_r_squared.toFixed(4)}
            </span>
          </div>
          <div className="text-xs text-gray-400">
            (Proportion of variance explained including covariate)
          </div>
        </div>
      </div>
    </div>
  )
}

export default AncovaResults
