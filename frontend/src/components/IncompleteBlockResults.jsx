import { TrendingUp, AlertTriangle, CheckCircle, Info, Grid3x3 } from 'lucide-react'

/**
 * IncompleteBlockResults component displays incomplete block design analysis results
 * Shows design info, treatment effects, block effects, and efficiency metrics
 */
const IncompleteBlockResults = ({ incompleteData }) => {
  if (!incompleteData || incompleteData.error) {
    if (incompleteData?.error) {
      return (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200 font-medium">Incomplete Block Analysis Error: {incompleteData.error}</p>
        </div>
      )
    }
    return null
  }

  const { design_info, treatment_effect, block_effect, treatment_means,
          block_means, anova_table, efficiency, mse, r_squared, adj_r_squared } = incompleteData

  return (
    <div className="space-y-6">
      {/* Design Info */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-2 mb-4">
          <Grid3x3 className="w-6 h-6 text-indigo-400" />
          <h3 className="text-xl font-bold text-gray-100">Incomplete Block Design Info</h3>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Treatments</div>
            <div className="text-2xl font-bold text-indigo-400">{design_info.n_treatments}</div>
          </div>
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Blocks</div>
            <div className="text-2xl font-bold text-cyan-400">{design_info.n_blocks}</div>
          </div>
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Block Size</div>
            <div className="text-2xl font-bold text-emerald-400">{design_info.block_size}</div>
          </div>
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Total Runs</div>
            <div className="text-2xl font-bold text-purple-400">{design_info.total_runs}</div>
          </div>
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Efficiency</div>
            <div className="text-2xl font-bold text-orange-400">{(efficiency * 100).toFixed(1)}%</div>
          </div>
        </div>

        <div className="mt-4 flex gap-3">
          <div className={`px-3 py-1.5 rounded-lg text-sm font-semibold ${
            design_info.is_incomplete
              ? 'bg-orange-900/30 border border-orange-700/50 text-orange-300'
              : 'bg-green-900/30 border border-green-700/50 text-green-300'
          }`}>
            {design_info.is_incomplete ? 'Incomplete Design' : 'Complete Design'}
          </div>
          <div className={`px-3 py-1.5 rounded-lg text-sm font-semibold ${
            design_info.is_balanced
              ? 'bg-green-900/30 border border-green-700/50 text-green-300'
              : 'bg-orange-900/30 border border-orange-700/50 text-orange-300'
          }`}>
            {design_info.is_balanced ? 'Balanced' : 'Unbalanced'}
          </div>
        </div>

        {design_info.is_incomplete && (
          <div className="mt-4 bg-blue-900/20 border border-blue-700/30 rounded-lg p-3">
            <p className="text-xs text-blue-100">
              <strong>Incomplete Block Design:</strong> Not all treatments appear in every block.
              This is useful when block sizes must be smaller than the number of treatments
              (e.g., limited resources, experimental constraints). Efficiency of {(efficiency * 100).toFixed(1)}%
              relative to RCBD.
            </p>
          </div>
        )}
      </div>

      {/* Treatment Effect */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-2 mb-4">
          {treatment_effect.significant ? (
            <CheckCircle className="w-6 h-6 text-green-400" />
          ) : (
            <AlertTriangle className="w-6 h-6 text-orange-400" />
          )}
          <h4 className="font-semibold text-gray-100">Treatment Effect</h4>
        </div>

        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-700/30 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">F-statistic</div>
              <div className="text-xl font-bold text-blue-400 font-mono">{treatment_effect.F_statistic.toFixed(4)}</div>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">p-value</div>
              <div className={`text-xl font-bold font-mono ${
                treatment_effect.significant ? 'text-green-400' : 'text-orange-400'
              }`}>
                {treatment_effect.p_value.toFixed(4)}
              </div>
            </div>
          </div>

          <div className={`px-4 py-2 rounded-lg ${
            treatment_effect.significant
              ? 'bg-green-900/20 border border-green-700/50'
              : 'bg-orange-900/20 border border-orange-700/50'
          }`}>
            <p className="text-sm font-semibold">{treatment_effect.interpretation}</p>
          </div>

          <div className="bg-slate-700/30 rounded-lg p-4">
            <div className="text-xs text-gray-400 mb-2">Treatment Means:</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {Object.entries(treatment_means).map(([treatment, mean]) => (
                <div key={treatment} className="bg-slate-800/50 rounded px-3 py-2">
                  <div className="text-xs text-gray-400">{treatment}</div>
                  <div className="text-lg font-bold text-indigo-400 font-mono">{mean.toFixed(2)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Block Effect */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-2 mb-4">
          {block_effect.significant ? (
            <CheckCircle className="w-6 h-6 text-green-400" />
          ) : (
            <Info className="w-6 h-6 text-blue-400" />
          )}
          <h4 className="font-semibold text-gray-100">Block Effect</h4>
        </div>

        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-700/30 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">F-statistic</div>
              <div className="text-xl font-bold text-blue-400 font-mono">{block_effect.F_statistic.toFixed(4)}</div>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">p-value</div>
              <div className={`text-xl font-bold font-mono ${
                block_effect.significant ? 'text-green-400' : 'text-gray-400'
              }`}>
                {block_effect.p_value.toFixed(4)}
              </div>
            </div>
          </div>

          <div className={`px-4 py-2 rounded-lg ${
            block_effect.significant
              ? 'bg-green-900/20 border border-green-700/50'
              : 'bg-slate-700/30 border border-slate-600'
          }`}>
            <p className="text-sm font-semibold">
              {block_effect.significant
                ? 'Significant block differences - blocking was effective'
                : 'No significant block differences - blocks were homogeneous'}
            </p>
          </div>
        </div>
      </div>

      {/* ANOVA Table */}
      {anova_table && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <h4 className="font-semibold text-gray-100 mb-3">ANOVA Table</h4>
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
                {Object.entries(anova_table).map(([source, values], idx) => (
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
                          values.significant ? 'text-green-400' : 'text-gray-400'
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

      {/* Model Quality */}
      <div className="bg-gradient-to-r from-indigo-900/20 to-purple-900/20 rounded-lg p-4 border border-indigo-700/30">
        <h5 className="font-semibold text-indigo-200 mb-2">Model Quality</h5>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-400">R²:</span>
            <span className="text-indigo-400 font-bold text-lg ml-2">{r_squared.toFixed(4)}</span>
          </div>
          <div>
            <span className="text-gray-400">Adjusted R²:</span>
            <span className="text-indigo-400 font-bold text-lg ml-2">{adj_r_squared.toFixed(4)}</span>
          </div>
          <div>
            <span className="text-gray-400">MSE:</span>
            <span className="text-indigo-400 font-mono ml-2">{mse.toFixed(4)}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default IncompleteBlockResults
