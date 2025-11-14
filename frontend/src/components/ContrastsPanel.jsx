import { TrendingUp, CheckCircle, XCircle, Info } from 'lucide-react'

const ContrastsPanel = ({ contrastsResult }) => {
  if (!contrastsResult) return null

  const { test_type, contrasts, summary, description, n_groups } = contrastsResult

  const getStatusIcon = (rejectNull) => {
    return rejectNull ? (
      <CheckCircle className="w-5 h-5 text-green-400" />
    ) : (
      <XCircle className="w-5 h-5 text-gray-400" />
    )
  }

  const getStatusColor = (rejectNull) => {
    return rejectNull
      ? 'border-green-500/50 bg-green-500/10'
      : 'border-gray-500/50 bg-gray-700/30'
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center space-x-3 mb-4">
        <TrendingUp className="w-6 h-6 text-purple-400" />
        <h3 className="text-xl font-bold text-gray-100">{test_type}</h3>
      </div>

      <p className="text-gray-400 text-sm mb-4">{description}</p>

      {/* Summary */}
      {summary && (
        <div className="mb-6 bg-slate-700/30 rounded-lg p-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-cyan-400">{n_groups}</div>
              <div className="text-xs text-gray-400">Groups</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-400">{summary.n_contrasts}</div>
              <div className="text-xs text-gray-400">Contrasts</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-400">{summary.n_significant}</div>
              <div className="text-xs text-gray-400">Significant</div>
            </div>
            <div>
              <div className="text-xl font-bold text-yellow-400">{summary.bonferroni_alpha}</div>
              <div className="text-xs text-gray-400">Bonferroni α</div>
            </div>
          </div>
          {summary.note && (
            <div className="mt-3 text-xs text-yellow-200 bg-yellow-900/20 border border-yellow-700/50 rounded p-2">
              <strong>Note:</strong> {summary.note}
            </div>
          )}
        </div>
      )}

      {/* Contrasts Results */}
      <div className="space-y-4">
        {contrasts.map((contrast, index) => (
          <div
            key={index}
            className={`rounded-lg border p-4 ${getStatusColor(contrast.reject_null)}`}
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-1">
                  {getStatusIcon(contrast.reject_null)}
                  <h4 className="font-semibold text-gray-100">{contrast.name}</h4>
                </div>
                <p className="text-xs text-gray-400">{contrast.interpretation}</p>
              </div>
              <div className="text-right">
                <div className={`text-lg font-bold ${contrast.reject_null ? 'text-green-400' : 'text-gray-400'}`}>
                  {contrast.reject_null ? 'Significant' : 'Not Significant'}
                </div>
                <div className="text-xs text-gray-400">p = {contrast.p_value.toFixed(6)}</div>
              </div>
            </div>

            {/* Statistics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
              <div className="bg-slate-800/50 rounded p-2">
                <div className="text-xs text-gray-400">Estimate (ψ)</div>
                <div className="text-sm font-mono text-gray-200">{contrast.contrast_estimate.toFixed(4)}</div>
              </div>
              <div className="bg-slate-800/50 rounded p-2">
                <div className="text-xs text-gray-400">Std. Error</div>
                <div className="text-sm font-mono text-gray-200">{contrast.standard_error.toFixed(4)}</div>
              </div>
              <div className="bg-slate-800/50 rounded p-2">
                <div className="text-xs text-gray-400">t-statistic</div>
                <div className="text-sm font-mono text-gray-200">{contrast.t_statistic.toFixed(4)}</div>
              </div>
              <div className="bg-slate-800/50 rounded p-2">
                <div className="text-xs text-gray-400">df</div>
                <div className="text-sm font-mono text-gray-200">{contrast.df}</div>
              </div>
            </div>

            {/* Confidence Interval */}
            <div className="bg-slate-800/50 rounded p-3 mb-3">
              <div className="text-xs text-gray-400 mb-1">95% Confidence Interval</div>
              <div className="flex items-center space-x-2">
                <span className="text-sm font-mono text-gray-200">[{contrast.ci_lower.toFixed(4)},</span>
                <span className="text-sm font-mono text-gray-200">{contrast.ci_upper.toFixed(4)}]</span>
                {contrast.ci_lower > 0 || contrast.ci_upper < 0 ? (
                  <span className="text-xs text-green-400">(does not include 0)</span>
                ) : (
                  <span className="text-xs text-gray-400">(includes 0)</span>
                )}
              </div>
            </div>

            {/* Coefficients */}
            <div className="bg-slate-800/50 rounded p-3">
              <div className="text-xs text-gray-400 mb-2">Contrast Coefficients</div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(contrast.group_means).map(([group, mean], i) => (
                  <div key={group} className="bg-slate-700/50 rounded px-3 py-1">
                    <div className="text-xs text-gray-400">{group}</div>
                    <div className="text-sm font-mono">
                      <span className="text-purple-300">{contrast.coefficients[i].toFixed(3)}</span>
                      <span className="text-gray-500 mx-1">×</span>
                      <span className="text-gray-300">{mean.toFixed(2)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Interpretation Guide */}
      <div className="mt-6 bg-slate-700/30 rounded-lg p-4">
        <div className="flex items-center space-x-2 mb-3">
          <Info className="w-4 h-4 text-blue-400" />
          <h5 className="font-semibold text-gray-200">Understanding Contrasts</h5>
        </div>
        <div className="text-xs text-gray-400 space-y-2">
          <p>
            <strong>Contrast (ψ):</strong> A weighted sum of group means using coefficients that sum to zero.
            Tests specific hypotheses about group differences.
          </p>
          <p>
            <strong>Polynomial Contrasts:</strong> Test for linear, quadratic, or cubic trends across ordered groups
            (e.g., dose levels, time points).
          </p>
          <p>
            <strong>Helmert Contrasts:</strong> Compare each group with the mean of all subsequent groups,
            useful for comparing a baseline with later conditions.
          </p>
          <p>
            <strong>Custom Contrasts:</strong> User-specified coefficients to test planned comparisons
            (e.g., control vs. treatment groups, or specific group combinations).
          </p>
        </div>
      </div>
    </div>
  )
}

export default ContrastsPanel
