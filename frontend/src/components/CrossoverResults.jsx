import { TrendingUp, AlertTriangle, CheckCircle, Activity } from 'lucide-react'

/**
 * CrossoverResults component displays crossover design analysis results
 * Shows treatment effects, period effects, and carryover effects
 */
const CrossoverResults = ({ crossoverData }) => {
  if (!crossoverData || crossoverData.error) {
    if (crossoverData?.error) {
      return (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200 font-medium">Crossover Analysis Error: {crossoverData.error}</p>
        </div>
      )
    }
    return null
  }

  const { design_type, n_subjects, n_periods, n_treatments, treatment_effect, period_effect,
          carryover_effect, treatment_means, period_means, variance_components, model_summary } = crossoverData

  return (
    <div className="space-y-6">
      {/* Design Info */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="w-6 h-6 text-purple-400" />
          <h3 className="text-xl font-bold text-gray-100">{design_type} Results</h3>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Subjects</div>
            <div className="text-2xl font-bold text-purple-400">{n_subjects}</div>
          </div>
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Periods</div>
            <div className="text-2xl font-bold text-cyan-400">{n_periods}</div>
          </div>
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Treatments</div>
            <div className="text-2xl font-bold text-emerald-400">{n_treatments}</div>
          </div>
        </div>
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
          <div className={`px-4 py-2 rounded-lg ${treatment_effect.significant ? 'bg-green-900/20 border border-green-700/50' : 'bg-orange-900/20 border border-orange-700/50'}`}>
            <p className="text-sm font-semibold">
              {treatment_effect.significant ?
                'Significant treatment differences detected' :
                'No significant treatment differences'}
            </p>
          </div>

          <div className="bg-slate-700/30 rounded-lg p-4">
            <div className="text-xs text-gray-400 mb-2">Treatment Means:</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {Object.entries(treatment_means).map(([treatment, mean]) => (
                <div key={treatment} className="bg-slate-800/50 rounded px-3 py-2">
                  <div className="text-xs text-gray-400">{treatment}</div>
                  <div className="text-lg font-bold text-emerald-400 font-mono">{mean}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Period Effect */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-2 mb-4">
          {period_effect.significant ? (
            <AlertTriangle className="w-6 h-6 text-orange-400" />
          ) : (
            <CheckCircle className="w-6 h-6 text-green-400" />
          )}
          <h4 className="font-semibold text-gray-100">Period Effect</h4>
        </div>

        <div className="space-y-3">
          <div className={`px-4 py-2 rounded-lg ${period_effect.significant ? 'bg-orange-900/20 border border-orange-700/50' : 'bg-green-900/20 border border-green-700/50'}`}>
            <p className="text-sm font-semibold">
              {period_effect.significant ?
                'Significant period effect - responses differ across time periods' :
                'No significant period effect - responses consistent across time'}
            </p>
          </div>

          <div className="bg-slate-700/30 rounded-lg p-4">
            <div className="text-xs text-gray-400 mb-2">Period Means:</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {Object.entries(period_means).map(([period, mean]) => (
                <div key={period} className="bg-slate-800/50 rounded px-3 py-2">
                  <div className="text-xs text-gray-400">Period {period}</div>
                  <div className="text-lg font-bold text-cyan-400 font-mono">{mean}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Carryover Effect (2x2 designs) */}
      {carryover_effect && !carryover_effect.error && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <div className="flex items-center gap-2 mb-4">
            {carryover_effect.significant ? (
              <AlertTriangle className="w-6 h-6 text-red-400" />
            ) : (
              <CheckCircle className="w-6 h-6 text-green-400" />
            )}
            <h4 className="font-semibold text-gray-100">Carryover Effect Test</h4>
          </div>

          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-700/30 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">t-statistic</div>
                <div className="text-xl font-bold text-blue-400 font-mono">{carryover_effect.test_statistic.toFixed(4)}</div>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">p-value</div>
                <div className={`text-xl font-bold font-mono ${carryover_effect.significant ? 'text-red-400' : 'text-green-400'}`}>
                  {carryover_effect.p_value.toFixed(4)}
                </div>
              </div>
            </div>

            <div className={`px-4 py-2 rounded-lg ${carryover_effect.significant ? 'bg-red-900/20 border border-red-700/50' : 'bg-green-900/20 border border-green-700/50'}`}>
              <p className="text-sm font-semibold">{carryover_effect.interpretation}</p>
              {carryover_effect.significant && (
                <p className="text-xs text-red-300 mt-1">
                  Warning: First treatment may be influencing response in second period. Consider longer washout period.
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Variance Components */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <h4 className="font-semibold text-gray-100 mb-4">Variance Components</h4>

        <div className="grid grid-cols-3 gap-4">
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Subject Variance</div>
            <div className="text-lg font-bold text-purple-400 font-mono">{variance_components.subject_variance}</div>
          </div>
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Residual Variance</div>
            <div className="text-lg font-bold text-orange-400 font-mono">{variance_components.residual_variance}</div>
          </div>
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Total Variance</div>
            <div className="text-lg font-bold text-cyan-400 font-mono">{variance_components.total_variance}</div>
          </div>
        </div>

        <div className="mt-4 bg-blue-900/20 border border-blue-700/30 rounded-lg p-3">
          <p className="text-xs text-blue-100">
            <strong>Subject variance</strong> quantifies between-subject variability. Crossover designs control for this by having each subject receive all treatments.
          </p>
        </div>
      </div>

      {/* Model Summary */}
      <div className="bg-gradient-to-r from-purple-900/20 to-blue-900/20 rounded-lg p-4 border border-purple-700/30">
        <h5 className="font-semibold text-purple-200 mb-2">Model Quality</h5>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Log-Likelihood:</span>
            <span className="text-purple-400 font-mono ml-2">{model_summary.log_likelihood}</span>
          </div>
          <div>
            <span className="text-gray-400">AIC:</span>
            <span className="text-purple-400 font-mono ml-2">{model_summary.aic}</span>
          </div>
          <div>
            <span className="text-gray-400">BIC:</span>
            <span className="text-purple-400 font-mono ml-2">{model_summary.bic}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default CrossoverResults
