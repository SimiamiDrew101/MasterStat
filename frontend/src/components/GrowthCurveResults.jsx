import { TrendingUp, Info, CheckCircle2, AlertCircle } from 'lucide-react'

const GrowthCurveResults = ({ result }) => {
  if (!result) return null

  const {
    polynomial_order,
    random_effects_structure,
    n_subjects,
    n_observations,
    n_timepoints,
    fixed_effects,
    random_effects_variance,
    icc,
    model_summary,
    interpretation
  } = result

  return (
    <div className="space-y-6">
      {/* Model Summary Card */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <TrendingUp className="w-6 h-6 text-emerald-400" />
          <h3 className="text-xl font-bold text-gray-100">Growth Curve Model Summary</h3>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Polynomial Order</div>
            <div className="text-lg font-bold text-emerald-300 capitalize">{polynomial_order}</div>
          </div>
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Random Effects</div>
            <div className="text-sm font-bold text-emerald-300">
              {random_effects_structure === 'intercept_slope' ? 'Intercept + Slope' : 'Intercept Only'}
            </div>
          </div>
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Subjects</div>
            <div className="text-lg font-bold text-gray-200">{n_subjects}</div>
          </div>
          <div className="bg-slate-700/30 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Observations</div>
            <div className="text-lg font-bold text-gray-200">{n_observations}</div>
          </div>
        </div>

        {/* Interpretation */}
        {interpretation && (
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 flex items-start space-x-3">
            <Info className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
            <p className="text-sm text-blue-100 leading-relaxed">{interpretation}</p>
          </div>
        )}
      </div>

      {/* Fixed Effects Table */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <h4 className="text-lg font-bold text-gray-100 mb-4">Fixed Effects (Time Trends)</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-3 px-4 text-gray-300 font-semibold">Parameter</th>
                <th className="text-right py-3 px-4 text-gray-300 font-semibold">Estimate</th>
                <th className="text-right py-3 px-4 text-gray-300 font-semibold">Std Error</th>
                <th className="text-right py-3 px-4 text-gray-300 font-semibold">t-value</th>
                <th className="text-right py-3 px-4 text-gray-300 font-semibold">p-value</th>
                <th className="text-right py-3 px-4 text-gray-300 font-semibold">95% CI</th>
                <th className="text-center py-3 px-4 text-gray-300 font-semibold">Sig</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(fixed_effects.coefficients).map((param) => {
                const coef = fixed_effects.coefficients[param]
                const se = fixed_effects.std_errors[param]
                const tval = fixed_effects.t_values[param]
                const pval = fixed_effects.p_values[param]
                const ci = fixed_effects.conf_int[param]
                const isSig = pval < 0.05

                return (
                  <tr key={param} className="border-b border-slate-700/50 hover:bg-slate-700/20">
                    <td className="py-3 px-4 text-gray-200 font-medium">{param}</td>
                    <td className="py-3 px-4 text-right font-mono text-gray-300">{coef.toFixed(4)}</td>
                    <td className="py-3 px-4 text-right font-mono text-gray-400">{se.toFixed(4)}</td>
                    <td className="py-3 px-4 text-right font-mono text-gray-300">{tval.toFixed(3)}</td>
                    <td className={`py-3 px-4 text-right font-mono ${isSig ? 'text-green-400 font-bold' : 'text-gray-400'}`}>
                      {pval < 0.0001 ? '<0.0001' : pval.toFixed(4)}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-gray-400 text-xs">
                      [{ci[0].toFixed(3)}, {ci[1].toFixed(3)}]
                    </td>
                    <td className="py-3 px-4 text-center">
                      {isSig ? (
                        <CheckCircle2 className="w-4 h-4 text-green-400 inline" />
                      ) : (
                        <div className="w-4 h-4 inline-block" />
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Random Effects Variance Components */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <h4 className="text-lg font-bold text-gray-100 mb-4">Random Effects Variance Components</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-700/30 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-2">Intercept Variance</div>
            <div className="text-2xl font-mono font-bold text-purple-300">
              {random_effects_variance.intercept_var?.toFixed(4) || 'N/A'}
            </div>
            <div className="text-xs text-gray-500 mt-1">Between-subject variability in initial status</div>
          </div>

          {random_effects_variance.slope_var !== undefined && (
            <div className="bg-slate-700/30 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-2">Slope Variance</div>
              <div className="text-2xl font-mono font-bold text-blue-300">
                {random_effects_variance.slope_var.toFixed(4)}
              </div>
              <div className="text-xs text-gray-500 mt-1">Between-subject variability in growth rates</div>
            </div>
          )}

          {random_effects_variance.intercept_slope_corr !== undefined && (
            <div className="bg-slate-700/30 rounded-lg p-4">
              <div className="text-sm text-gray-400 mb-2">Intercept-Slope Correlation</div>
              <div className="text-2xl font-mono font-bold text-cyan-300">
                {random_effects_variance.intercept_slope_corr.toFixed(4)}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {Math.abs(random_effects_variance.intercept_slope_corr) > 0.5
                  ? random_effects_variance.intercept_slope_corr > 0
                    ? 'Higher starters grow faster'
                    : 'Higher starters grow slower'
                  : 'Weak correlation'}
              </div>
            </div>
          )}

          <div className="bg-slate-700/30 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-2">Residual Variance</div>
            <div className="text-2xl font-mono font-bold text-gray-300">
              {random_effects_variance.residual_var?.toFixed(4) || 'N/A'}
            </div>
            <div className="text-xs text-gray-500 mt-1">Within-subject variability (measurement error)</div>
          </div>

          <div className="bg-slate-700/30 rounded-lg p-4">
            <div className="text-sm text-gray-400 mb-2">Intraclass Correlation (ICC)</div>
            <div className="text-2xl font-mono font-bold text-orange-300">
              {icc?.toFixed(4) || 'N/A'}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {icc > 0.7 ? 'High' : icc > 0.4 ? 'Moderate' : 'Low'} between-subject variability
            </div>
          </div>
        </div>

        {/* ICC Interpretation */}
        <div className="mt-4 bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
          <div className="flex items-start space-x-2">
            <Info className="w-4 h-4 text-purple-400 mt-0.5 flex-shrink-0" />
            <p className="text-xs text-purple-100">
              <strong>ICC = {(icc * 100).toFixed(1)}%</strong> of the total variance is due to between-subject differences.
              {icc > 0.5
                ? ' Subjects differ substantially in their trajectories.'
                : ' Most variability is within-subject over time.'}
            </p>
          </div>
        </div>
      </div>

      {/* Model Fit Statistics */}
      {model_summary && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <h4 className="text-lg font-bold text-gray-100 mb-4">Model Fit Statistics</h4>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-slate-700/30 rounded-lg p-3 text-center">
              <div className="text-xs text-gray-400 mb-1">Log-Likelihood</div>
              <div className="text-lg font-mono text-gray-200">{model_summary.log_likelihood?.toFixed(2)}</div>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-3 text-center">
              <div className="text-xs text-gray-400 mb-1">AIC</div>
              <div className="text-lg font-mono text-gray-200">{model_summary.aic?.toFixed(2)}</div>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-3 text-center">
              <div className="text-xs text-gray-400 mb-1">BIC</div>
              <div className="text-lg font-mono text-gray-200">{model_summary.bic?.toFixed(2)}</div>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-3 text-center">
            Lower AIC/BIC values indicate better model fit. Use these to compare models with different polynomial orders.
          </p>
        </div>
      )}
    </div>
  )
}

export default GrowthCurveResults
