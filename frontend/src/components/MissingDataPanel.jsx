import { AlertTriangle, Check, Info, Database } from 'lucide-react'

/**
 * MissingDataPanel component displays missing data analysis results
 * Shows pattern statistics, MCAR test results, and imputation information
 */
const MissingDataPanel = ({ missingData }) => {
  if (!missingData || !missingData.pattern || !missingData.pattern.has_missing) {
    return null  // No missing data to display
  }

  const { pattern, mcar_test, imputation, method_used } = missingData

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50 space-y-6">
      <div className="flex items-center gap-2 mb-4">
        <Database className="w-6 h-6 text-orange-400" />
        <h3 className="text-xl font-bold text-gray-100">Missing Data Analysis</h3>
      </div>

      {/* Missing Data Pattern */}
      <div className="bg-orange-900/20 border border-orange-700/50 rounded-lg p-5">
        <div className="flex items-center gap-2 mb-3">
          <AlertTriangle className="w-5 h-5 text-orange-400" />
          <h4 className="font-semibold text-gray-100">Missing Data Pattern</h4>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Total Observations</div>
            <div className="text-2xl font-bold text-cyan-400">{pattern.n_total}</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Complete Cases</div>
            <div className="text-2xl font-bold text-green-400">{pattern.n_complete}</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">Missing Values</div>
            <div className="text-2xl font-bold text-orange-400">{pattern.n_missing}</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="text-xs text-gray-400 mb-1">% Missing</div>
            <div className="text-2xl font-bold text-red-400">{pattern.percent_missing.toFixed(1)}%</div>
          </div>
        </div>

        {pattern.percent_missing > 30 && (
          <div className="mt-4 bg-red-900/20 border border-red-700/50 rounded-lg p-3">
            <p className="text-sm text-red-200">
              <strong>Warning:</strong> High proportion of missing data ({pattern.percent_missing.toFixed(1)}%).
              Results should be interpreted with caution. Consider investigating the source of missingness.
            </p>
          </div>
        )}
      </div>

      {/* Little's MCAR Test */}
      {mcar_test && mcar_test.conclusion && (
        <div className="bg-slate-700/30 rounded-lg p-5">
          <div className="flex items-center gap-2 mb-3">
            <Info className="w-5 h-5 text-blue-400" />
            <h4 className="font-semibold text-gray-100">Little's MCAR Test</h4>
          </div>

          <div className="space-y-3">
            {mcar_test.chi2_statistic !== null && mcar_test.p_value !== null ? (
              <>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400 mb-1">Chi-Square Statistic</div>
                    <div className="text-xl font-bold text-blue-400 font-mono">{mcar_test.chi2_statistic.toFixed(4)}</div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400 mb-1">p-value</div>
                    <div className={`text-xl font-bold font-mono ${
                      mcar_test.p_value > 0.05 ? 'text-green-400' : 'text-orange-400'
                    }`}>
                      {mcar_test.p_value.toFixed(4)}
                    </div>
                  </div>
                </div>

                <div className="bg-slate-800/50 rounded-lg p-3">
                  <div className="flex items-center gap-2">
                    {mcar_test.p_value > 0.05 ? (
                      <Check className="w-5 h-5 text-green-400 flex-shrink-0" />
                    ) : (
                      <AlertTriangle className="w-5 h-5 text-orange-400 flex-shrink-0" />
                    )}
                    <div>
                      <div className="text-sm font-semibold text-gray-200 mb-1">Conclusion:</div>
                      <div className="text-sm text-gray-300">{mcar_test.conclusion}</div>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="text-sm text-gray-400">
                {mcar_test.conclusion}
              </div>
            )}

            <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-3">
              <p className="text-xs text-blue-100">
                <strong>MCAR (Missing Completely At Random):</strong> Data is MCAR if missingness is completely random
                and unrelated to any variables. If p {'>'} 0.05, we fail to reject MCAR assumption, suggesting simple
                imputation methods may be appropriate.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Imputation Results */}
      {imputation && method_used !== 'none' && (
        <div className="bg-slate-700/30 rounded-lg p-5">
          <div className="flex items-center gap-2 mb-3">
            <Check className="w-5 h-5 text-green-400" />
            <h4 className="font-semibold text-gray-100">Imputation Results</h4>
          </div>

          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-800/50 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">Method</div>
                <div className="text-lg font-bold text-emerald-400 uppercase">{imputation.method.replace('_', ' ')}</div>
              </div>
              <div className="bg-slate-800/50 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">Values Imputed</div>
                <div className="text-lg font-bold text-emerald-400">{imputation.n_imputed}</div>
              </div>
            </div>

            {imputation.method === 'mean_imputation' && imputation.imputed_values && (
              <div className="bg-slate-800/50 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-2">
                  {imputation.group_based ? 'Group-Specific Imputation Values:' : 'Overall Imputation Value:'}
                </div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(imputation.imputed_values).map(([group, value]) => (
                    <span key={group} className="px-2 py-1 bg-emerald-900/30 border border-emerald-700/50 rounded text-sm text-emerald-300 font-mono">
                      {group}: {value.toFixed(3)}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {imputation.method === 'em_imputation' && (
              <div className="bg-slate-800/50 rounded-lg p-3">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-gray-400">Iterations:</span>
                    <span className="text-gray-200 ml-2 font-mono">{imputation.iterations}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Converged:</span>
                    <span className={`ml-2 font-semibold ${imputation.converged ? 'text-green-400' : 'text-orange-400'}`}>
                      {imputation.converged ? 'Yes' : 'No'}
                    </span>
                  </div>
                </div>
                {imputation.fallback && (
                  <div className="mt-2 text-xs text-orange-300">
                    Note: EM algorithm failed, fell back to {imputation.fallback}
                  </div>
                )}
              </div>
            )}

            <div className="bg-green-900/20 border border-green-700/30 rounded-lg p-3">
              <p className="text-xs text-green-100">
                <strong>Imputation Note:</strong> The analysis results below are based on the imputed dataset.
                Imputed values have been estimated using {imputation.method === 'mean_imputation' ? 'group-specific means' : 'EM algorithm with treatment and block predictors'}.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* No Imputation Warning */}
      {method_used === 'none' && pattern.has_missing && (
        <div className="bg-orange-900/20 border border-orange-700/50 rounded-lg p-4">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-5 h-5 text-orange-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm font-semibold text-orange-200 mb-1">No Imputation Applied</p>
              <p className="text-sm text-orange-300">
                Missing data detected but no imputation method was selected. Analysis proceeds with complete cases only,
                which may reduce power and introduce bias if data is not MCAR.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default MissingDataPanel
