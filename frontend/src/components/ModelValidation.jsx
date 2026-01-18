import Plot from 'react-plotly.js'
import { getPlotlyConfig } from '../utils/plotlyConfig'
import { CheckCircle, AlertTriangle, XCircle, TrendingUp, Award } from 'lucide-react'

/**
 * ModelValidation component for displaying comprehensive model validation results
 * Shows validation metrics, cross-validation results, adequacy tests, and diagnostics
 * Used across ANOVA, Factorial, Mixed Models, and Nonlinear Regression modules
 */
const ModelValidation = ({ validationData }) => {
  if (!validationData) {
    return (
      <div className="bg-slate-700/50 rounded-lg p-6">
        <p className="text-gray-400 text-center">No validation data available</p>
      </div>
    )
  }

  const {
    model_info,
    metrics,
    press,
    cross_validation,
    adequacy,
    residuals
  } = validationData

  // Helper to get status icon
  const getStatusIcon = (passed) => {
    if (passed === true) return <CheckCircle className="w-5 h-5 text-green-400" />
    if (passed === false) return <XCircle className="w-5 h-5 text-red-400" />
    return <AlertTriangle className="w-5 h-5 text-yellow-400" />
  }

  // Helper to get score color
  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-400'
    if (score >= 60) return 'text-blue-400'
    if (score >= 40) return 'text-yellow-400'
    return 'text-red-400'
  }

  // Helper to get metric quality color
  const getMetricColor = (value, thresholds = { good: 0.8, moderate: 0.6 }) => {
    if (value >= thresholds.good) return 'text-green-400'
    if (value >= thresholds.moderate) return 'text-yellow-400'
    return 'text-red-400'
  }

  return (
    <div className="space-y-6">
      {/* Header with Adequacy Score */}
      <div className="bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800 rounded-lg p-6 border border-slate-600">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-2xl font-bold text-gray-100 mb-2">Model Validation Report</h3>
            {model_info && (
              <p className="text-gray-400 text-sm">
                {model_info.formula || model_info.model_name} • {model_info.n_observations} observations
              </p>
            )}
          </div>
          {adequacy && (
            <div className="text-center">
              <div className="flex items-center justify-center gap-2 mb-1">
                <Award className="w-6 h-6 text-yellow-400" />
                <span className={`text-4xl font-bold ${getScoreColor(adequacy.adequacy_score)}`}>
                  {adequacy.adequacy_score}
                </span>
                <span className="text-gray-400 text-lg">/100</span>
              </div>
              <p className="text-sm text-gray-400">Adequacy Score</p>
            </div>
          )}
        </div>
        {adequacy && (
          <div className="mt-4 p-3 bg-blue-900/20 border border-blue-700/50 rounded">
            <p className="text-blue-200 text-sm">{adequacy.overall_assessment}</p>
          </div>
        )}
      </div>

      {/* Validation Metrics */}
      {metrics && (
        <div className="bg-slate-700/50 rounded-lg p-6">
          <h4 className="text-gray-100 font-semibold text-lg mb-4">Validation Metrics</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {metrics.r2 !== undefined && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm mb-1">R²</p>
                <p className={`text-2xl font-bold ${getMetricColor(metrics.r2)}`}>
                  {metrics.r2.toFixed(4)}
                </p>
              </div>
            )}
            {metrics.r2_adjusted !== undefined && metrics.r2_adjusted !== null && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm mb-1">R² Adjusted</p>
                <p className={`text-2xl font-bold ${getMetricColor(metrics.r2_adjusted)}`}>
                  {metrics.r2_adjusted.toFixed(4)}
                </p>
              </div>
            )}
            {metrics.r2_prediction !== undefined && metrics.r2_prediction !== null && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm mb-1">R² Prediction</p>
                <p className={`text-2xl font-bold ${getMetricColor(metrics.r2_prediction)}`}>
                  {metrics.r2_prediction.toFixed(4)}
                </p>
              </div>
            )}
            {metrics.rmse !== undefined && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm mb-1">RMSE</p>
                <p className="text-2xl font-bold text-gray-200">{metrics.rmse.toFixed(4)}</p>
              </div>
            )}
            {metrics.mae !== undefined && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm mb-1">MAE</p>
                <p className="text-2xl font-bold text-gray-200">{metrics.mae.toFixed(4)}</p>
              </div>
            )}
            {metrics.aic !== undefined && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm mb-1">AIC</p>
                <p className="text-2xl font-bold text-gray-200">{metrics.aic.toFixed(2)}</p>
              </div>
            )}
            {metrics.bic !== undefined && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm mb-1">BIC</p>
                <p className="text-2xl font-bold text-gray-200">{metrics.bic.toFixed(2)}</p>
              </div>
            )}
            {metrics.mape !== undefined && metrics.mape !== null && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm mb-1">MAPE</p>
                <p className="text-2xl font-bold text-gray-200">{metrics.mape.toFixed(2)}%</p>
              </div>
            )}
          </div>
          {metrics.interpretation && (
            <div className="mt-4 p-3 bg-slate-800/50 rounded text-sm text-gray-300">
              <p><strong>R² Quality:</strong> {metrics.interpretation.r2_quality}</p>
              {metrics.interpretation.prediction_quality && (
                <p><strong>Prediction Quality:</strong> {metrics.interpretation.prediction_quality}</p>
              )}
              {metrics.interpretation.note && (
                <p className="text-gray-400 mt-2">{metrics.interpretation.note}</p>
              )}
            </div>
          )}
        </div>
      )}

      {/* PRESS Statistic */}
      {press && (
        <div className="bg-slate-700/50 rounded-lg p-6">
          <h4 className="text-gray-100 font-semibold text-lg mb-4">PRESS Statistic</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-800/50 rounded-lg p-4">
              <p className="text-gray-400 text-sm mb-1">PRESS</p>
              <p className="text-2xl font-bold text-gray-200">{press.press?.toFixed(4) || 'N/A'}</p>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-4">
              <p className="text-gray-400 text-sm mb-1">R² Prediction</p>
              <p className={`text-2xl font-bold ${getMetricColor(press.r2_prediction || 0)}`}>
                {press.r2_prediction?.toFixed(4) || 'N/A'}
              </p>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-4">
              <p className="text-gray-400 text-sm mb-1">Interpretation</p>
              <p className="text-sm text-gray-300">{press.interpretation || 'N/A'}</p>
            </div>
          </div>
          {press.recommendation && (
            <div className="mt-4 p-3 bg-blue-900/20 border border-blue-700/50 rounded text-sm text-blue-200">
              {press.recommendation}
            </div>
          )}
        </div>
      )}

      {/* Cross-Validation Results */}
      {cross_validation && cross_validation.success && (
        <div className="bg-slate-700/50 rounded-lg p-6">
          <h4 className="text-gray-100 font-semibold text-lg mb-4">
            {cross_validation.k_folds}-Fold Cross-Validation
          </h4>

          {/* Summary Metrics */}
          {cross_validation.summary && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-slate-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm mb-1">Avg R²</p>
                <p className={`text-2xl font-bold ${getMetricColor(cross_validation.summary.avg_r2 || 0)}`}>
                  {cross_validation.summary.avg_r2?.toFixed(4) || 'N/A'}
                </p>
                <p className="text-xs text-gray-500 mt-1">± {cross_validation.summary.std_r2?.toFixed(4) || 'N/A'}</p>
              </div>
              <div className="bg-slate-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm mb-1">Avg RMSE</p>
                <p className="text-2xl font-bold text-gray-200">
                  {cross_validation.summary.avg_rmse?.toFixed(4) || 'N/A'}
                </p>
                <p className="text-xs text-gray-500 mt-1">± {cross_validation.summary.std_rmse?.toFixed(4) || 'N/A'}</p>
              </div>
              <div className="bg-slate-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm mb-1">Avg MAE</p>
                <p className="text-2xl font-bold text-gray-200">
                  {cross_validation.summary.avg_mae?.toFixed(4) || 'N/A'}
                </p>
                <p className="text-xs text-gray-500 mt-1">± {cross_validation.summary.std_mae?.toFixed(4) || 'N/A'}</p>
              </div>
              <div className="bg-slate-800/50 rounded-lg p-4">
                <p className="text-gray-400 text-sm mb-1">Overall R²</p>
                <p className={`text-2xl font-bold ${getMetricColor(cross_validation.summary.overall_r2 || 0)}`}>
                  {cross_validation.summary.overall_r2?.toFixed(4) || 'N/A'}
                </p>
              </div>
            </div>
          )}

          {/* Fold-by-Fold Results */}
          {cross_validation.fold_results && cross_validation.fold_results.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-800 text-gray-300">
                  <tr>
                    <th className="px-4 py-2 text-left">Fold</th>
                    <th className="px-4 py-2 text-right">R²</th>
                    <th className="px-4 py-2 text-right">RMSE</th>
                    <th className="px-4 py-2 text-right">MAE</th>
                    <th className="px-4 py-2 text-right">Test Size</th>
                  </tr>
                </thead>
                <tbody className="text-gray-300">
                  {cross_validation.fold_results.map((fold, idx) => (
                    <tr key={idx} className="border-b border-slate-600">
                      <td className="px-4 py-2">Fold {fold.fold}</td>
                      <td className="px-4 py-2 text-right">{fold.r2?.toFixed(4) || 'N/A'}</td>
                      <td className="px-4 py-2 text-right">{fold.rmse?.toFixed(4) || 'N/A'}</td>
                      <td className="px-4 py-2 text-right">{fold.mae?.toFixed(4) || 'N/A'}</td>
                      <td className="px-4 py-2 text-right">{fold.n_test || 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Predicted vs Actual Plot */}
          {cross_validation.predictions && (
            <div className="mt-6">
              <PredictedVsActualPlot predictions={cross_validation.predictions} />
            </div>
          )}

          {cross_validation.interpretation && (
            <div className="mt-4 p-3 bg-blue-900/20 border border-blue-700/50 rounded text-sm text-blue-200">
              {cross_validation.interpretation}
            </div>
          )}
        </div>
      )}

      {/* Model Adequacy Tests */}
      {adequacy && adequacy.tests && (
        <div className="bg-slate-700/50 rounded-lg p-6">
          <h4 className="text-gray-100 font-semibold text-lg mb-4">Model Adequacy Tests</h4>

          <div className="space-y-4">
            {/* Normality Test */}
            {adequacy.tests.normality && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h5 className="text-gray-200 font-medium">Normality (Shapiro-Wilk)</h5>
                  {getStatusIcon(adequacy.tests.normality.pass)}
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-400">Statistic:</p>
                    <p className="text-gray-200">{adequacy.tests.normality.statistic}</p>
                  </div>
                  <div>
                    <p className="text-gray-400">P-value:</p>
                    <p className="text-gray-200">{adequacy.tests.normality.p_value}</p>
                  </div>
                </div>
                <p className="text-sm text-gray-300 mt-2">{adequacy.tests.normality.interpretation}</p>
              </div>
            )}

            {/* Homoscedasticity Test */}
            {adequacy.tests.homoscedasticity && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h5 className="text-gray-200 font-medium">Homoscedasticity (Breusch-Pagan)</h5>
                  {getStatusIcon(adequacy.tests.homoscedasticity.pass)}
                </div>
                {adequacy.tests.homoscedasticity.statistic !== null && (
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-400">Statistic:</p>
                      <p className="text-gray-200">{adequacy.tests.homoscedasticity.statistic}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">P-value:</p>
                      <p className="text-gray-200">{adequacy.tests.homoscedasticity.p_value}</p>
                    </div>
                  </div>
                )}
                <p className="text-sm text-gray-300 mt-2">{adequacy.tests.homoscedasticity.interpretation}</p>
              </div>
            )}

            {/* Autocorrelation Test */}
            {adequacy.tests.autocorrelation && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h5 className="text-gray-200 font-medium">Autocorrelation (Durbin-Watson)</h5>
                  {getStatusIcon(adequacy.tests.autocorrelation.pass)}
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-400">Statistic:</p>
                    <p className="text-gray-200">{adequacy.tests.autocorrelation.statistic}</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Expected Range:</p>
                    <p className="text-gray-200">1.5 - 2.5</p>
                  </div>
                </div>
                <p className="text-sm text-gray-300 mt-2">{adequacy.tests.autocorrelation.interpretation}</p>
              </div>
            )}

            {/* Randomness Test (for nonlinear) */}
            {adequacy.tests.randomness && (
              <div className="bg-slate-800/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h5 className="text-gray-200 font-medium">Randomness (Runs Test)</h5>
                  {getStatusIcon(adequacy.tests.randomness.pass)}
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-400">Runs:</p>
                    <p className="text-gray-200">{adequacy.tests.randomness.runs}</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Expected:</p>
                    <p className="text-gray-200">{adequacy.tests.randomness.expected_runs}</p>
                  </div>
                </div>
                <p className="text-sm text-gray-300 mt-2">{adequacy.tests.randomness.interpretation}</p>
              </div>
            )}
          </div>

          {/* Diagnostics Summary */}
          {adequacy.diagnostics && (
            <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
              <h5 className="text-gray-200 font-medium mb-2">Diagnostic Summary</h5>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                {adequacy.diagnostics.n_outliers !== undefined && (
                  <div>
                    <p className="text-gray-400">Outliers:</p>
                    <p className="text-gray-200">{adequacy.diagnostics.n_outliers}</p>
                  </div>
                )}
                {adequacy.diagnostics.n_high_leverage !== undefined && (
                  <div>
                    <p className="text-gray-400">High Leverage:</p>
                    <p className="text-gray-200">{adequacy.diagnostics.n_high_leverage}</p>
                  </div>
                )}
                {adequacy.diagnostics.n_influential !== undefined && (
                  <div>
                    <p className="text-gray-400">Influential Points:</p>
                    <p className="text-gray-200">{adequacy.diagnostics.n_influential}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Recommendations */}
          {adequacy.recommendations && adequacy.recommendations.length > 0 && (
            <div className="mt-4 bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-4">
              <h5 className="text-yellow-200 font-medium mb-2">Recommendations</h5>
              <ul className="space-y-1 text-sm text-yellow-100">
                {adequacy.recommendations.map((rec, idx) => (
                  <li key={idx}>• {rec}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Helper component for Predicted vs Actual plot
const PredictedVsActualPlot = ({ predictions }) => {
  if (!predictions || !predictions.actual || !predictions.predicted) return null

  const { actual, predicted } = predictions

  // Calculate perfect fit line
  const minVal = Math.min(...actual, ...predicted)
  const maxVal = Math.max(...actual, ...predicted)

  const scatterTrace = {
    type: 'scatter',
    mode: 'markers',
    x: actual,
    y: predicted,
    marker: {
      size: 8,
      color: '#3b82f6',
      line: {
        color: '#f1f5f9',
        width: 1
      }
    },
    name: 'Predictions',
    hovertemplate: 'Actual: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>'
  }

  const perfectFitTrace = {
    type: 'scatter',
    mode: 'lines',
    x: [minVal, maxVal],
    y: [minVal, maxVal],
    line: {
      color: '#ef4444',
      width: 2,
      dash: 'dash'
    },
    name: 'Perfect Fit',
    hoverinfo: 'skip'
  }

  const layout = {
    title: {
      text: 'Predicted vs Actual Values',
      font: {
        size: 16,
        color: '#f1f5f9'
      }
    },
    xaxis: {
      title: 'Actual Values',
      gridcolor: '#475569',
      zerolinecolor: '#64748b',
      color: '#e2e8f0'
    },
    yaxis: {
      title: 'Predicted Values',
      gridcolor: '#475569',
      zerolinecolor: '#64748b',
      color: '#e2e8f0'
    },
    paper_bgcolor: '#334155',
    plot_bgcolor: '#1e293b',
    font: {
      color: '#e2e8f0'
    },
    margin: {
      l: 60,
      r: 60,
      b: 60,
      t: 60
    },
    showlegend: true,
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(51, 65, 85, 0.8)',
      bordercolor: '#475569',
      borderwidth: 1
    }
  }

  const config = getPlotlyConfig('cv-predicted-vs-actual', {
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
  })

  return (
    <div className="bg-slate-800/50 rounded-lg p-4">
      <Plot
        data={[scatterTrace, perfectFitTrace]}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '400px' }}
        useResizeHandler={true}
      />
    </div>
  )
}

export default ModelValidation
