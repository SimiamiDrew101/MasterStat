import { useState } from 'react'
import { Loader2, CheckCircle, AlertCircle, BarChart3, Info } from 'lucide-react'
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts'

const CrossValidationResults = ({ cvResults, loading, error }) => {
  const [showDetails, setShowDetails] = useState(false)

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-12 bg-slate-800/30 rounded-lg border border-slate-700">
        <Loader2 className="w-12 h-12 text-blue-500 animate-spin mb-4" />
        <p className="text-gray-300">Running K-fold Cross-Validation...</p>
        <p className="text-gray-400 text-sm mt-2">This may take a few moments</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-2">
          <AlertCircle className="w-6 h-6 text-red-400" />
          <h4 className="text-xl font-bold text-red-300">Cross-Validation Error</h4>
        </div>
        <p className="text-red-200">{error}</p>
      </div>
    )
  }

  if (!cvResults) {
    return null
  }

  const { k_folds, n_observations, fold_scores, average_metrics, overall_cv_r2, predictions_vs_actual, interpretation, recommendations } = cvResults

  // Prepare data for scatter plot
  const scatterData = predictions_vs_actual.predictions.map((pred, idx) => ({
    actual: predictions_vs_actual.actuals[idx],
    predicted: pred
  }))

  // Get min/max for reference line
  const allValues = [...predictions_vs_actual.predictions, ...predictions_vs_actual.actuals]
  const minVal = Math.min(...allValues)
  const maxVal = Math.max(...allValues)

  // Determine performance color
  const getPerformanceColor = (r2) => {
    if (r2 >= 0.9) return 'text-green-400'
    if (r2 >= 0.7) return 'text-blue-400'
    if (r2 >= 0.5) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getPerformanceBg = (r2) => {
    if (r2 >= 0.9) return 'bg-green-900/20 border-green-700/50'
    if (r2 >= 0.7) return 'bg-blue-900/20 border-blue-700/50'
    if (r2 >= 0.5) return 'bg-yellow-900/20 border-yellow-700/50'
    return 'bg-red-900/20 border-red-700/50'
  }

  return (
    <div className="space-y-6">
      {/* Header with Overall CV R² */}
      <div className={`rounded-lg p-6 border ${getPerformanceBg(overall_cv_r2)}`}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <CheckCircle className={`w-8 h-8 ${getPerformanceColor(overall_cv_r2)}`} />
            <div>
              <h3 className="text-2xl font-bold text-gray-100">Cross-Validation Complete</h3>
              <p className="text-gray-300 text-sm">
                {k_folds}-Fold Cross-Validation on {n_observations} observations
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-gray-400 text-sm">Overall CV R²</p>
            <p className={`text-4xl font-bold ${getPerformanceColor(overall_cv_r2)}`}>
              {overall_cv_r2.toFixed(4)}
            </p>
          </div>
        </div>

        {/* Interpretations */}
        {interpretation && interpretation.length > 0 && (
          <div className="mt-4 space-y-2">
            {interpretation.map((interp, idx) => (
              <div key={idx} className="flex items-start gap-2">
                <Info className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
                <p className="text-gray-200 text-sm">{interp}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
          <h4 className="text-lg font-bold text-blue-300 mb-2 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Recommendations
          </h4>
          <ul className="space-y-2">
            {recommendations.map((rec, idx) => (
              <li key={idx} className="text-gray-200 text-sm flex items-start gap-2">
                <span className="text-blue-500 mt-0.5">•</span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Average Metrics Summary */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm mb-1">Average R²</p>
          <p className={`text-3xl font-bold ${getPerformanceColor(average_metrics.r2)}`}>
            {average_metrics.r2.toFixed(4)}
          </p>
          <p className="text-gray-400 text-xs mt-1">
            ± {average_metrics.r2_std.toFixed(4)}
          </p>
        </div>
        <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm mb-1">Average RMSE</p>
          <p className="text-3xl font-bold text-orange-400">
            {average_metrics.rmse.toFixed(4)}
          </p>
          <p className="text-gray-400 text-xs mt-1">
            ± {average_metrics.rmse_std.toFixed(4)}
          </p>
        </div>
        <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm mb-1">Average MAE</p>
          <p className="text-3xl font-bold text-purple-400">
            {average_metrics.mae.toFixed(4)}
          </p>
          <p className="text-gray-400 text-xs mt-1">
            ± {average_metrics.mae_std.toFixed(4)}
          </p>
        </div>
      </div>

      {/* Predicted vs Actual Scatter Plot */}
      <div className="bg-slate-800/30 border border-slate-700 rounded-lg p-6">
        <h4 className="text-xl font-bold text-gray-100 mb-4">Predicted vs Actual Values</h4>
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis
              type="number"
              dataKey="actual"
              name="Actual"
              stroke="#9ca3af"
              label={{ value: 'Actual Values', position: 'insideBottom', offset: -10, fill: '#9ca3af' }}
            />
            <YAxis
              type="number"
              dataKey="predicted"
              name="Predicted"
              stroke="#9ca3af"
              label={{ value: 'Predicted Values', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
              labelStyle={{ color: '#e2e8f0' }}
              itemStyle={{ color: '#60a5fa' }}
            />
            <Legend wrapperStyle={{ color: '#9ca3af' }} />
            <ReferenceLine
              segment={[{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }]}
              stroke="#ef4444"
              strokeWidth={2}
              strokeDasharray="5 5"
              label={{ value: 'Perfect Fit', fill: '#ef4444', fontSize: 12 }}
            />
            <Scatter name="Cross-Validation Predictions" data={scatterData} fill="#3b82f6" />
          </ScatterChart>
        </ResponsiveContainer>
        <p className="text-gray-400 text-xs text-center mt-2">
          Points closer to the red line indicate better predictive accuracy
        </p>
      </div>

      {/* Fold-by-Fold Details (Collapsible) */}
      <div className="bg-slate-800/30 border border-slate-700 rounded-lg overflow-hidden">
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="w-full p-4 flex items-center justify-between hover:bg-slate-700/30 transition-colors"
        >
          <h4 className="text-xl font-bold text-gray-100">Fold-by-Fold Performance</h4>
          <span className="text-blue-400 text-sm">
            {showDetails ? 'Hide Details' : 'Show Details'}
          </span>
        </button>

        {showDetails && (
          <div className="p-4 border-t border-slate-700">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left py-3 px-4 text-gray-300 font-semibold">Fold</th>
                    <th className="text-right py-3 px-4 text-gray-300 font-semibold">R²</th>
                    <th className="text-right py-3 px-4 text-gray-300 font-semibold">RMSE</th>
                    <th className="text-right py-3 px-4 text-gray-300 font-semibold">MAE</th>
                    <th className="text-right py-3 px-4 text-gray-300 font-semibold">Test Size</th>
                  </tr>
                </thead>
                <tbody>
                  {fold_scores.map((fold, idx) => (
                    <tr key={idx} className="border-b border-slate-800 hover:bg-slate-700/20">
                      <td className="py-3 px-4 text-gray-200 font-medium">Fold {fold.fold}</td>
                      <td className={`text-right py-3 px-4 font-mono ${getPerformanceColor(fold.r2)}`}>
                        {fold.r2.toFixed(4)}
                      </td>
                      <td className="text-right py-3 px-4 text-orange-300 font-mono">
                        {fold.rmse.toFixed(4)}
                      </td>
                      <td className="text-right py-3 px-4 text-purple-300 font-mono">
                        {fold.mae.toFixed(4)}
                      </td>
                      <td className="text-right py-3 px-4 text-gray-400 font-mono">
                        {fold.n_test}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Statistics Summary */}
            <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
              <div className="bg-slate-700/30 rounded p-3">
                <p className="text-gray-400 mb-1">R² Range</p>
                <p className="text-gray-200 font-mono">
                  {Math.min(...fold_scores.map(f => f.r2)).toFixed(4)} - {Math.max(...fold_scores.map(f => f.r2)).toFixed(4)}
                </p>
              </div>
              <div className="bg-slate-700/30 rounded p-3">
                <p className="text-gray-400 mb-1">RMSE Range</p>
                <p className="text-gray-200 font-mono">
                  {Math.min(...fold_scores.map(f => f.rmse)).toFixed(4)} - {Math.max(...fold_scores.map(f => f.rmse)).toFixed(4)}
                </p>
              </div>
              <div className="bg-slate-700/30 rounded p-3">
                <p className="text-gray-400 mb-1">MAE Range</p>
                <p className="text-gray-200 font-mono">
                  {Math.min(...fold_scores.map(f => f.mae)).toFixed(4)} - {Math.max(...fold_scores.map(f => f.mae)).toFixed(4)}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Methodology Note */}
      <div className="bg-slate-800/20 border border-slate-700/50 rounded-lg p-4">
        <h5 className="text-sm font-semibold text-gray-300 mb-2">About K-Fold Cross-Validation</h5>
        <p className="text-gray-400 text-xs leading-relaxed">
          K-fold cross-validation splits your data into {k_folds} equal parts (folds). The model is trained on {k_folds - 1} folds
          and tested on the remaining fold, repeating this process {k_folds} times. This provides a more robust estimate of model
          performance than a single train-test split, helping identify overfitting and assess generalization capability.
        </p>
      </div>
    </div>
  )
}

export default CrossValidationResults
