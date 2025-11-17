import { AlertTriangle, TrendingUp } from 'lucide-react'

const InfluenceDiagnostics = ({ influenceData }) => {
  if (!influenceData) return null

  const { cooks_distance, leverage, dffits, influential_indices, thresholds, n_influential } = influenceData

  // Find top 10 most influential observations by Cook's D
  const cooksSorted = cooks_distance
    .map((value, index) => ({ value, index: index + 1 }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 10)

  const hasInfluential = n_influential > 0

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center space-x-3 mb-4">
        <TrendingUp className="w-6 h-6 text-orange-400" />
        <h3 className="text-xl font-bold text-gray-100">Influence Diagnostics</h3>
      </div>

      <p className="text-gray-400 text-sm mb-6">
        Identifies observations that have disproportionate influence on the model. High influence points may warrant investigation.
      </p>

      {/* Summary */}
      <div className={`rounded-lg border p-4 mb-6 ${
        hasInfluential
          ? 'bg-yellow-500/10 border-yellow-500/50'
          : 'bg-green-500/10 border-green-500/50'
      }`}>
        <div className="flex items-center space-x-2 mb-2">
          {hasInfluential ? (
            <AlertTriangle className="w-5 h-5 text-yellow-400" />
          ) : (
            <div className="w-5 h-5 rounded-full bg-green-500" />
          )}
          <h4 className="font-semibold text-gray-100">
            {hasInfluential
              ? `${n_influential} Influential Observation${n_influential > 1 ? 's' : ''} Detected`
              : 'No Influential Observations Detected'}
          </h4>
        </div>
        <p className="text-sm text-gray-300">
          {hasInfluential
            ? `Observation${influential_indices.length > 1 ? 's' : ''} #${influential_indices.slice(0, 5).map(i => i + 1).join(', ')}${influential_indices.length > 5 ? '...' : ''} exceed${influential_indices.length === 1 ? 's' : ''} influence thresholds.`
            : 'All observations have acceptable influence on the model.'}
        </p>
      </div>

      {/* Top Influential Observations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Cook's Distance */}
        <div className="bg-slate-700/30 rounded-lg p-4">
          <h5 className="font-semibold text-gray-200 mb-3">Cook's Distance (Top 10)</h5>
          <p className="text-xs text-gray-400 mb-3">
            Threshold: {thresholds.cooks_d} (values above indicate influence)
          </p>
          <div className="space-y-2">
            {cooksSorted.map(({ value, index }, i) => (
              <div key={i} className="flex items-center justify-between">
                <span className="text-sm text-gray-300">Observation #{index}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-32 bg-slate-600 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${value > thresholds.cooks_d ? 'bg-red-500' : 'bg-blue-500'}`}
                      style={{ width: `${Math.min(100, (value / (thresholds.cooks_d * 2)) * 100)}%` }}
                    />
                  </div>
                  <span className={`text-xs font-mono ${value > thresholds.cooks_d ? 'text-red-400' : 'text-gray-400'}`}>
                    {value.toFixed(4)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* DFFITS */}
        <div className="bg-slate-700/30 rounded-lg p-4">
          <h5 className="font-semibold text-gray-200 mb-3">DFFITS (Top 10)</h5>
          <p className="text-xs text-gray-400 mb-3">
            Threshold: ±{thresholds.dffits} (absolute values above indicate influence)
          </p>
          <div className="space-y-2">
            {dffits
              .map((value, index) => ({ value, index: index + 1, absValue: Math.abs(value) }))
              .sort((a, b) => b.absValue - a.absValue)
              .slice(0, 10)
              .map(({ value, index, absValue }, i) => (
                <div key={i} className="flex items-center justify-between">
                  <span className="text-sm text-gray-300">Observation #{index}</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-32 bg-slate-600 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${absValue > thresholds.dffits ? 'bg-red-500' : 'bg-purple-500'}`}
                        style={{ width: `${Math.min(100, (absValue / (thresholds.dffits * 2)) * 100)}%` }}
                      />
                    </div>
                    <span className={`text-xs font-mono ${absValue > thresholds.dffits ? 'text-red-400' : 'text-gray-400'}`}>
                      {value.toFixed(4)}
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* Interpretation Guide */}
      <div className="bg-slate-700/30 rounded-lg p-4">
        <h5 className="font-semibold text-gray-200 mb-3">Interpretation</h5>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-gray-400">
          <div>
            <h6 className="font-semibold text-gray-300 mb-2">Cook's Distance:</h6>
            <p className="mb-2">
              Measures overall influence of each observation on the fitted model.
              Values {'>'} {thresholds.cooks_d.toFixed(4)} suggest potential influence.
            </p>
            <p className="text-yellow-200">
              <strong>Action:</strong> Investigate high values for data entry errors or genuine outliers.
            </p>
          </div>
          <div>
            <h6 className="font-semibold text-gray-300 mb-2">DFFITS:</h6>
            <p className="mb-2">
              Measures change in fitted value when observation is deleted.
              |Values| {'>'} {thresholds.dffits.toFixed(4)} indicate substantial influence.
            </p>
            <p className="text-yellow-200">
              <strong>Action:</strong> Consider sensitivity analysis with/without influential points.
            </p>
          </div>
        </div>
      </div>

      {hasInfluential && (
        <div className="mt-4 bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-3">
          <p className="text-yellow-200 text-sm">
            <strong>Recommendation:</strong> Influential observations don't necessarily mean bad data.
            Verify data accuracy, check for recording errors, and consider reporting results both with and without these observations.
          </p>
        </div>
      )}
    </div>
  )
}

export default InfluenceDiagnostics
