import { TrendingUp, Info, AlertCircle, CheckCircle2 } from 'lucide-react'
import { useState } from 'react'

const ICCDisplay = ({ iccData }) => {
  const [showDetails, setShowDetails] = useState(false)

  if (!iccData || Object.keys(iccData).length === 0) return null

  // Quality color mapping
  const qualityColors = {
    poor: { bg: 'bg-red-500/20', border: 'border-red-500', text: 'text-red-400', gauge: 'text-red-500' },
    moderate: { bg: 'bg-yellow-500/20', border: 'border-yellow-500', text: 'text-yellow-400', gauge: 'text-yellow-500' },
    good: { bg: 'bg-blue-500/20', border: 'border-blue-500', text: 'text-blue-400', gauge: 'text-blue-500' },
    excellent: { bg: 'bg-green-500/20', border: 'border-green-500', text: 'text-green-400', gauge: 'text-green-500' }
  }

  const QualityIcon = ({ quality }) => {
    if (quality === 'excellent' || quality === 'good') {
      return <CheckCircle2 className="w-5 h-5 text-green-400" />
    }
    return <AlertCircle className="w-5 h-5 text-yellow-400" />
  }

  // Gauge visualization for ICC value
  const ICCGauge = ({ icc, quality, ciLower, ciUpper }) => {
    const colors = qualityColors[quality]
    const percentage = icc * 100
    const ciLowerPct = ciLower * 100
    const ciUpperPct = ciUpper * 100

    return (
      <div className="relative w-full h-32 flex items-end justify-center">
        {/* Background arc */}
        <div className="relative w-48 h-24 overflow-hidden">
          <svg className="w-full h-full" viewBox="0 0 200 100">
            {/* Background arc */}
            <path
              d="M 10 95 A 90 90 0 0 1 190 95"
              fill="none"
              stroke="currentColor"
              strokeWidth="12"
              className="text-slate-700"
            />

            {/* Colored arc based on ICC value */}
            <path
              d={`M 10 95 A 90 90 0 0 1 ${10 + percentage * 1.8} ${95 - Math.sin(Math.acos((percentage * 1.8 - 90) / 90)) * 90}`}
              fill="none"
              stroke="currentColor"
              strokeWidth="12"
              className={colors.gauge}
              strokeLinecap="round"
            />

            {/* CI range indicator */}
            {ciLower !== undefined && ciUpper !== undefined && (
              <>
                <line
                  x1={10 + ciLowerPct * 1.8}
                  y1={95 - Math.sin(Math.acos((ciLowerPct * 1.8 - 90) / 90)) * 90}
                  x2={10 + ciLowerPct * 1.8}
                  y2={100}
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-gray-400 opacity-50"
                />
                <line
                  x1={10 + ciUpperPct * 1.8}
                  y1={95 - Math.sin(Math.acos((ciUpperPct * 1.8 - 90) / 90)) * 90}
                  x2={10 + ciUpperPct * 1.8}
                  y2={100}
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-gray-400 opacity-50"
                />
              </>
            )}

            {/* Threshold markers */}
            <line x1="55" y1="95" x2="55" y2="100" stroke="white" strokeWidth="1" opacity="0.3" />
            <line x1="100" y1="10" x2="100" y2="5" stroke="white" strokeWidth="1" opacity="0.3" />
            <line x1="145" y1="95" x2="145" y2="100" stroke="white" strokeWidth="1" opacity="0.3" />

            {/* Labels */}
            <text x="10" y="110" fill="currentColor" fontSize="10" className="text-gray-400">0.0</text>
            <text x="90" y="8" fill="currentColor" fontSize="10" className="text-gray-400">0.75</text>
            <text x="180" y="110" fill="currentColor" fontSize="10" className="text-gray-400">1.0</text>
          </svg>

          {/* Center value display */}
          <div className="absolute inset-0 flex flex-col items-center justify-end pb-2">
            <div className={`text-3xl font-bold ${colors.text}`}>
              {icc.toFixed(3)}
            </div>
            <div className="text-xs text-gray-400 mt-1">
              95% CI: [{ciLower.toFixed(3)}, {ciUpper.toFixed(3)}]
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <TrendingUp className="w-6 h-6 text-purple-400" />
          <h3 className="text-xl font-bold text-gray-100">Intraclass Correlation (ICC)</h3>
        </div>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-gray-400 hover:text-gray-200 transition-colors"
        >
          <Info className="w-5 h-5" />
        </button>
      </div>

      <p className="text-gray-400 text-sm mb-6">
        ICC quantifies the proportion of variance attributable to clustering/grouping. Higher values indicate stronger within-group homogeneity.
      </p>

      {/* ICC results for each random effect */}
      <div className="space-y-6">
        {Object.entries(iccData).map(([factor, iccResult]) => {
          // Handle legacy format (nested designs) where ICC is just a number
          if (typeof iccResult === 'number') {
            const iccValue = iccResult
            const quality = iccValue < 0.5 ? 'poor' : iccValue < 0.75 ? 'moderate' : iccValue < 0.9 ? 'good' : 'excellent'
            const interpretation = iccValue < 0.5 ? 'Poor reliability' : iccValue < 0.75 ? 'Moderate reliability' : iccValue < 0.9 ? 'Good reliability' : 'Excellent reliability'
            const colors = qualityColors[quality]

            return (
              <div key={factor} className={`rounded-lg border ${colors.border} ${colors.bg} p-5`}>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-2">
                    <QualityIcon quality={quality} />
                    <h4 className="font-semibold text-gray-200 text-lg">{factor}</h4>
                  </div>
                  <div className={`text-sm font-medium ${colors.text}`}>
                    {interpretation}
                  </div>
                </div>
                {/* Simplified display for legacy format */}
                <div className="text-center py-6">
                  <div className={`text-5xl font-bold ${colors.text} mb-2`}>
                    {(iccValue * 100).toFixed(1)}%
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-4 mt-4">
                    <div
                      className={`h-4 rounded-full transition-all duration-500 ${
                        quality === 'poor' ? 'bg-red-500' :
                        quality === 'moderate' ? 'bg-yellow-500' :
                        quality === 'good' ? 'bg-blue-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${iccValue * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            )
          }

          // Handle error case
          if (iccResult.error) {
            return (
              <div key={factor} className="bg-red-500/10 border border-red-500/50 rounded-lg p-4">
                <p className="text-red-400 text-sm">
                  <strong>{factor}:</strong> {iccResult.error}
                </p>
              </div>
            )
          }

          // Handle full metadata format
          const { icc, icc_type, ci_lower, ci_upper, quality, interpretation, f_statistic, p_value, n_groups, avg_group_size } = iccResult
          const colors = qualityColors[quality]

          return (
            <div key={factor} className={`rounded-lg border ${colors.border} ${colors.bg} p-5`}>
              {/* Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <QualityIcon quality={quality} />
                  <h4 className="font-semibold text-gray-200 text-lg">{factor}</h4>
                  <span className="text-xs bg-slate-700 px-2 py-1 rounded text-gray-300">{icc_type}</span>
                </div>
                <div className={`text-sm font-medium ${colors.text}`}>
                  {interpretation}
                </div>
              </div>

              {/* Gauge */}
              <ICCGauge icc={icc} quality={quality} ciLower={ci_lower} ciUpper={ci_upper} />

              {/* Detailed statistics */}
              {showDetails && (
                <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-slate-600">
                  <div className="text-center">
                    <div className="text-xs text-gray-400 mb-1">F-statistic</div>
                    <div className="text-sm font-mono text-gray-200">{f_statistic.toFixed(3)}</div>
                  </div>
                  <div className="text-center">
                    <div className="text-xs text-gray-400 mb-1">p-value</div>
                    <div className={`text-sm font-mono ${p_value < 0.05 ? 'text-green-400' : 'text-gray-400'}`}>
                      {p_value < 0.001 ? '<0.001' : p_value.toFixed(4)}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-xs text-gray-400 mb-1">Groups</div>
                    <div className="text-sm font-mono text-gray-200">{n_groups}</div>
                  </div>
                  <div className="text-center">
                    <div className="text-xs text-gray-400 mb-1">Avg Group Size</div>
                    <div className="text-sm font-mono text-gray-200">{avg_group_size.toFixed(1)}</div>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Interpretation guide */}
      {showDetails && (
        <div className="mt-6 bg-slate-700/30 rounded-lg p-4">
          <h5 className="font-semibold text-gray-200 mb-3 flex items-center">
            <Info className="w-4 h-4 mr-2" />
            Interpretation Guidelines
          </h5>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-gray-400">
            <div>
              <span className="font-semibold text-red-400">Poor (ICC {'<'} 0.50):</span>
              <span className="ml-1">Low reliability; most variance is within groups rather than between groups.</span>
            </div>
            <div>
              <span className="font-semibold text-yellow-400">Moderate (0.50 ≤ ICC {'<'} 0.75):</span>
              <span className="ml-1">Fair reliability; some clustering effect present.</span>
            </div>
            <div>
              <span className="font-semibold text-blue-400">Good (0.75 ≤ ICC {'<'} 0.90):</span>
              <span className="ml-1">Strong reliability; substantial clustering within groups.</span>
            </div>
            <div>
              <span className="font-semibold text-green-400">Excellent (ICC ≥ 0.90):</span>
              <span className="ml-1">Very strong reliability; observations within groups are highly similar.</span>
            </div>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-600">
            <p className="text-xs text-gray-400">
              <strong>ICC Types:</strong> ICC(1) measures single measurement reliability, ICC(2) average measurement reliability,
              and ICC(3) measures consistency. Higher ICC values justify hierarchical/mixed models.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default ICCDisplay
