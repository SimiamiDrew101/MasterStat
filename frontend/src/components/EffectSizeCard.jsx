import { TrendingUp, Info } from 'lucide-react'

const EffectSizeCard = ({ effectSize, powerAnalysis }) => {
  if (!effectSize) return null

  const { cohens_d, hedges_g, interpretation, confidence_interval } = effectSize

  // Get color based on interpretation
  const getInterpretationColor = (interp) => {
    switch (interp) {
      case 'negligible':
        return 'text-gray-400'
      case 'small':
        return 'text-yellow-400'
      case 'medium':
        return 'text-orange-400'
      case 'large':
        return 'text-green-400'
      default:
        return 'text-gray-400'
    }
  }

  const getInterpretationBg = (interp) => {
    switch (interp) {
      case 'negligible':
        return 'bg-gray-900/20 border-gray-700/30'
      case 'small':
        return 'bg-yellow-900/20 border-yellow-700/30'
      case 'medium':
        return 'bg-orange-900/20 border-orange-700/30'
      case 'large':
        return 'bg-green-900/20 border-green-700/30'
      default:
        return 'bg-gray-900/20 border-gray-700/30'
    }
  }

  // Calculate gauge position (0-100%)
  const gaugePosition = Math.min(Math.abs(cohens_d) / 1.5 * 100, 100)

  // Determine gauge color
  const gaugeColor = interpretation === 'negligible' ? '#6b7280' :
                     interpretation === 'small' ? '#eab308' :
                     interpretation === 'medium' ? '#f97316' : '#22c55e'

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center gap-2 mb-6">
        <TrendingUp className="w-5 h-5 text-green-400" />
        <h3 className="text-xl font-bold text-gray-100">Effect Size Analysis</h3>
      </div>

      <div className="space-y-6">
        {/* Effect Size Magnitude */}
        <div className={`rounded-lg p-5 border ${getInterpretationBg(interpretation)}`}>
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="text-lg font-semibold text-gray-100">Effect Magnitude</h4>
              <p className={`text-2xl font-bold mt-1 ${getInterpretationColor(interpretation)}`}>
                {interpretation.toUpperCase()}
              </p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-400">Cohen's d</p>
              <p className="text-3xl font-bold text-gray-100">{cohens_d.toFixed(3)}</p>
            </div>
          </div>

          {/* Visual gauge */}
          <div className="relative h-8 bg-slate-700/50 rounded-full overflow-hidden">
            <div
              className="absolute top-0 left-0 h-full transition-all duration-500 ease-out rounded-full"
              style={{
                width: `${gaugePosition}%`,
                backgroundColor: gaugeColor,
                boxShadow: `0 0 10px ${gaugeColor}50`
              }}
            />
            {/* Threshold markers */}
            <div className="absolute top-0 left-[13.33%] w-0.5 h-full bg-slate-500"></div>
            <div className="absolute top-0 left-[33.33%] w-0.5 h-full bg-slate-500"></div>
            <div className="absolute top-0 left-[53.33%] w-0.5 h-full bg-slate-500"></div>
          </div>

          {/* Threshold labels */}
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>0.0</span>
            <span>0.2</span>
            <span>0.5</span>
            <span>0.8</span>
            <span>1.5+</span>
          </div>
        </div>

        {/* Effect Size Values */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-semibold text-gray-300">Cohen's d</h4>
              <Info className="w-4 h-4 text-gray-400" />
            </div>
            <p className="text-2xl font-bold text-gray-100">{cohens_d.toFixed(4)}</p>
            <p className="text-xs text-gray-400 mt-1">
              Standardized mean difference
            </p>
            {confidence_interval && (
              <p className="text-xs text-gray-400 mt-2">
                95% CI: [{confidence_interval.lower.toFixed(3)}, {confidence_interval.upper.toFixed(3)}]
              </p>
            )}
          </div>

          {hedges_g !== null && hedges_g !== undefined && (
            <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold text-gray-300">Hedges' g</h4>
                <Info className="w-4 h-4 text-gray-400" />
              </div>
              <p className="text-2xl font-bold text-gray-100">{hedges_g.toFixed(4)}</p>
              <p className="text-xs text-gray-400 mt-1">
                Bias-corrected Cohen's d
              </p>
            </div>
          )}
        </div>

        {/* Interpretation Guide */}
        <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-700/30">
          <h4 className="text-blue-200 font-semibold mb-3">Effect Size Interpretation (Cohen, 1988)</h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div className="text-gray-300">
              <span className="font-semibold text-gray-400">Negligible:</span> |d| &lt; 0.2
            </div>
            <div className="text-gray-300">
              <span className="font-semibold text-yellow-400">Small:</span> |d| = 0.2 - 0.5
            </div>
            <div className="text-gray-300">
              <span className="font-semibold text-orange-400">Medium:</span> |d| = 0.5 - 0.8
            </div>
            <div className="text-gray-300">
              <span className="font-semibold text-green-400">Large:</span> |d| &gt; 0.8
            </div>
          </div>
        </div>

        {/* Power Analysis */}
        {powerAnalysis && powerAnalysis.post_hoc_power !== null && (
          <div className="bg-purple-900/20 rounded-lg p-5 border border-purple-700/30">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-lg font-semibold text-gray-100">Post-Hoc Power</h4>
                <p className={`text-xl font-bold mt-1 ${
                  powerAnalysis.post_hoc_power > 0.8 ? 'text-green-400' :
                  powerAnalysis.post_hoc_power > 0.5 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {powerAnalysis.interpretation}
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm text-gray-400">Statistical Power (1-β)</p>
                <p className="text-3xl font-bold text-gray-100">
                  {(powerAnalysis.post_hoc_power * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Power gauge */}
            <div className="relative h-6 bg-slate-700/50 rounded-full overflow-hidden">
              <div
                className="absolute top-0 left-0 h-full transition-all duration-500 ease-out rounded-full"
                style={{
                  width: `${powerAnalysis.post_hoc_power * 100}%`,
                  backgroundColor: powerAnalysis.post_hoc_power > 0.8 ? '#22c55e' :
                                   powerAnalysis.post_hoc_power > 0.5 ? '#eab308' : '#ef4444',
                  boxShadow: powerAnalysis.post_hoc_power > 0.8
                    ? '0 0 10px #22c55e50'
                    : powerAnalysis.post_hoc_power > 0.5
                    ? '0 0 10px #eab30850'
                    : '0 0 10px #ef444450'
                }}
              />
              {/* 80% threshold marker */}
              <div className="absolute top-0 left-[80%] w-0.5 h-full bg-white opacity-50"></div>
            </div>
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>0%</span>
              <span>50%</span>
              <span className="text-green-400 font-semibold">80%</span>
              <span>100%</span>
            </div>

            <div className="mt-3 pt-3 border-t border-gray-600">
              <p className="text-xs text-gray-300">
                {powerAnalysis.post_hoc_power > 0.8
                  ? '✓ High power: The test had good ability to detect the observed effect'
                  : powerAnalysis.post_hoc_power > 0.5
                  ? '⚠ Moderate power: Consider larger sample size for more reliable results'
                  : '✗ Low power: High risk of Type II error (failing to detect a real effect)'}
              </p>
            </div>
          </div>
        )}

        {/* Educational note */}
        <div className="bg-slate-700/20 rounded-lg p-4 border border-slate-600">
          <p className="text-sm text-gray-300">
            <strong className="text-gray-100">Note:</strong> Effect size quantifies the magnitude of the difference between groups,
            independent of sample size. It provides practical significance beyond statistical significance (p-value).
            Cohen's d represents the difference in means divided by the pooled standard deviation.
          </p>
        </div>
      </div>
    </div>
  )
}

export default EffectSizeCard
