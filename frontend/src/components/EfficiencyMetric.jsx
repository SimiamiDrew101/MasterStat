import { TrendingUp, Info } from 'lucide-react'

/**
 * EfficiencyMetric component displays the relative efficiency of a block design
 * compared to a completely randomized design (CRD)
 */
const EfficiencyMetric = ({ relativeEfficiency, designType }) => {
  if (!relativeEfficiency || relativeEfficiency === null) return null

  const percentageGain = ((relativeEfficiency - 1) * 100).toFixed(1)
  const isEfficient = relativeEfficiency > 1

  return (
    <div className="bg-gradient-to-r from-cyan-900/20 to-blue-900/20 rounded-lg p-6 border border-cyan-700/30">
      <div className="flex items-center gap-3 mb-4">
        <TrendingUp className={`w-6 h-6 ${isEfficient ? 'text-cyan-400' : 'text-gray-400'}`} />
        <h4 className="text-xl font-bold text-gray-100">Design Efficiency</h4>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Metric Display */}
        <div className="bg-slate-800/50 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-2">Relative Efficiency vs CRD</div>
          <div className="text-4xl font-bold text-cyan-400 mb-2">
            {relativeEfficiency.toFixed(3)}
          </div>
          {isEfficient && (
            <div className="flex items-center gap-2 text-green-400">
              <TrendingUp className="w-5 h-5" />
              <span className="text-lg font-semibold">+{percentageGain}% more efficient</span>
            </div>
          )}
        </div>

        {/* Interpretation */}
        <div className="bg-slate-800/50 rounded-lg p-4">
          <div className="flex items-start gap-2 mb-3">
            <Info className="w-5 h-5 text-blue-400 mt-0.5" />
            <div className="text-sm text-gray-300">
              <p className="font-semibold text-blue-200 mb-2">What this means:</p>
              {isEfficient ? (
                <>
                  <p className="mb-2">
                    This {designType} is <span className="text-green-400 font-semibold">{percentageGain}% more efficient</span> than
                    a completely randomized design (CRD).
                  </p>
                  <p className="text-xs text-gray-400">
                    You would need {relativeEfficiency.toFixed(2)}× more experimental units in a CRD
                    to achieve the same precision as this blocked design.
                  </p>
                </>
              ) : (
                <p>
                  The blocking strategy provides similar efficiency to a CRD.
                  This may indicate weak blocking effects.
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Educational Note */}
      <div className="mt-4 bg-blue-900/10 rounded-lg p-3 border border-blue-700/20">
        <p className="text-xs text-gray-300">
          <strong className="text-blue-300">Relative Efficiency Formula:</strong> RE =
          [(b-1)MS<sub>Block</sub> + b(t-1)MS<sub>Error</sub>] / [(bt-1)MS<sub>Error</sub>]
        </p>
        <p className="text-xs text-gray-400 mt-1">
          where b = number of blocks, t = number of treatments
        </p>
      </div>
    </div>
  )
}

export default EfficiencyMetric
