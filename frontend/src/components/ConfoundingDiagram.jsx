import { AlertTriangle, CheckCircle, Info } from 'lucide-react'

/**
 * ConfoundingDiagram component for visualizing alias structure in factorial designs
 * Shows which effects are confounded (aliased) with each other
 * Displays resolution badges and interpretation guidance
 */
const ConfoundingDiagram = ({ confoundingData }) => {
  if (!confoundingData) {
    return (
      <div className="bg-slate-700/50 rounded-lg p-6">
        <p className="text-gray-400 text-center">No confounding data available</p>
      </div>
    )
  }

  const {
    design_type,
    resolution,
    n_factors,
    defining_relation,
    alias_structure,
    alias_groups,
    interpretation
  } = confoundingData

  // Resolution badge color mapping
  const getResolutionColor = (res) => {
    if (typeof res === 'string') {
      if (res.includes('Complex')) return 'bg-purple-600'
      const numMatch = res.match(/\d+/)
      if (numMatch) res = parseInt(numMatch[0])
    }

    if (typeof res === 'number') {
      if (res === 3) return 'bg-yellow-600'
      if (res === 4) return 'bg-blue-600'
      if (res === 5) return 'bg-green-600'
      if (res >= 6) return 'bg-emerald-600'
    }

    return 'bg-purple-600'
  }

  // Get icon for effect estimability
  const getEstimabilityIcon = (estimable) => {
    if (typeof estimable === 'string') {
      if (estimable.toLowerCase().includes('yes')) {
        return <CheckCircle className="w-5 h-5 text-green-400" />
      }
      if (estimable.toLowerCase().includes('no')) {
        return <AlertTriangle className="w-5 h-5 text-red-400" />
      }
      if (estimable.toLowerCase().includes('partial')) {
        return <Info className="w-5 h-5 text-yellow-400" />
      }
    }
    return <Info className="w-5 h-5 text-gray-400" />
  }

  return (
    <div className="bg-slate-700/50 rounded-lg p-6 space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-gray-100 font-semibold text-lg">Confounding Analysis</h4>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Resolution:</span>
            <span className={`px-3 py-1 rounded-full text-white font-bold text-sm ${getResolutionColor(resolution)}`}>
              {resolution}
            </span>
          </div>
        </div>
        <p className="text-gray-400 text-sm">
          {design_type} with {n_factors} factor{n_factors !== 1 ? 's' : ''}
        </p>
      </div>

      {/* Defining Relation */}
      {defining_relation && (
        <div className="bg-slate-800/50 border border-slate-600 rounded-lg p-4">
          <h5 className="text-gray-100 font-medium mb-2">Defining Relation</h5>
          <p className="text-lg font-mono text-blue-300">{defining_relation}</p>
          <p className="text-xs text-gray-400 mt-2">
            The generator that determines the aliasing pattern
          </p>
        </div>
      )}

      {/* Alias Structure */}
      {alias_structure && (
        <div className="space-y-3">
          <h5 className="text-gray-100 font-medium">Alias Structure</h5>

          {/* Main Effects */}
          {alias_structure.main_effects && (
            <div className="bg-slate-800/50 rounded-lg p-4">
              <div className="flex items-start justify-between mb-3">
                <h6 className="text-gray-200 font-medium">Main Effects</h6>
                {getEstimabilityIcon(alias_structure.main_effects.estimable)}
              </div>

              <div className="space-y-2 text-sm">
                {alias_structure.main_effects.clear_of && alias_structure.main_effects.clear_of.length > 0 && (
                  <div>
                    <span className="text-green-400 font-medium">✓ Clear of:</span>
                    <ul className="ml-6 mt-1 space-y-1">
                      {alias_structure.main_effects.clear_of.map((item, idx) => (
                        <li key={idx} className="text-gray-300">{item}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {alias_structure.main_effects.confounded_with && alias_structure.main_effects.confounded_with.length > 0 && (
                  <div>
                    <span className="text-yellow-400 font-medium">⚠ Confounded with:</span>
                    <ul className="ml-6 mt-1 space-y-1">
                      {alias_structure.main_effects.confounded_with.map((item, idx) => (
                        <li key={idx} className="text-gray-300">{item}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="pt-2 border-t border-slate-600">
                  <span className="text-gray-400 text-xs">Estimable: </span>
                  <span className="text-gray-200 text-xs">{alias_structure.main_effects.estimable}</span>
                </div>
              </div>
            </div>
          )}

          {/* Quadratic Effects (if present) */}
          {alias_structure.quadratic_effects && (
            <div className="bg-slate-800/50 rounded-lg p-4">
              <div className="flex items-start justify-between mb-3">
                <h6 className="text-gray-200 font-medium">Quadratic Effects</h6>
                {getEstimabilityIcon(alias_structure.quadratic_effects.estimable)}
              </div>

              <div className="space-y-2 text-sm">
                {alias_structure.quadratic_effects.clear_of && alias_structure.quadratic_effects.clear_of.length > 0 && (
                  <div>
                    <span className="text-green-400 font-medium">✓ Clear of:</span>
                    <ul className="ml-6 mt-1 space-y-1">
                      {alias_structure.quadratic_effects.clear_of.map((item, idx) => (
                        <li key={idx} className="text-gray-300">{item}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {alias_structure.quadratic_effects.confounded_with && alias_structure.quadratic_effects.confounded_with.length > 0 && (
                  <div>
                    <span className="text-yellow-400 font-medium">⚠ Confounded with:</span>
                    <ul className="ml-6 mt-1 space-y-1">
                      {alias_structure.quadratic_effects.confounded_with.map((item, idx) => (
                        <li key={idx} className="text-gray-300">{item}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="pt-2 border-t border-slate-600">
                  <span className="text-gray-400 text-xs">Estimable: </span>
                  <span className="text-gray-200 text-xs">{alias_structure.quadratic_effects.estimable}</span>
                </div>
              </div>
            </div>
          )}

          {/* Two-Factor Interactions */}
          {alias_structure.two_factor_interactions && (
            <div className="bg-slate-800/50 rounded-lg p-4">
              <div className="flex items-start justify-between mb-3">
                <h6 className="text-gray-200 font-medium">Two-Factor Interactions</h6>
                {getEstimabilityIcon(alias_structure.two_factor_interactions.estimable)}
              </div>

              <div className="space-y-2 text-sm">
                {alias_structure.two_factor_interactions.clear_of && alias_structure.two_factor_interactions.clear_of.length > 0 && (
                  <div>
                    <span className="text-green-400 font-medium">✓ Clear of:</span>
                    <ul className="ml-6 mt-1 space-y-1">
                      {alias_structure.two_factor_interactions.clear_of.map((item, idx) => (
                        <li key={idx} className="text-gray-300">{item}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {alias_structure.two_factor_interactions.confounded_with && alias_structure.two_factor_interactions.confounded_with.length > 0 && (
                  <div>
                    <span className="text-yellow-400 font-medium">⚠ Confounded with:</span>
                    <ul className="ml-6 mt-1 space-y-1">
                      {alias_structure.two_factor_interactions.confounded_with.map((item, idx) => (
                        <li key={idx} className="text-gray-300">{item}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="pt-2 border-t border-slate-600">
                  <span className="text-gray-400 text-xs">Estimable: </span>
                  <span className="text-gray-200 text-xs">{alias_structure.two_factor_interactions.estimable}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Specific Alias Groups (for PB and fractional factorial) */}
      {alias_groups && Object.keys(alias_groups).length > 0 && (
        <div className="space-y-2">
          <h5 className="text-gray-100 font-medium">Alias Groups</h5>
          <div className="bg-slate-800/50 rounded-lg p-4 max-h-64 overflow-y-auto">
            <div className="space-y-2 text-sm">
              {Object.entries(alias_groups).map(([effect, aliases]) => (
                <div key={effect} className="flex items-start gap-3 py-2 border-b border-slate-600 last:border-0">
                  <span className="text-blue-300 font-mono min-w-[60px]">{effect}</span>
                  <span className="text-gray-400">=</span>
                  <span className="text-gray-300">{aliases}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Interpretation */}
      {interpretation && (
        <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
          <h5 className="text-blue-100 font-medium mb-3 flex items-center gap-2">
            <Info className="w-5 h-5" />
            Interpretation
          </h5>

          {interpretation.overall && (
            <p className="text-blue-200 text-sm mb-3">
              <strong>Summary:</strong> {interpretation.overall}
            </p>
          )}

          {interpretation.meaning && (
            <p className="text-blue-200 text-sm mb-3">
              {interpretation.meaning}
            </p>
          )}

          {interpretation.strengths && interpretation.strengths.length > 0 && (
            <div className="mb-3">
              <p className="text-green-300 text-sm font-medium mb-2">✓ Strengths:</p>
              <ul className="ml-6 space-y-1 text-sm">
                {interpretation.strengths.map((strength, idx) => (
                  <li key={idx} className="text-blue-200">{strength}</li>
                ))}
              </ul>
            </div>
          )}

          {interpretation.limitations && interpretation.limitations.length > 0 && (
            <div className="mb-3">
              <p className="text-yellow-300 text-sm font-medium mb-2">⚠ Limitations:</p>
              <ul className="ml-6 space-y-1 text-sm">
                {interpretation.limitations.map((limitation, idx) => (
                  <li key={idx} className="text-blue-200">{limitation}</li>
                ))}
              </ul>
            </div>
          )}

          {interpretation.recommendation && (
            <div className="bg-blue-800/30 rounded p-3 mt-3">
              <p className="text-blue-100 text-sm">
                <strong>Recommendation:</strong> {interpretation.recommendation}
              </p>
            </div>
          )}

          {interpretation.warning && (
            <div className="bg-red-900/30 border border-red-700/50 rounded p-3 mt-3">
              <p className="text-red-200 text-sm font-medium">
                {interpretation.warning}
              </p>
            </div>
          )}

          {interpretation.assumption && (
            <div className="bg-yellow-900/20 border border-yellow-700/50 rounded p-3 mt-3">
              <p className="text-yellow-200 text-sm">
                <strong>Assumption:</strong> {interpretation.assumption}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Resolution Guide */}
      <div className="bg-slate-800/50 rounded-lg p-4">
        <h5 className="text-gray-100 font-medium mb-3">Resolution Guide</h5>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          <div className="flex items-start gap-2">
            <span className="px-2 py-1 rounded bg-yellow-600 text-white font-bold">III</span>
            <span className="text-gray-300">Main effects confounded with 2-factor interactions</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="px-2 py-1 rounded bg-blue-600 text-white font-bold">IV</span>
            <span className="text-gray-300">Main effects clear; 2FI confounded with each other</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="px-2 py-1 rounded bg-green-600 text-white font-bold">V</span>
            <span className="text-gray-300">Main effects and 2FI clear of each other</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="px-2 py-1 rounded bg-purple-600 text-white font-bold">*</span>
            <span className="text-gray-300">Complex aliasing (not traditional resolution)</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ConfoundingDiagram
