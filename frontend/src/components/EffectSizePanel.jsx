import { TrendingUp, Info } from 'lucide-react'

const EffectSizePanel = ({ effectSizes, testType }) => {
  if (!effectSizes) return null

  const getEffectColorClass = (interpretation) => {
    switch (interpretation) {
      case 'negligible':
        return 'text-gray-400'
      case 'small':
        return 'text-blue-400'
      case 'medium':
        return 'text-yellow-400'
      case 'large':
        return 'text-red-400'
      default:
        return 'text-gray-400'
    }
  }

  const getEffectBgClass = (interpretation) => {
    switch (interpretation) {
      case 'negligible':
        return 'bg-gray-500/20 border-gray-500/50'
      case 'small':
        return 'bg-blue-500/20 border-blue-500/50'
      case 'medium':
        return 'bg-yellow-500/20 border-yellow-500/50'
      case 'large':
        return 'bg-red-500/20 border-red-500/50'
      default:
        return 'bg-gray-500/20 border-gray-500/50'
    }
  }

  // For one-way ANOVA
  if (testType === 'One-Way ANOVA') {
    return (
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <TrendingUp className="w-6 h-6 text-purple-400" />
          <h3 className="text-xl font-bold text-gray-100">Effect Sizes</h3>
        </div>

        <p className="text-gray-400 text-sm mb-6">
          Effect sizes measure the magnitude of differences between groups, providing practical significance beyond statistical significance.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Eta-squared */}
          <div className={`rounded-lg border p-4 ${getEffectBgClass(effectSizes.eta_squared.interpretation)}`}>
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-semibold text-gray-100">Eta-Squared (η²)</h4>
              <Info className="w-4 h-4 text-gray-400" />
            </div>
            <div className="mb-3">
              <div className={`text-3xl font-bold ${getEffectColorClass(effectSizes.eta_squared.interpretation)}`}>
                {effectSizes.eta_squared.value}
              </div>
              <div className={`text-sm font-semibold capitalize ${getEffectColorClass(effectSizes.eta_squared.interpretation)}`}>
                {effectSizes.eta_squared.interpretation}
              </div>
            </div>
            <p className="text-xs text-gray-400">{effectSizes.eta_squared.description}</p>
            <div className="mt-3 pt-3 border-t border-gray-600">
              <p className="text-xs text-gray-300">
                {(effectSizes.eta_squared.value * 100).toFixed(1)}% of variance explained
              </p>
            </div>
          </div>

          {/* Omega-squared */}
          <div className={`rounded-lg border p-4 ${getEffectBgClass(effectSizes.omega_squared.interpretation)}`}>
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-semibold text-gray-100">Omega-Squared (ω²)</h4>
              <span className="text-xs bg-purple-500/30 text-purple-200 px-2 py-1 rounded">Recommended</span>
            </div>
            <div className="mb-3">
              <div className={`text-3xl font-bold ${getEffectColorClass(effectSizes.omega_squared.interpretation)}`}>
                {effectSizes.omega_squared.value}
              </div>
              <div className={`text-sm font-semibold capitalize ${getEffectColorClass(effectSizes.omega_squared.interpretation)}`}>
                {effectSizes.omega_squared.interpretation}
              </div>
            </div>
            <p className="text-xs text-gray-400">{effectSizes.omega_squared.description}</p>
            <div className="mt-3 pt-3 border-t border-gray-600">
              <p className="text-xs text-gray-300">
                {(effectSizes.omega_squared.value * 100).toFixed(1)}% of variance explained (unbiased)
              </p>
            </div>
          </div>

          {/* Cohen's f */}
          <div className={`rounded-lg border p-4 ${getEffectBgClass(effectSizes.cohens_f.interpretation)}`}>
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-semibold text-gray-100">Cohen's f</h4>
              <Info className="w-4 h-4 text-gray-400" />
            </div>
            <div className="mb-3">
              <div className={`text-3xl font-bold ${getEffectColorClass(effectSizes.cohens_f.interpretation)}`}>
                {effectSizes.cohens_f.value}
              </div>
              <div className={`text-sm font-semibold capitalize ${getEffectColorClass(effectSizes.cohens_f.interpretation)}`}>
                {effectSizes.cohens_f.interpretation}
              </div>
            </div>
            <p className="text-xs text-gray-400">{effectSizes.cohens_f.description}</p>
          </div>
        </div>

        {/* Interpretation Guide */}
        <div className="mt-6 bg-slate-700/30 rounded-lg p-4">
          <h5 className="font-semibold text-gray-200 mb-3">Interpretation Guidelines (Cohen, 1988)</h5>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div>
              <h6 className="font-semibold text-gray-300 mb-2">η² and ω²:</h6>
              <ul className="space-y-1 text-gray-400">
                <li>• Small: 0.01 (1% of variance)</li>
                <li>• Medium: 0.06 (6% of variance)</li>
                <li>• Large: 0.14 (14% of variance)</li>
              </ul>
            </div>
            <div>
              <h6 className="font-semibold text-gray-300 mb-2">Cohen's f:</h6>
              <ul className="space-y-1 text-gray-400">
                <li>• Small: 0.10</li>
                <li>• Medium: 0.25</li>
                <li>• Large: 0.40</li>
              </ul>
            </div>
          </div>
          <p className="text-xs text-gray-400 mt-3">
            <strong>Note:</strong> Omega-squared (ω²) is preferred over eta-squared (η²) as it's less biased,
            especially with smaller sample sizes.
          </p>
        </div>
      </div>
    )
  }

  // For two-way ANOVA (multiple effects)
  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center space-x-3 mb-4">
        <TrendingUp className="w-6 h-6 text-purple-400" />
        <h3 className="text-xl font-bold text-gray-100">Effect Sizes</h3>
      </div>

      <p className="text-gray-400 text-sm mb-6">
        Partial eta-squared measures the proportion of variance explained by each effect, controlling for other effects in the model.
      </p>

      <div className="space-y-4">
        {Object.entries(effectSizes).map(([effectName, sizes]) => (
          <div key={effectName} className="bg-slate-700/30 rounded-lg p-4">
            <h4 className="font-semibold text-gray-100 mb-3">{effectName}</h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Partial Eta-squared */}
              <div className={`rounded-lg border p-3 ${getEffectBgClass(sizes.partial_eta_squared.interpretation)}`}>
                <div className="flex items-center justify-between mb-2">
                  <h5 className="text-sm font-semibold text-gray-100">Partial η²</h5>
                  <span className="text-xs bg-purple-500/30 text-purple-200 px-2 py-1 rounded">Recommended</span>
                </div>
                <div className="mb-2">
                  <div className={`text-2xl font-bold ${getEffectColorClass(sizes.partial_eta_squared.interpretation)}`}>
                    {sizes.partial_eta_squared.value}
                  </div>
                  <div className={`text-xs font-semibold capitalize ${getEffectColorClass(sizes.partial_eta_squared.interpretation)}`}>
                    {sizes.partial_eta_squared.interpretation}
                  </div>
                </div>
                <p className="text-xs text-gray-400 mb-2">{sizes.partial_eta_squared.description}</p>
                <div className="pt-2 border-t border-gray-600">
                  <p className="text-xs text-gray-300">
                    {(sizes.partial_eta_squared.value * 100).toFixed(1)}% of variance
                  </p>
                </div>
              </div>

              {/* Cohen's f */}
              <div className={`rounded-lg border p-3 ${getEffectBgClass(sizes.cohens_f.interpretation)}`}>
                <div className="flex items-center justify-between mb-2">
                  <h5 className="text-sm font-semibold text-gray-100">Cohen's f</h5>
                  <Info className="w-4 h-4 text-gray-400" />
                </div>
                <div className="mb-2">
                  <div className={`text-2xl font-bold ${getEffectColorClass(sizes.cohens_f.interpretation)}`}>
                    {sizes.cohens_f.value}
                  </div>
                  <div className={`text-xs font-semibold capitalize ${getEffectColorClass(sizes.cohens_f.interpretation)}`}>
                    {sizes.cohens_f.interpretation}
                  </div>
                </div>
                <p className="text-xs text-gray-400">{sizes.cohens_f.description}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Interpretation Guide */}
      <div className="mt-6 bg-slate-700/30 rounded-lg p-4">
        <h5 className="font-semibold text-gray-200 mb-3">Interpretation Guidelines (Cohen, 1988)</h5>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div>
            <h6 className="font-semibold text-gray-300 mb-2">Partial η²:</h6>
            <ul className="space-y-1 text-gray-400">
              <li>• Small: 0.01 (1% of variance)</li>
              <li>• Medium: 0.06 (6% of variance)</li>
              <li>• Large: 0.14 (14% of variance)</li>
            </ul>
          </div>
          <div>
            <h6 className="font-semibold text-gray-300 mb-2">Cohen's f:</h6>
            <ul className="space-y-1 text-gray-400">
              <li>• Small: 0.10</li>
              <li>• Medium: 0.25</li>
              <li>• Large: 0.40</li>
            </ul>
          </div>
        </div>
        <p className="text-xs text-gray-400 mt-3">
          <strong>Note:</strong> Partial η² is appropriate for factorial designs as it accounts for variance explained by other factors.
        </p>
      </div>
    </div>
  )
}

export default EffectSizePanel
