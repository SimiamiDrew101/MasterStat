import { useState } from 'react'
import {
  Network,
  Info,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Lightbulb,
  ChevronDown,
  ChevronUp
} from 'lucide-react'
import InteractiveTooltip, { InlineTooltip } from './InteractiveTooltip'
import {
  generateAllInteractions,
  formatInteractionName,
  getInteractionRecommendations
} from '../utils/interactionAnalysis'

const FactorInteractionSelector = ({
  nFactors,
  factorNames,
  goal,
  selectedInteractions = [],
  onInteractionsChange
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Generate all possible interactions
  const allInteractions = generateAllInteractions(nFactors, factorNames)

  // Get recommendations
  const recommendations = getInteractionRecommendations(nFactors, goal)

  // Check if an interaction is selected
  const isSelected = (factor1, factor2) => {
    return selectedInteractions.some(
      int => (int.factor1 === factor1 && int.factor2 === factor2) ||
             (int.factor1 === factor2 && int.factor2 === factor1)
    )
  }

  // Toggle interaction selection
  const toggleInteraction = (factor1, factor2) => {
    const exists = selectedInteractions.find(
      int => (int.factor1 === factor1 && int.factor2 === factor2) ||
             (int.factor1 === factor2 && int.factor2 === factor1)
    )

    if (exists) {
      // Remove
      onInteractionsChange(
        selectedInteractions.filter(
          int => !((int.factor1 === factor1 && int.factor2 === factor2) ||
                   (int.factor1 === factor2 && int.factor2 === factor1))
        )
      )
    } else {
      // Add
      onInteractionsChange([...selectedInteractions, { factor1, factor2 }])
    }
  }

  // Quick select all
  const selectAll = () => {
    const all = allInteractions.map(int => ({ factor1: int.factor1, factor2: int.factor2 }))
    onInteractionsChange(all)
  }

  // Clear all
  const clearAll = () => {
    onInteractionsChange([])
  }

  // If only 2 factors, there's only 1 possible interaction
  if (nFactors < 2) {
    return (
      <div className="bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-yellow-200 font-semibold">Interaction Selection Not Available</p>
            <p className="text-yellow-300 text-sm mt-1">
              You need at least 2 factors to have interactions. Add more factors to use this feature.
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h4 className="text-lg font-bold text-gray-100 mb-1 flex items-center gap-2">
          <Network className="w-5 h-5 text-purple-400" />
          <InlineTooltip term="interactions">Factor Interactions</InlineTooltip>
          <span className="text-sm font-normal text-gray-400">(Optional)</span>
        </h4>
        <p className="text-gray-300 text-sm">
          Do you suspect specific factors interact? Select them to prioritize designs that can estimate these effects.
        </p>
      </div>

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="space-y-2">
          {recommendations.map((rec, idx) => (
            <div
              key={idx}
              className={`border rounded-lg p-3 ${
                rec.type === 'warning'
                  ? 'bg-yellow-900/20 border-yellow-700/50'
                  : rec.type === 'info'
                  ? 'bg-blue-900/20 border-blue-700/50'
                  : 'bg-green-900/20 border-green-700/50'
              }`}
            >
              <div className="flex items-start gap-2">
                {rec.type === 'warning' ? (
                  <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                ) : rec.type === 'info' ? (
                  <Info className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                ) : (
                  <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                )}
                <div className="flex-1">
                  <p
                    className={`text-sm ${
                      rec.type === 'warning'
                        ? 'text-yellow-200'
                        : rec.type === 'info'
                        ? 'text-blue-200'
                        : 'text-green-200'
                    }`}
                  >
                    {rec.message}
                  </p>
                  {rec.action && (
                    <p
                      className={`text-xs mt-1 italic ${
                        rec.type === 'warning'
                          ? 'text-yellow-300'
                          : rec.type === 'info'
                          ? 'text-blue-300'
                          : 'text-green-300'
                      }`}
                    >
                      → {rec.action}
                    </p>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Interaction Matrix */}
      <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <p className="text-gray-200 font-semibold text-sm">
              Select Suspected Interactions
            </p>
            {selectedInteractions.length > 0 && (
              <span className="px-2 py-0.5 bg-purple-700 text-purple-100 text-xs font-bold rounded-full">
                {selectedInteractions.length} selected
              </span>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={selectAll}
              className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-200 rounded text-xs font-medium transition-colors"
            >
              Select All
            </button>
            <button
              onClick={clearAll}
              className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-200 rounded text-xs font-medium transition-colors"
            >
              Clear All
            </button>
          </div>
        </div>

        {/* Grid of interaction buttons */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
          {allInteractions.map((interaction) => {
            const selected = isSelected(interaction.factor1, interaction.factor2)
            return (
              <button
                key={interaction.id}
                onClick={() => toggleInteraction(interaction.factor1, interaction.factor2)}
                className={`p-3 rounded-lg border-2 transition-all text-left ${
                  selected
                    ? 'border-purple-500 bg-purple-900/30 shadow-md'
                    : 'border-slate-600 bg-slate-800/30 hover:border-slate-500'
                }`}
              >
                <div className="flex items-center gap-2">
                  <div
                    className={`w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 ${
                      selected
                        ? 'border-purple-500 bg-purple-500'
                        : 'border-slate-500 bg-transparent'
                    }`}
                  >
                    {selected && <CheckCircle className="w-4 h-4 text-white" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-100 truncate">
                      {interaction.name}
                    </p>
                    <p className="text-xs text-gray-400">
                      {factorNames[interaction.factor1] || `F${interaction.factor1 + 1}`} ×{' '}
                      {factorNames[interaction.factor2] || `F${interaction.factor2 + 1}`}
                    </p>
                  </div>
                </div>
              </button>
            )
          })}
        </div>

        {/* Empty state */}
        {allInteractions.length === 0 && (
          <div className="text-center py-8 text-gray-400">
            <Network className="w-12 h-12 mx-auto mb-2 opacity-30" />
            <p className="text-sm">No interactions available for selection</p>
          </div>
        )}
      </div>

      {/* Advanced Section */}
      <div>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-blue-300 hover:text-blue-200 text-sm font-medium transition-colors"
        >
          {showAdvanced ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
          Advanced: Understanding Interaction Estimation
        </button>

        {showAdvanced && (
          <div className="mt-3 bg-blue-900/20 border border-blue-700/50 rounded-lg p-4 space-y-3">
            <div className="flex items-start gap-2">
              <Lightbulb className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-blue-200 space-y-2">
                <p className="font-semibold">How Designs Handle Interactions:</p>
                <ul className="space-y-2 ml-4 list-none">
                  <li className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <strong>Full Factorial & RSM (CCD, Box-Behnken):</strong> Can estimate ALL
                      2-way interactions clearly
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <strong>Fractional Factorial (Resolution IV):</strong> Can estimate
                      interactions but some are confounded (aliased) with each other
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <XCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <strong>Fractional Factorial (Resolution III) & Plackett-Burman:</strong>{' '}
                      Cannot reliably estimate interactions
                    </div>
                  </li>
                </ul>

                <div className="mt-3 pt-3 border-t border-blue-700/30">
                  <p className="font-semibold mb-1">When to Specify Interactions:</p>
                  <ul className="ml-4 list-disc space-y-1 text-blue-300">
                    <li>
                      You have <strong>prior knowledge</strong> or theory suggesting specific
                      factors interact
                    </li>
                    <li>
                      Previous experiments or <strong>literature</strong> indicate interactions
                    </li>
                    <li>
                      Physical/chemical theory suggests <strong>synergistic effects</strong> (e.g.,
                      temperature × catalyst)
                    </li>
                    <li>
                      You want to <strong>focus resources</strong> on estimating specific
                      interactions
                    </li>
                  </ul>
                </div>

                <div className="mt-3 pt-3 border-t border-blue-700/30">
                  <p className="font-semibold mb-1">Design Selection Impact:</p>
                  <p className="text-blue-300">
                    The wizard will <strong>prioritize designs</strong> that can estimate your
                    selected interactions clearly. Designs that cannot estimate your interactions
                    will receive lower scores. This helps you choose the right design for your
                    specific hypotheses.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Summary when interactions selected */}
      {selectedInteractions.length > 0 && (
        <div className="bg-purple-900/20 border border-purple-700/50 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <Network className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-purple-200 font-semibold mb-2">
                {selectedInteractions.length} Interaction
                {selectedInteractions.length !== 1 ? 's' : ''} Selected
              </p>
              <div className="flex flex-wrap gap-2">
                {selectedInteractions.map((int, idx) => (
                  <span
                    key={idx}
                    className="px-2 py-1 bg-purple-900/50 border border-purple-700/30 rounded text-purple-200 text-xs"
                  >
                    {formatInteractionName(int.factor1, int.factor2, factorNames)}
                  </span>
                ))}
              </div>
              <p className="text-purple-300 text-xs mt-2">
                The wizard will prioritize designs that can estimate these specific interactions.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default FactorInteractionSelector
