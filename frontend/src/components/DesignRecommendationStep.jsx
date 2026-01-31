import { useState, useEffect } from 'react'
import { Check, X, AlertTriangle, TrendingUp, Info, ChevronDown, ChevronUp, Star, Zap, Shield, DollarSign, Clock, Target, GitCompare, BarChart3, Network } from 'lucide-react'
import InteractiveTooltip, { InlineTooltip } from './InteractiveTooltip'
import { calculateInteractionCoverage, getInteractionCapability } from '../utils/interactionAnalysis'

const DesignRecommendationStep = ({ nFactors, budget, goal, minimumRuns, selectedInteractions = [], selectedDesign, onSelectDesign }) => {
  const [recommendations, setRecommendations] = useState([])
  const [allDesigns, setAllDesigns] = useState([])
  const [warning, setWarning] = useState(null)
  const [powerWarning, setPowerWarning] = useState(null)
  const [comparisonMode, setComparisonMode] = useState(false)
  const [selectedForComparison, setSelectedForComparison] = useState([])

  // Map design codes to glossary terms for tooltips
  const getGlossaryTerm = (designCode) => {
    const termMap = {
      'ccd-face': 'ccd',
      'ccd-rotatable': 'ccd',
      'box-behnken': 'box-behnken',
      'full-factorial': 'full-factorial',
      'fractional-factorial': 'fractional-factorial',
      'plackett-burman': 'plackett-burman'
    }
    return termMap[designCode] || null
  }

  useEffect(() => {
    generateRecommendations()
  }, [nFactors, budget, goal, minimumRuns, selectedInteractions])

  const generateRecommendations = () => {
    const designs = getAllDesigns(nFactors, budget, goal)

    // Apply interaction coverage scoring if interactions are specified
    const scoredDesigns = designs.map(design => {
      const interactionCoverage = calculateInteractionCoverage(design.design_code, nFactors, selectedInteractions)
      const interactionCapability = getInteractionCapability(design.design_code, nFactors)

      // If interactions are specified, adjust score based on coverage
      let adjustedScore = design.score
      if (selectedInteractions && selectedInteractions.length > 0) {
        // Weight: 70% original score + 30% interaction coverage
        adjustedScore = design.score * 0.7 + interactionCoverage * 100 * 0.3
      }

      return {
        ...design,
        baseScore: design.score,
        interactionCoverage: interactionCoverage,
        interactionCapability: interactionCapability,
        score: Math.round(adjustedScore)
      }
    })

    setAllDesigns(scoredDesigns)

    // Sort by adjusted score and filter based on budget
    const filtered = budget
      ? scoredDesigns.filter(d => d.runs <= budget)
      : scoredDesigns

    const sorted = filtered.sort((a, b) => b.score - a.score)
    setRecommendations(sorted.slice(0, 4)) // Top 4 recommendations

    // Check for budget warnings
    if (budget && filtered.length === 0) {
      setWarning(`Budget of ${budget} runs is insufficient for ${nFactors} factors. Minimum ${designs[0]?.runs} runs recommended.`)
    } else if (budget && filtered.length < designs.length) {
      const excluded = designs.filter(d => d.runs > budget)
      setWarning(`Budget constraint excludes ${excluded.length} design option(s). Consider increasing budget for more options.`)
    } else {
      setWarning(null)
    }

    // Check for power analysis warnings
    if (minimumRuns) {
      if (budget && budget < minimumRuns) {
        setPowerWarning(
          `⚠️ Your budget (${budget} runs) is below the recommended minimum (${minimumRuns} runs) from power analysis. ` +
          `This may result in an underpowered experiment that cannot reliably detect ${goal === 'screening' ? 'important factors' : 'meaningful effects'}.`
        )
      } else if (!budget && sorted.length > 0 && sorted[0].runs < minimumRuns) {
        setPowerWarning(
          `💡 The top recommended design has ${sorted[0].runs} runs, but power analysis suggests ${minimumRuns} runs ` +
          `for adequate statistical power. Consider designs with more runs or accept lower statistical power.`
        )
      } else {
        setPowerWarning(null)
      }
    }

    // Auto-select top recommendation
    if (sorted.length > 0 && !selectedDesign) {
      onSelectDesign(sorted[0])
    }
  }

  const toggleComparison = (design) => {
    if (selectedForComparison.find(d => d.design_code === design.design_code)) {
      setSelectedForComparison(selectedForComparison.filter(d => d.design_code !== design.design_code))
    } else {
      if (selectedForComparison.length < 4) { // Max 4 designs for comparison
        setSelectedForComparison([...selectedForComparison, design])
      }
    }
  }

  const startComparison = () => {
    if (selectedForComparison.length === 0) {
      // Auto-select top 3 recommendations
      setSelectedForComparison(recommendations.slice(0, Math.min(3, recommendations.length)))
    }
    setComparisonMode(true)
  }

  const exitComparison = () => {
    setComparisonMode(false)
  }

  if (comparisonMode) {
    return (
      <DesignComparison
        designs={selectedForComparison.length > 0 ? selectedForComparison : recommendations.slice(0, 3)}
        onExit={exitComparison}
        onSelectDesign={onSelectDesign}
        selectedDesign={selectedDesign}
      />
    )
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-2xl font-bold text-gray-100 mb-1">Recommended Experimental Designs</h3>
          <p className="text-gray-300 text-sm">
            Based on {nFactors} factors, {goal} goal{budget ? `, and ${budget}-run budget` : ''}, here are expert recommendations:
          </p>
        </div>
        <button
          onClick={startComparison}
          className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-lg font-semibold transition-all hover:scale-105 shadow-lg"
        >
          <GitCompare className="w-5 h-5" />
          Compare Designs
        </button>
      </div>

      {/* Power Analysis Warning */}
      {powerWarning && (
        <div className="mb-6 bg-orange-900/20 border border-orange-700/50 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <Zap className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-orange-300 font-semibold mb-1">Statistical Power Alert</p>
              <p className="text-orange-200 text-sm">{powerWarning}</p>
            </div>
          </div>
        </div>
      )}

      {/* Interaction Coverage Info */}
      {selectedInteractions && selectedInteractions.length > 0 && (
        <div className="mb-6 bg-purple-900/20 border border-purple-700/50 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <Network className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-purple-300 font-semibold mb-1">Interaction Coverage Prioritized</p>
              <p className="text-purple-200 text-sm">
                Design scores have been adjusted to prioritize designs that can estimate your {selectedInteractions.length} selected interaction{selectedInteractions.length !== 1 ? 's' : ''}.
                Designs with higher interaction coverage will rank higher.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Budget Warning */}
      {warning && (
        <div className="mb-6 bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-yellow-300 font-semibold mb-1">Budget Consideration</p>
              <p className="text-yellow-200 text-sm">{warning}</p>
            </div>
          </div>
        </div>
      )}

      {/* Recommendations Grid */}
      <div className="space-y-4 mb-6">
        {recommendations.map((design, index) => (
          <DesignCard
            key={design.design_code}
            design={design}
            rank={index + 1}
            isRecommended={index === 0}
            isSelected={selectedDesign?.design_code === design.design_code}
            onSelect={() => onSelectDesign(design)}
            glossaryTerm={getGlossaryTerm(design.design_code)}
          />
        ))}
      </div>

      {/* All Options Expandable */}
      {allDesigns.length > recommendations.length && (
        <AllDesignsSection
          allDesigns={allDesigns}
          recommendations={recommendations}
          selectedDesign={selectedDesign}
          onSelectDesign={onSelectDesign}
        />
      )}

      {/* Expert Guidance */}
      <div className="mt-6 bg-gradient-to-r from-blue-900/20 to-purple-900/20 border border-blue-700/50 rounded-lg p-5">
        <h4 className="font-bold text-blue-200 mb-3 flex items-center gap-2">
          <Info className="w-5 h-5" />
          Expert Guidance
        </h4>
        <div className="space-y-2 text-sm text-blue-100">
          <p>
            <strong>Top Recommendation:</strong> The #1 ranked design balances statistical power, efficiency, and practical considerations for your specific scenario.
          </p>
          <p>
            <strong>Choosing Alternatives:</strong> Consider lower-ranked options if you have specific constraints (budget, time, equipment limitations) or requirements (rotatability, orthogonality).
          </p>
          <p>
            <strong>Need Help?:</strong> Expand any design card to see detailed pros/cons, practical considerations, and when to use each design.
          </p>
        </div>
      </div>
    </div>
  )
}

const AllDesignsSection = ({ allDesigns, recommendations, selectedDesign, onSelectDesign }) => {
  const [expanded, setExpanded] = useState(false)
  const otherDesigns = allDesigns.filter(d => !recommendations.find(r => r.design_code === d.design_code))

  return (
    <div className="border border-slate-600 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-5 py-4 bg-slate-800/50 hover:bg-slate-700/50 transition-colors flex items-center justify-between"
      >
        <div className="flex items-center gap-2">
          <span className="text-gray-200 font-semibold">View All Design Options</span>
          <span className="text-gray-400 text-sm">({otherDesigns.length} more)</span>
        </div>
        {expanded ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
      </button>

      {expanded && (
        <div className="p-4 space-y-4 bg-slate-900/30">
          {otherDesigns.map((design, index) => (
            <DesignCard
              key={design.design_code}
              design={design}
              rank={recommendations.length + index + 1}
              isRecommended={false}
              isSelected={selectedDesign?.design_code === design.design_code}
              onSelect={() => onSelectDesign(design)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

const DesignCard = ({ design, rank, isRecommended, isSelected, onSelect, glossaryTerm }) => {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className={`border-2 rounded-xl transition-all duration-200 ${
        isSelected
          ? 'border-blue-500 bg-blue-900/30 shadow-lg shadow-blue-500/20'
          : 'border-slate-600 bg-slate-800/30 hover:border-slate-500'
      }`}
    >
      {/* Header */}
      <div
        className="p-5 cursor-pointer"
        onClick={onSelect}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2 flex-wrap">
              {/* Rank Badge */}
              <div className={`flex items-center justify-center w-8 h-8 rounded-full font-bold text-sm ${
                rank === 1 ? 'bg-yellow-500 text-yellow-900' :
                rank === 2 ? 'bg-gray-400 text-gray-900' :
                rank === 3 ? 'bg-orange-600 text-orange-100' :
                'bg-slate-600 text-slate-200'
              }`}>
                #{rank}
              </div>

              {isRecommended && (
                <span className="px-2 py-1 bg-green-600 text-white text-xs font-bold rounded flex items-center gap-1">
                  <Star className="w-3 h-3" />
                  BEST CHOICE
                </span>
              )}

              <h4 className="text-xl font-bold text-gray-100 flex items-center gap-2">
                {design.type}
                {glossaryTerm && <InteractiveTooltip term={glossaryTerm} mode="both" />}
              </h4>

              <div
                className={`w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all ${
                  isSelected
                    ? 'border-blue-500 bg-blue-500'
                    : 'border-slate-500 bg-transparent'
                }`}
              >
                {isSelected && <Check className="w-4 h-4 text-white" />}
              </div>
            </div>

            <p className="text-gray-300 mb-2">{design.description}</p>
            <p className="text-blue-300 text-sm font-medium italic flex items-center gap-2">
              <Target className="w-4 h-4" />
              {design.best_for}
            </p>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
          {/* Runs */}
          <div className="bg-slate-700/50 rounded-lg p-3">
            <p className="text-gray-400 text-xs mb-1">Experimental Runs</p>
            <p className="text-xl font-bold text-blue-300">{design.runs}</p>
          </div>

          {/* Cost */}
          <div className="bg-slate-700/50 rounded-lg p-3">
            <p className="text-gray-400 text-xs mb-1 flex items-center gap-1">
              <DollarSign className="w-3 h-3" />
              Cost
            </p>
            <p className="text-lg font-bold text-gray-200">
              {design.cost_rating}
            </p>
          </div>

          {/* Efficiency */}
          <div className="bg-slate-700/50 rounded-lg p-3">
            <p className="text-gray-400 text-xs mb-1 flex items-center gap-1">
              <Zap className="w-3 h-3" />
              Efficiency
            </p>
            <p className="text-lg font-bold text-gray-200">
              {design.efficiency_rating}
            </p>
          </div>

          {/* Robustness */}
          <div className="bg-slate-700/50 rounded-lg p-3">
            <p className="text-gray-400 text-xs mb-1 flex items-center gap-1">
              <Shield className="w-3 h-3" />
              Robustness
            </p>
            <p className="text-lg font-bold text-gray-200">
              {design.robustness_rating}
            </p>
          </div>

          {/* Score */}
          <div className="bg-gradient-to-br from-green-900/30 to-emerald-900/30 border border-green-700/50 rounded-lg p-3">
            <p className="text-gray-300 text-xs mb-1">Overall Score</p>
            <div className="flex items-center gap-1">
              <TrendingUp className="w-4 h-4 text-green-400" />
              <p className="text-xl font-bold text-green-300">{design.score}/100</p>
            </div>
          </div>
        </div>

        {/* Quick Stats Row */}
        <div className="flex flex-wrap gap-2 mb-3">
          {design.properties.rotatable && (
            <span className="px-2 py-1 bg-purple-900/30 border border-purple-700/50 rounded text-purple-200 text-xs">
              ✓ Rotatable
            </span>
          )}
          {design.properties.orthogonal && (
            <span className="px-2 py-1 bg-indigo-900/30 border border-indigo-700/50 rounded text-indigo-200 text-xs">
              ✓ Orthogonal
            </span>
          )}
          {design.properties.requires_center_points && (
            <span className="px-2 py-1 bg-blue-900/30 border border-blue-700/50 rounded text-blue-200 text-xs">
              • Center Points
            </span>
          )}
          {design.properties.three_level && (
            <span className="px-2 py-1 bg-teal-900/30 border border-teal-700/50 rounded text-teal-200 text-xs">
              3-Level
            </span>
          )}
          {design.interactionCapability && (
            <span
              className={`px-2 py-1 rounded text-xs font-medium flex items-center gap-1 ${
                design.interactionCapability.color === 'green'
                  ? 'bg-green-900/30 border border-green-700/50 text-green-200'
                  : design.interactionCapability.color === 'blue'
                  ? 'bg-blue-900/30 border border-blue-700/50 text-blue-200'
                  : design.interactionCapability.color === 'yellow'
                  ? 'bg-yellow-900/30 border border-yellow-700/50 text-yellow-200'
                  : 'bg-red-900/30 border border-red-700/50 text-red-200'
              }`}
              title={design.interactionCapability.description}
            >
              <Network className="w-3 h-3" />
              {design.interactionCapability.clear === design.interactionCapability.total ? '✓ All Interactions' :
               design.interactionCapability.clear > 0 ? `${design.interactionCapability.clear}/${design.interactionCapability.total} Interactions` :
               design.interactionCapability.confounded > 0 ? 'Some Interactions' : 'No Interactions'}
            </span>
          )}
        </div>

        {/* Toggle Details */}
        <button
          onClick={(e) => {
            e.stopPropagation()
            setExpanded(!expanded)
          }}
          className="text-blue-400 text-sm font-medium hover:text-blue-300 transition-colors flex items-center gap-2"
        >
          {expanded ? (
            <>
              <ChevronUp className="w-4 h-4" />
              Hide Detailed Analysis
            </>
          ) : (
            <>
              <ChevronDown className="w-4 h-4" />
              Show Detailed Analysis & Pros/Cons
            </>
          )}
        </button>
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div className="px-5 pb-5 border-t border-slate-700/50 pt-5 space-y-5">
          {/* Pros and Cons */}
          <div className="grid md:grid-cols-2 gap-5">
            {/* Pros */}
            <div className="bg-green-900/10 border border-green-700/30 rounded-lg p-4">
              <h5 className="font-bold text-green-400 mb-3 flex items-center gap-2 text-lg">
                <Check className="w-5 h-5" />
                Advantages
              </h5>
              <ul className="space-y-2">
                {design.pros.map((pro, i) => (
                  <li key={i} className="text-gray-200 text-sm flex items-start gap-2">
                    <span className="text-green-500 mt-1 text-lg">•</span>
                    <span>{pro}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Cons */}
            <div className="bg-red-900/10 border border-red-700/30 rounded-lg p-4">
              <h5 className="font-bold text-red-400 mb-3 flex items-center gap-2 text-lg">
                <X className="w-5 h-5" />
                Disadvantages
              </h5>
              <ul className="space-y-2">
                {design.cons.map((con, i) => (
                  <li key={i} className="text-gray-200 text-sm flex items-start gap-2">
                    <span className="text-red-500 mt-1 text-lg">•</span>
                    <span>{con}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* When to Use */}
          <div className="bg-blue-900/10 border border-blue-700/30 rounded-lg p-4">
            <h5 className="font-bold text-blue-400 mb-2 text-lg">When to Use This Design</h5>
            <p className="text-gray-200 text-sm">{design.when_to_use}</p>
          </div>

          {/* Practical Considerations */}
          <div className="bg-purple-900/10 border border-purple-700/30 rounded-lg p-4">
            <h5 className="font-bold text-purple-400 mb-3 text-lg">Practical Considerations</h5>
            <ul className="space-y-2">
              {design.practical_considerations.map((consideration, i) => (
                <li key={i} className="text-gray-200 text-sm flex items-start gap-2">
                  <span className="text-purple-400 mt-1">→</span>
                  <span>{consideration}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Statistical Properties */}
          <div className="bg-slate-700/30 rounded-lg p-4">
            <h5 className="font-bold text-gray-300 mb-3">Statistical Properties</h5>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
              {Object.entries(design.properties).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between p-2 bg-slate-800/50 rounded">
                  <span className="text-gray-400">{formatKey(key)}:</span>
                  <span className="text-gray-100 font-semibold ml-2">
                    {formatValue(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Resource Requirements */}
          {design.resource_requirements && (
            <div className="bg-orange-900/10 border border-orange-700/30 rounded-lg p-4">
              <h5 className="font-bold text-orange-400 mb-3 flex items-center gap-2">
                <Clock className="w-5 h-5" />
                Resource Requirements
              </h5>
              <div className="grid md:grid-cols-3 gap-3 text-sm">
                <div>
                  <span className="text-gray-400">Time:</span>
                  <span className="text-gray-100 font-semibold ml-2">{design.resource_requirements.time}</span>
                </div>
                <div>
                  <span className="text-gray-400">Budget:</span>
                  <span className="text-gray-100 font-semibold ml-2">{design.resource_requirements.budget}</span>
                </div>
                <div>
                  <span className="text-gray-400">Expertise:</span>
                  <span className="text-gray-100 font-semibold ml-2">{design.resource_requirements.expertise}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Helper function to generate all design recommendations
const getAllDesigns = (nFactors, budget, goal) => {
  const designs = []

  // Calculate runs for various designs
  const fullFactorialRuns = Math.pow(2, nFactors)
  const ccdFaceCenteredRuns = fullFactorialRuns + 2 * nFactors + 4 // cube + star + center
  const ccdRotatableRuns = fullFactorialRuns + 2 * nFactors + 6
  const boxBehnkenRuns = nFactors >= 3 ? (nFactors * (nFactors - 1) / 2) * 4 + 3 : 0
  const fractionalRuns = nFactors >= 4 ? Math.pow(2, nFactors - 1) + 3 : 0
  const plackettBurmanRuns = nFactors >= 4 ? Math.ceil((nFactors + 1) / 4) * 4 + 4 : 0
  const dsdRuns = nFactors >= 4 ? 2 * nFactors + 1 : 0

  // Full Factorial Design
  if (nFactors <= 6) {
    designs.push({
      design_code: 'full-factorial',
      type: `Full Factorial Design (2^${nFactors})`,
      description: 'Tests all possible combinations of factor levels. Gold standard for comprehensive factor exploration.',
      best_for: 'Complete understanding of main effects and all interactions',
      runs: fullFactorialRuns,
      score: goal === 'screening' && nFactors <= 4 ? 90 : goal === 'modeling' && nFactors <= 3 ? 85 : nFactors <= 4 ? 80 : 60,
      cost_rating: nFactors <= 3 ? 'Low' : nFactors <= 4 ? 'Medium' : nFactors <= 5 ? 'High' : 'Very High',
      efficiency_rating: nFactors <= 4 ? 'Excellent' : nFactors <= 5 ? 'Good' : 'Fair',
      robustness_rating: 'Excellent',
      properties: {
        orthogonal: true,
        rotatable: false,
        three_level: false,
        requires_center_points: false,
        estimates_all_interactions: true,
        resolution: 'Full',
      },
      pros: [
        'Estimates all main effects and interactions with maximum precision',
        'Orthogonal design ensures independent factor effect estimates',
        'No confounding - each effect estimated separately',
        'Well-understood and widely accepted methodology',
        'Provides complete response surface information',
        'Easy to analyze and interpret results',
        'Suitable for subsequent response surface methodology'
      ],
      cons: [
        `Requires ${fullFactorialRuns} experimental runs - can be expensive for many factors`,
        'Run count grows exponentially (2^k) with number of factors',
        'May be impractical for experiments with high per-run costs',
        'Can be time-consuming for 5+ factors',
        'Only explores factor space at two levels (unless extended)',
        'Might generate more data than necessary for simple screening'
      ],
      when_to_use: `Use when you have ≤4 factors and need complete information about all interactions, or when each experimental run is relatively inexpensive. Ideal for ${nFactors} factors if budget allows for ${fullFactorialRuns} runs. Essential when interaction effects are expected to be significant.`,
      practical_considerations: [
        `Budget: Requires ${fullFactorialRuns} runs - ensure resources are available`,
        'Best for situations where you need definitive answers about all factor combinations',
        'Consider randomizing run order to protect against time trends',
        'Add center points (3-5) to check for curvature in the response',
        'Can be split into blocks if all runs cannot be completed in one batch',
        'Provides foundation for follow-up optimization studies'
      ],
      resource_requirements: {
        time: nFactors <= 3 ? 'Short' : nFactors <= 4 ? 'Medium' : 'Long',
        budget: nFactors <= 3 ? 'Low' : nFactors <= 4 ? 'Medium' : nFactors <= 5 ? 'High' : 'Very High',
        expertise: 'Basic'
      }
    })
  }

  // Fractional Factorial Design
  if (nFactors >= 4 && nFactors <= 8) {
    const resolution = nFactors <= 5 ? 'V' : 'IV'
    designs.push({
      design_code: 'fractional-factorial',
      type: `Fractional Factorial Design (2^${nFactors-1})`,
      description: `Resolution ${resolution} fractional factorial. Tests half the combinations while maintaining key information about effects.`,
      best_for: 'Efficient screening when you have 4-8 factors and limited budget',
      runs: fractionalRuns,
      score: goal === 'screening' && nFactors >= 5 ? 92 : goal === 'screening' ? 88 : 75,
      cost_rating: nFactors <= 5 ? 'Low' : 'Medium',
      efficiency_rating: 'Excellent',
      robustness_rating: 'Very Good',
      properties: {
        orthogonal: true,
        rotatable: false,
        three_level: false,
        requires_center_points: true,
        estimates_all_interactions: false,
        resolution: resolution,
      },
      pros: [
        `Reduces runs by 50% (only ${fractionalRuns} runs vs ${fullFactorialRuns})`,
        'Maintains orthogonality - unbiased main effect estimates',
        `Resolution ${resolution} ensures ${resolution === 'V' ? 'no main effects or 2-way interactions are confounded' : 'main effects clear of 2-way interactions'}`,
        'Highly efficient for screening important factors',
        'Can estimate main effects and selected interactions',
        'Well-studied design with established analysis methods',
        'Cost-effective for industrial experiments',
        'Can be augmented to full factorial if needed'
      ],
      cons: [
        'Some higher-order interactions are confounded (aliased)',
        'Cannot estimate all two-factor interactions independently',
        'Requires careful selection of fraction to minimize confounding',
        'May need follow-up experiments to resolve aliases',
        'Assumes higher-order interactions are negligible',
        'Less information per run compared to full factorial'
      ],
      when_to_use: `Ideal for ${nFactors}-factor screening studies when budget is limited but you still want orthogonal estimates. Use when two-factor interactions are expected to be modest, and main effects are of primary interest. Perfect for initial exploration before committing to full factorial.`,
      practical_considerations: [
        `Requires ${fractionalRuns} runs - 50% reduction from full factorial`,
        'Choose the principal fraction to minimize aliasing of important effects',
        'Add 3-5 center points to assess curvature and provide pure error estimate',
        'Document the alias structure for proper interpretation',
        'Plan for potential follow-up fold-over experiments',
        'Suitable for sequential experimentation strategies'
      ],
      resource_requirements: {
        time: 'Short to Medium',
        budget: 'Low to Medium',
        expertise: 'Intermediate'
      }
    })
  }

  // Plackett-Burman Design
  if (nFactors >= 4 && nFactors <= 12) {
    designs.push({
      design_code: 'plackett-burman',
      type: 'Plackett-Burman Design',
      description: 'Ultra-efficient screening design. Identifies important factors with minimum runs.',
      best_for: 'Rapid screening of many factors with severe budget constraints',
      runs: plackettBurmanRuns,
      score: goal === 'screening' && nFactors >= 6 ? 95 : goal === 'screening' ? 85 : 65,
      cost_rating: 'Very Low',
      efficiency_rating: 'Maximum',
      robustness_rating: 'Good',
      properties: {
        orthogonal: true,
        rotatable: false,
        three_level: false,
        requires_center_points: false,
        estimates_all_interactions: false,
        resolution: 'III',
      },
      pros: [
        `Extremely efficient - only ${plackettBurmanRuns} runs for ${nFactors} factors`,
        'Orthogonal design ensures unbiased main effect estimates',
        'Ideal for screening large numbers of factors quickly',
        'Minimizes experimental cost and time',
        'Well-suited for industrial R&D environments',
        'Can handle up to k-1 factors where k is multiple of 4',
        'Perfect for narrowing down factor list before detailed study',
        'Proven track record in manufacturing and process industries'
      ],
      cons: [
        'Resolution III: main effects confounded with two-factor interactions',
        'Cannot estimate any interaction effects',
        'Assumes all interactions are negligible (strong effect sparsity)',
        'Complex alias structure requires careful interpretation',
        'Not suitable if interactions are expected to be important',
        'Follow-up experiments usually required',
        'Limited ability to model response surface'
      ],
      when_to_use: `Use for initial screening when you have many (${nFactors}+) factors and need to quickly identify the vital few. Best when you're confident that most factors will have negligible effects, and interactions are unlikely. Excellent for preliminary studies before committing to more detailed designs.`,
      practical_considerations: [
        `Requires only ${plackettBurmanRuns} runs - extremely economical`,
        'Best used as first stage in sequential experimentation',
        'Document which factors are active before follow-up studies',
        'Consider using effect sparsity principle (only few factors matter)',
        'Plan for follow-up factorial or RSM on significant factors',
        'Can combine with foldover to break aliases if needed',
        'Most effective when combined with engineering knowledge'
      ],
      resource_requirements: {
        time: 'Very Short',
        budget: 'Very Low',
        expertise: 'Intermediate'
      }
    })
  }

  // CCD - Face-Centered
  designs.push({
    design_code: 'face-centered',
    type: 'Central Composite Design (Face-Centered)',
    description: 'Response Surface Method design with axial points on the faces of the design cube. Excellent balance of efficiency and performance.',
    best_for: 'Response surface modeling with three-level factors',
    runs: ccdFaceCenteredRuns,
    score: goal === 'modeling' ? 95 : goal === 'optimization' ? 92 : 82,
    cost_rating: nFactors <= 3 ? 'Medium' : nFactors <= 4 ? 'High' : 'Very High',
    efficiency_rating: 'Very Good',
    robustness_rating: 'Excellent',
    properties: {
      orthogonal: true,
      rotatable: false,
      three_level: true,
      requires_center_points: true,
      estimates_quadratic_effects: true,
      resolution: 'Full',
    },
    pros: [
      'Fits full quadratic model (main effects, interactions, quadratic terms)',
      'All factor levels stay within experimental region (-1 to +1)',
      'No need to extrapolate beyond tested conditions',
      'Three-level design allows curvature detection',
      'Orthogonal blocking is possible',
      'Estimates optimal operating conditions',
      'Well-suited for sequential experimentation',
      'Can build on previous factorial experiments',
      'Widely used and accepted in industry',
      'Excellent for process optimization'
    ],
    cons: [
      `Requires ${ccdFaceCenteredRuns} runs (more than factorial)`,
      'Not rotatable - prediction variance varies with direction',
      'Requires more experimental resources than screening designs',
      'Assumes second-order model is adequate',
      'May not be suitable if true optimum is at extreme levels',
      'Center points needed (adds runs but provides valuable information)'
    ],
    when_to_use: `Ideal for response surface modeling and optimization when factors must stay within tested ranges. Use when you've completed screening and identified ${nFactors} important factors to optimize. Perfect when you need quadratic model but want to avoid extrapolation risks. Best choice for confined experimental regions.`,
    practical_considerations: [
      `Requires ${ccdFaceCenteredRuns} runs: ${fullFactorialRuns} factorial + ${2*nFactors} axial + 4 center points`,
      'Can augment existing 2^k factorial by adding axial and center points',
      'All points at ±1 or 0 - easy to implement practically',
      'Use 4-6 center points for pure error estimation and curvature detection',
      'Consider blocking if runs must be split across batches',
      'Provides foundation for steepest ascent/descent methods',
      'Response surface plots clearly show optimal regions'
    ],
    resource_requirements: {
      time: nFactors <= 3 ? 'Medium' : 'Long',
      budget: nFactors <= 3 ? 'Medium' : 'High',
      expertise: 'Intermediate to Advanced'
    }
  })

  // CCD - Rotatable
  if (nFactors <= 5) {
    const alpha = Math.pow(2, nFactors / 4).toFixed(2)
    designs.push({
      design_code: 'rotatable',
      type: 'Central Composite Design (Rotatable)',
      description: `Rotatable RSM design with axial points at α = ±${alpha}. Provides uniform prediction variance in all directions.`,
      best_for: 'Response surface modeling with equal precision in all directions',
      runs: ccdRotatableRuns,
      score: goal === 'modeling' ? 90 : goal === 'optimization' ? 93 : 80,
      cost_rating: nFactors <= 3 ? 'Medium' : nFactors <= 4 ? 'High' : 'Very High',
      efficiency_rating: 'Excellent',
      robustness_rating: 'Excellent',
      properties: {
        orthogonal: true,
        rotatable: true,
        three_level: false,
        requires_center_points: true,
        estimates_quadratic_effects: true,
        alpha_value: parseFloat(alpha),
        resolution: 'Full',
      },
      pros: [
        'Rotatable: uniform prediction variance in all radial directions',
        'Optimal for locating true optimum regardless of direction',
        'Excellent statistical properties for response surface analysis',
        'Fits full second-order model with interactions',
        'Orthogonal design minimizes correlation between coefficient estimates',
        'Well-established methodology with extensive literature',
        'Provides reliable predictions throughout design space',
        'Ideal for unknown optimal region location',
        'Supports contour plots and response surface visualization',
        'Can identify saddle points and multiple optima'
      ],
      cons: [
        `Requires ${ccdRotatableRuns} runs (slightly more than face-centered)`,
        `Axial points at α = ${alpha} may be outside practical operating range`,
        'May require extrapolation beyond feasible factor levels',
        'Some factor combinations might be difficult to achieve',
        'More complex to set up than face-centered design',
        'Rotatability property may not be critical for all applications',
        'Additional center points increase experimental burden'
      ],
      when_to_use: `Use when direction of optimal response is unknown and you need equal precision in all directions. Ideal when ${nFactors}-factor system optimum could be anywhere in design space. Choose when statistical properties matter more than practical constraints. Best for fundamental research and method development.`,
      practical_considerations: [
        `Requires ${ccdRotatableRuns} runs: ${fullFactorialRuns} cube + ${2*nFactors} axial (at ±${alpha}) + 6 center`,
        `Axial points at ±${alpha} may require extrapolation - verify feasibility`,
        'Check that all factor combinations are physically achievable',
        'More center points (6) provide better estimate of pure error',
        'Rotatability ensures consistent prediction quality',
        'Particularly valuable when optimum location is uncertain',
        'Consider if practical constraints limit factor ranges'
      ],
      resource_requirements: {
        time: nFactors <= 3 ? 'Medium' : 'Long',
        budget: nFactors <= 3 ? 'Medium' : 'High',
        expertise: 'Advanced'
      }
    })
  }

  // Box-Behnken Design
  if (nFactors >= 3 && nFactors <= 6 && boxBehnkenRuns > 0) {
    designs.push({
      design_code: 'box-behnken',
      type: 'Box-Behnken Design',
      description: 'Efficient three-level response surface design. All points on edges and center - no corner points.',
      best_for: 'RSM when corner points are impractical or you need fewer runs',
      runs: boxBehnkenRuns,
      score: goal === 'optimization' ? 88 : goal === 'modeling' ? 87 : 78,
      cost_rating: nFactors <= 4 ? 'Medium' : 'High',
      efficiency_rating: 'Very Good',
      robustness_rating: 'Good',
      properties: {
        orthogonal: true,
        rotatable: true,
        three_level: true,
        requires_center_points: true,
        estimates_quadratic_effects: true,
        spherical_design: true,
        resolution: 'Full',
      },
      pros: [
        `Requires fewer runs than CCD (${boxBehnkenRuns} vs ${ccdFaceCenteredRuns})`,
        'Rotatable design - equal prediction precision in all directions',
        'Never tests all factors at extreme levels simultaneously',
        'Avoids potentially problematic corner point combinations',
        'Three-level design detects curvature effectively',
        'Efficient for fitting second-order models',
        'Spherical design space is statistically elegant',
        'Good for process optimization with safety constraints',
        'Well-suited for constrained experimental regions',
        'Orthogonal blocks available for some configurations'
      ],
      cons: [
        'Cannot estimate third-order and higher interactions',
        'Less intuitive structure than CCD',
        'Not suitable for sequential experimentation (cannot augment factorial)',
        'Poor prediction at corners (not tested)',
        'Requires all factors at three levels from start',
        'May be less efficient than CCD for some factor counts',
        'Center points required for model adequacy',
        'Cannot leverage existing 2^k factorial data'
      ],
      when_to_use: `Best for ${nFactors}-factor optimization when corner points are unsafe, impossible, or extremely expensive. Use when you want rotatable design but need fewer runs than rotatable CCD. Ideal when extreme factor combinations should be avoided (safety, equipment limits, cost).`,
      practical_considerations: [
        `Requires ${boxBehnkenRuns} runs - fewer than comparable CCD`,
        'All factor combinations are at edge midpoints or center',
        'No combinations test all factors at their extremes',
        'Add 3-5 center point replicates for pure error and curvature check',
        'Good choice when factorial base is not available',
        'Particularly suitable when corner points are problematic',
        'Consider if sequential build-up from screening is not planned'
      ],
      resource_requirements: {
        time: 'Medium',
        budget: 'Medium',
        expertise: 'Advanced'
      }
    })
  }

  // Definitive Screening Design
  if (nFactors >= 4 && nFactors <= 10 && dsdRuns > 0) {
    designs.push({
      design_code: 'dsd',
      type: 'Definitive Screening Design (DSD)',
      description: 'Modern three-level screening design. Efficiently screens factors AND detects curvature with minimal runs.',
      best_for: 'Efficient screening that also detects quadratic effects',
      runs: dsdRuns,
      score: goal === 'screening' ? 93 : goal === 'modeling' ? 85 : 88,
      cost_rating: 'Low',
      efficiency_rating: 'Excellent',
      robustness_rating: 'Very Good',
      properties: {
        orthogonal: false,
        rotatable: false,
        three_level: true,
        requires_center_points: false,
        estimates_quadratic_effects: true,
        estimates_some_interactions: true,
        resolution: 'III+',
      },
      pros: [
        `Very efficient: only ${dsdRuns} runs for ${nFactors} factors`,
        'Three-level design allows simultaneous screening and curvature detection',
        'Can estimate main effects clear of two-factor interactions',
        'Some two-factor interactions estimable if effect sparsity holds',
        'Suitable for both screening and optimization',
        'Modern design gaining popularity in industry',
        'Reduces need for sequential experiments',
        'Excellent for systems with both linear and quadratic effects',
        'No confounding of main effects with each other',
        'Can identify important interactions for follow-up'
      ],
      cons: [
        'Not fully orthogonal - some correlation between estimates',
        'Requires effect sparsity assumption for interaction estimation',
        'Cannot estimate all two-factor interactions',
        'Relatively new - less established than classical designs',
        'Analysis can be more complex than traditional designs',
        'May require specialized software for optimal analysis',
        'No center points to estimate pure error',
        'Prediction variance higher than response surface designs'
      ],
      when_to_use: `Excellent choice when you need to screen ${nFactors} factors efficiently but also want to detect curvature. Use when effect sparsity is reasonable (only few factors/interactions important). Ideal bridge between screening and optimization - can serve both purposes with one design.`,
      practical_considerations: [
        `Requires only ${dsdRuns} runs - very economical for ${nFactors} factors`,
        'Three levels per factor must be achievable',
        'Best when combined with engineering judgment for interpretation',
        'Follow-up experiments may be needed for detailed optimization',
        'Modern alternative to Plackett-Burman + augmentation',
        'Particularly valuable when experimental budget is severely limited',
        'Consider when you want screening + RSM capabilities in one design'
      ],
      resource_requirements: {
        time: 'Short',
        budget: 'Low',
        expertise: 'Advanced'
      }
    })
  }

  // Optimal Designs (D-optimal)
  if (nFactors >= 3 && budget && budget < ccdFaceCenteredRuns) {
    designs.push({
      design_code: 'optimal-design',
      type: 'Optimal Design (Computer-Generated)',
      description: 'Custom-tailored design optimized for your specific constraints. Maximizes information for your exact situation.',
      best_for: 'Situations with unusual constraints or specific budget limitations',
      runs: budget,
      score: 75,
      cost_rating: 'Custom',
      efficiency_rating: 'Very Good',
      robustness_rating: 'Good',
      properties: {
        orthogonal: false,
        rotatable: false,
        three_level: true,
        requires_center_points: false,
        custom_constraints: true,
        model_based: true,
      },
      pros: [
        'Tailored to your exact budget and constraints',
        'Maximizes statistical efficiency for specified model',
        'Can handle irregular experimental regions',
        'Accommodates prohibited factor combinations',
        'Flexible - works with any model complexity',
        'Can incorporate prior knowledge',
        'Efficient use of limited resources',
        'Suitable for constrained factor spaces'
      ],
      cons: [
        'Requires specialized software for generation',
        'Not orthogonal - coefficient estimates correlated',
        'May be model-dependent (different models need different designs)',
        'Less established than classical designs',
        'Harder to explain and document',
        'No "cookbook" approach - requires expertise',
        'Replicability across labs may be challenging',
        'May not generalize well if model is misspecified'
      ],
      when_to_use: `Consider when classical designs don't fit your constraints or budget. Use when you have ${budget} runs available and need maximum efficiency. Best when experimental region has unusual shape or forbidden combinations. Requires expertise to generate and analyze properly.`,
      practical_considerations: [
        `Uses your specified budget of ${budget} runs`,
        'Requires design generation software (MasterStat, Design-Expert, R)',
        'Document design generation criteria and assumptions',
        'Consider robustness to model misspecification',
        'May need expert statistical consultation',
        'Verify design adequacy before experimentation'
      ],
      resource_requirements: {
        time: 'Varies',
        budget: 'Custom',
        expertise: 'Expert'
      }
    })
  }

  return designs
}

// Helper functions
const formatKey = (key) => {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

const formatValue = (value) => {
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'number') return value.toFixed(2)
  return value.toString()
}

// Design Comparison Component
const DesignComparison = ({ designs, onExit, onSelectDesign, selectedDesign }) => {
  const metrics = [
    { key: 'runs', label: 'Experimental Runs', lower_better: true },
    { key: 'score', label: 'Overall Score', lower_better: false },
    { key: 'cost_rating', label: 'Cost Rating', format: 'text' },
    { key: 'efficiency_rating', label: 'Efficiency Rating', format: 'text' },
    { key: 'robustness_rating', label: 'Robustness Rating', format: 'text' }
  ]

  const getNumericValue = (value, key) => {
    if (key === 'cost_rating') {
      return value === 'Low' ? 33 : value === 'Medium' ? 66 : 100
    }
    if (key === 'efficiency_rating' || key === 'robustness_rating') {
      return value === 'Excellent' ? 100 : value === 'Good' ? 75 : value === 'Fair' ? 50 : 25
    }
    return value
  }

  const getColorForMetric = (value, key) => {
    if (key === 'cost_rating') {
      return value === 'Low' ? 'bg-green-500' : value === 'Medium' ? 'bg-yellow-500' : 'bg-red-500'
    }
    if (key === 'efficiency_rating' || key === 'robustness_rating') {
      return value === 'Excellent' ? 'bg-green-500' : value === 'Good' ? 'bg-blue-500' : 'bg-yellow-500'
    }
    return 'bg-blue-500'
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-2xl font-bold text-gray-100 mb-1 flex items-center gap-2">
            <BarChart3 className="w-7 h-7 text-blue-400" />
            Design Comparison
          </h3>
          <p className="text-gray-300 text-sm">
            Comparing {designs.length} experimental designs side-by-side
          </p>
        </div>
        <button
          onClick={onExit}
          className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-gray-200 rounded-lg font-semibold transition-all"
        >
          <X className="w-5 h-5" />
          Exit Comparison
        </button>
      </div>

      {/* Comparison Table */}
      <div className="bg-slate-800/50 border border-slate-700 rounded-xl overflow-hidden mb-6">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-600 bg-slate-900/50">
                <th className="text-left py-4 px-4 text-gray-300 font-semibold w-48">Metric</th>
                {designs.map((design, idx) => (
                  <th key={idx} className="text-center py-4 px-4 text-gray-100 font-semibold min-w-[200px]">
                    <div className="space-y-1">
                      <div className="text-sm font-normal text-gray-400">Design {idx + 1}</div>
                      <div className="text-base">{design.type}</div>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {/* Basic Info */}
              <tr className="border-b border-slate-700/50 bg-slate-800/30">
                <td className="py-3 px-4 text-gray-300 font-medium">Design Type</td>
                {designs.map((design, idx) => (
                  <td key={idx} className="py-3 px-4 text-center text-gray-100 text-sm">
                    {design.type}
                  </td>
                ))}
              </tr>

              {/* Metrics with visual bars */}
              {metrics.map((metric) => (
                <tr key={metric.key} className="border-b border-slate-700/50 hover:bg-slate-700/20">
                  <td className="py-3 px-4 text-gray-300 font-medium">{metric.label}</td>
                  {designs.map((design, idx) => {
                    const value = design[metric.key]
                    const numericValue = getNumericValue(value, metric.key)
                    const maxValue = Math.max(...designs.map(d => getNumericValue(d[metric.key], metric.key)))
                    const percentage = (numericValue / maxValue) * 100

                    return (
                      <td key={idx} className="py-3 px-4">
                        <div className="space-y-1">
                          <div className="text-center text-gray-100 font-semibold text-sm">
                            {value}
                          </div>
                          {metric.format !== 'text' && (
                            <div className="w-full bg-slate-700 rounded-full h-2">
                              <div
                                className={`${getColorForMetric(value, metric.key)} rounded-full h-2 transition-all`}
                                style={{ width: `${percentage}%` }}
                              />
                            </div>
                          )}
                        </div>
                      </td>
                    )
                  })}
                </tr>
              ))}

              {/* Properties */}
              <tr className="border-b border-slate-700/50 bg-slate-800/30">
                <td className="py-3 px-4 text-gray-300 font-medium">Key Properties</td>
                {designs.map((design, idx) => (
                  <td key={idx} className="py-3 px-4">
                    <div className="flex flex-wrap gap-1 justify-center">
                      {design.properties.orthogonal && (
                        <span className="px-2 py-1 bg-green-900/30 border border-green-700/50 rounded text-green-300 text-xs">
                          ✓ Orthogonal
                        </span>
                      )}
                      {design.properties.rotatable && (
                        <span className="px-2 py-1 bg-blue-900/30 border border-blue-700/50 rounded text-blue-300 text-xs">
                          ✓ Rotatable
                        </span>
                      )}
                      {design.properties.three_level && (
                        <span className="px-2 py-1 bg-purple-900/30 border border-purple-700/50 rounded text-purple-300 text-xs">
                          3-Level
                        </span>
                      )}
                    </div>
                  </td>
                ))}
              </tr>

              {/* Pros/Cons Count */}
              <tr className="border-b border-slate-700/50">
                <td className="py-3 px-4 text-gray-300 font-medium">Advantages</td>
                {designs.map((design, idx) => (
                  <td key={idx} className="py-3 px-4 text-center">
                    <span className="text-green-400 font-semibold">{design.pros.length} pros</span>
                  </td>
                ))}
              </tr>
              <tr className="border-b border-slate-700/50">
                <td className="py-3 px-4 text-gray-300 font-medium">Disadvantages</td>
                {designs.map((design, idx) => (
                  <td key={idx} className="py-3 px-4 text-center">
                    <span className="text-red-400 font-semibold">{design.cons.length} cons</span>
                  </td>
                ))}
              </tr>

              {/* Action Row */}
              <tr className="bg-slate-900/50">
                <td className="py-4 px-4 text-gray-300 font-medium">Select Design</td>
                {designs.map((design, idx) => (
                  <td key={idx} className="py-4 px-4 text-center">
                    <button
                      onClick={() => {
                        onSelectDesign(design)
                        onExit()
                      }}
                      className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                        selectedDesign?.design_code === design.design_code
                          ? 'bg-green-600 text-white'
                          : 'bg-blue-600 hover:bg-blue-700 text-white'
                      }`}
                    >
                      {selectedDesign?.design_code === design.design_code ? (
                        <>
                          <Check className="w-4 h-4 inline mr-1" />
                          Selected
                        </>
                      ) : (
                        'Select This'
                      )}
                    </button>
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Trade-off Analysis */}
      <div className="bg-gradient-to-r from-blue-900/20 to-purple-900/20 border border-blue-700/50 rounded-xl p-6 mb-6">
        <h4 className="text-lg font-bold text-blue-200 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          Trade-off Analysis
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-800/50 rounded-lg p-4">
            <h5 className="font-semibold text-gray-200 mb-2 flex items-center gap-2">
              <DollarSign className="w-4 h-4 text-yellow-400" />
              Cost vs. Power
            </h5>
            <p className="text-sm text-gray-300">
              Designs with fewer runs are cheaper but may have less statistical power.
              Consider your budget constraints and required precision.
            </p>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-4">
            <h5 className="font-semibold text-gray-200 mb-2 flex items-center gap-2">
              <Clock className="w-4 h-4 text-blue-400" />
              Efficiency vs. Complexity
            </h5>
            <p className="text-sm text-gray-300">
              Simpler designs are easier to execute but may not capture all interactions.
              Complex designs provide more information but require careful planning.
            </p>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-4">
            <h5 className="font-semibold text-gray-200 mb-2 flex items-center gap-2">
              <Shield className="w-4 h-4 text-green-400" />
              Robustness vs. Precision
            </h5>
            <p className="text-sm text-gray-300">
              Robust designs protect against model misspecification but may require more runs.
              Precise designs give exact estimates under the right model assumptions.
            </p>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-4">
            <h5 className="font-semibold text-gray-200 mb-2 flex items-center gap-2">
              <Target className="w-4 h-4 text-purple-400" />
              Coverage vs. Resources
            </h5>
            <p className="text-sm text-gray-300">
              Full factorial explores all combinations but becomes expensive with many factors.
              Fractional designs reduce cost while maintaining critical information.
            </p>
          </div>
        </div>
      </div>

      {/* Recommendation */}
      <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-5">
        <h4 className="font-bold text-green-200 mb-2 flex items-center gap-2">
          <Star className="w-5 h-5" />
          Expert Recommendation
        </h4>
        <p className="text-green-100 text-sm">
          Based on your comparison, <strong>{designs[0].type}</strong> appears to be the strongest choice with a score of {designs[0].score}/100.
          However, consider your specific constraints (budget, time, equipment) when making the final decision.
          All displayed designs are statistically valid for your {designs[0].runs}-factor experiment.
        </p>
      </div>
    </div>
  )
}

export default DesignRecommendationStep
