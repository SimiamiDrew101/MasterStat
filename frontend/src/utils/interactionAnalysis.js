// Factor Interaction Analysis Utility
// Analyzes which experimental designs can estimate which factor interactions

/**
 * Determines which 2-way interactions a design can estimate clearly
 * Based on design type, resolution, and confounding patterns
 */
export const getEstimableInteractions = (designCode, nFactors) => {
  const interactions = []

  // Generate all possible 2-way interactions
  for (let i = 0; i < nFactors; i++) {
    for (let j = i + 1; j < nFactors; j++) {
      interactions.push({ factor1: i, factor2: j, estimable: false, confounded: false })
    }
  }

  // Determine estimability based on design type
  switch (designCode) {
    case 'full-factorial':
      // Full factorial can estimate ALL 2-way interactions clearly
      return interactions.map(int => ({ ...int, estimable: true, confounded: false }))

    case 'ccd-face':
    case 'ccd-rotatable':
    case 'box-behnken':
      // RSM designs can estimate all 2-way interactions
      return interactions.map(int => ({ ...int, estimable: true, confounded: false }))

    case 'fractional-factorial':
      // Resolution depends on fraction size
      const resolution = getResolution(nFactors)
      if (resolution === 'V') {
        // Resolution V: All 2-way interactions estimable (clear of main effects and each other)
        return interactions.map(int => ({ ...int, estimable: true, confounded: false }))
      } else if (resolution === 'IV') {
        // Resolution IV: 2-way interactions confounded with each other
        // Can estimate, but with aliasing
        return interactions.map(int => ({ ...int, estimable: true, confounded: true }))
      } else {
        // Resolution III: 2-way interactions confounded with main effects (not estimable reliably)
        return interactions.map(int => ({ ...int, estimable: false, confounded: true }))
      }

    case 'plackett-burman':
      // Plackett-Burman: Cannot estimate any interactions
      return interactions.map(int => ({ ...int, estimable: false, confounded: true }))

    case 'screening-first':
    case 'dsd':
      // Sequential approaches: Limited interaction estimation
      return interactions.map(int => ({ ...int, estimable: false, confounded: true }))

    default:
      // Unknown design: Conservative assumption
      return interactions.map(int => ({ ...int, estimable: false, confounded: false }))
  }
}

/**
 * Get resolution for fractional factorial designs
 */
const getResolution = (nFactors) => {
  if (nFactors <= 5) return 'V'  // Resolution V for <= 5 factors
  if (nFactors <= 7) return 'IV' // Resolution IV for 6-7 factors
  return 'III'                    // Resolution III for 8+ factors
}

/**
 * Calculate interaction coverage score
 * Higher score = better coverage of user-specified interactions
 */
export const calculateInteractionCoverage = (designCode, nFactors, specifiedInteractions) => {
  if (!specifiedInteractions || specifiedInteractions.length === 0) {
    return 1.0 // No interactions specified, all designs equal
  }

  const estimable = getEstimableInteractions(designCode, nFactors)

  // Count how many specified interactions can be estimated clearly
  let clearCount = 0
  let confoundedCount = 0
  let notEstimableCount = 0

  specifiedInteractions.forEach(specified => {
    const interaction = estimable.find(
      e => (e.factor1 === specified.factor1 && e.factor2 === specified.factor2) ||
           (e.factor1 === specified.factor2 && e.factor2 === specified.factor1)
    )

    if (interaction) {
      if (interaction.estimable && !interaction.confounded) {
        clearCount++
      } else if (interaction.estimable && interaction.confounded) {
        confoundedCount++
      } else {
        notEstimableCount++
      }
    }
  })

  const total = specifiedInteractions.length

  // Scoring:
  // - Clear estimation: 1.0 point
  // - Confounded but estimable: 0.4 points
  // - Not estimable: 0.0 points
  const score = (clearCount * 1.0 + confoundedCount * 0.4) / total

  return score
}

/**
 * Get interaction capability summary for a design
 */
export const getInteractionCapability = (designCode, nFactors) => {
  const interactions = getEstimableInteractions(designCode, nFactors)

  const clear = interactions.filter(i => i.estimable && !i.confounded).length
  const confounded = interactions.filter(i => i.estimable && i.confounded).length
  const notEstimable = interactions.filter(i => !i.estimable).length
  const total = interactions.length

  let capability = 'none'
  let description = ''
  let color = 'red'

  if (clear === total) {
    capability = 'full'
    description = `Can estimate all ${total} interactions clearly`
    color = 'green'
  } else if (clear > 0) {
    capability = 'partial-clear'
    description = `Can estimate ${clear}/${total} interactions clearly`
    color = 'blue'
  } else if (confounded > 0) {
    capability = 'partial-confounded'
    description = `Can estimate ${confounded}/${total} interactions (with confounding)`
    color = 'yellow'
  } else {
    capability = 'none'
    description = 'Cannot estimate interactions'
    color = 'red'
  }

  return {
    capability,
    description,
    color,
    clear,
    confounded,
    notEstimable,
    total
  }
}

/**
 * Check if a specific interaction can be estimated by a design
 */
export const canEstimateInteraction = (designCode, nFactors, factor1, factor2) => {
  const interactions = getEstimableInteractions(designCode, nFactors)
  const interaction = interactions.find(
    i => (i.factor1 === factor1 && i.factor2 === factor2) ||
         (i.factor1 === factor2 && i.factor2 === factor1)
  )
  return interaction ? { estimable: interaction.estimable, confounded: interaction.confounded } : { estimable: false, confounded: false }
}

/**
 * Get recommended designs for specific interactions
 */
export const getRecommendedDesignsForInteractions = (nFactors, specifiedInteractions) => {
  const designs = ['full-factorial', 'fractional-factorial', 'ccd-face', 'ccd-rotatable', 'box-behnken', 'plackett-burman']

  const scores = designs.map(designCode => ({
    designCode,
    score: calculateInteractionCoverage(designCode, nFactors, specifiedInteractions),
    capability: getInteractionCapability(designCode, nFactors)
  }))

  // Sort by score descending
  scores.sort((a, b) => b.score - a.score)

  return scores
}

/**
 * Generate interaction recommendations based on goal and factors
 */
export const getInteractionRecommendations = (nFactors, goal) => {
  const recommendations = []

  if (goal === 'screening') {
    recommendations.push({
      type: 'info',
      message: 'Screening designs typically assume interactions are negligible. If you suspect important interactions, consider a characterization or optimization goal instead.',
      action: 'Consider switching to "modeling" goal if interactions are important'
    })
  } else if (goal === 'optimization') {
    recommendations.push({
      type: 'success',
      message: 'Response Surface designs (CCD, Box-Behnken) can estimate all 2-way interactions, which is important for finding optimal settings.',
      action: 'RSM designs will capture interaction effects in your model'
    })
  } else if (goal === 'modeling') {
    recommendations.push({
      type: 'success',
      message: 'Modeling goals benefit from specifying suspected interactions. The wizard will prioritize designs that can estimate these effects.',
      action: 'Select interactions you expect based on theory or prior experience'
    })
  }

  // Factor count warnings
  if (nFactors >= 5) {
    const totalInteractions = (nFactors * (nFactors - 1)) / 2
    recommendations.push({
      type: 'warning',
      message: `With ${nFactors} factors, there are ${totalInteractions} possible 2-way interactions. Consider screening first to reduce to 3-4 factors, then study interactions.`,
      action: 'Sequential experimentation recommended'
    })
  }

  return recommendations
}

/**
 * Analyze confounding pattern for fractional factorial
 */
export const getConfoundingPattern = (nFactors) => {
  const resolution = getResolution(nFactors)

  const patterns = {
    'V': {
      resolution: 'V',
      description: 'Main effects and 2-way interactions are clear (not confounded)',
      aliasing: 'Main effects clear of 2-way interactions; 2-way interactions clear of each other',
      recommendation: 'Excellent for studying interactions',
      runs: Math.pow(2, nFactors - 1)
    },
    'IV': {
      resolution: 'IV',
      description: 'Main effects clear, but some 2-way interactions confounded with each other',
      aliasing: 'Main effects clear; some 2-way interactions aliased together (e.g., AB = CD)',
      recommendation: 'Good for main effects, acceptable for interactions with caution',
      runs: Math.pow(2, nFactors - 2)
    },
    'III': {
      resolution: 'III',
      description: 'Main effects confounded with 2-way interactions',
      aliasing: 'Main effects aliased with 2-way interactions (e.g., A = BC)',
      recommendation: 'Suitable only for screening; cannot reliably estimate interactions',
      runs: Math.pow(2, nFactors - 3)
    }
  }

  return patterns[resolution] || patterns['III']
}

/**
 * Format interaction name from factor indices
 */
export const formatInteractionName = (factor1, factor2, factorNames = []) => {
  const name1 = factorNames[factor1] || `Factor ${factor1 + 1}`
  const name2 = factorNames[factor2] || `Factor ${factor2 + 1}`
  return `${name1} × ${name2}`
}

/**
 * Generate all possible interactions for selection
 */
export const generateAllInteractions = (nFactors, factorNames = []) => {
  const interactions = []

  for (let i = 0; i < nFactors; i++) {
    for (let j = i + 1; j < nFactors; j++) {
      interactions.push({
        factor1: i,
        factor2: j,
        name: formatInteractionName(i, j, factorNames),
        id: `${i}-${j}`
      })
    }
  }

  return interactions
}
