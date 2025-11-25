// Smart Validation Engine for Experiment Wizard
// Provides intelligent warnings and recommendations to prevent DOE mistakes

/**
 * Validation severity levels
 */
export const SEVERITY = {
  ERROR: 'error',     // Blocks progress - must be fixed
  WARNING: 'warning', // Strong recommendation - can proceed but risky
  INFO: 'info',       // Helpful tip - good to know
  SUCCESS: 'success'  // Positive feedback - confirms good choice
}

/**
 * Validation categories
 */
export const CATEGORY = {
  FACTOR_COUNT: 'factor_count',
  BUDGET: 'budget',
  DESIGN_LIMITATION: 'design_limitation',
  POWER: 'power',
  EFFICIENCY: 'efficiency',
  GOAL_MISMATCH: 'goal_mismatch'
}

/**
 * Main validation function - returns array of validation messages
 */
export const validateWizardData = (wizardData) => {
  const validations = []

  // Extract wizard data
  const { goal, nFactors, budget, selectedDesign, powerAnalysis, factorNames } = wizardData

  // Validate factor count
  validations.push(...validateFactorCount(nFactors, goal, budget))

  // Validate budget constraints
  if (budget && selectedDesign) {
    validations.push(...validateBudget(budget, selectedDesign, nFactors))
  }

  // Validate design selection
  if (selectedDesign) {
    validations.push(...validateDesignChoice(selectedDesign, nFactors, goal, budget))
  }

  // Validate power analysis
  if (powerAnalysis && powerAnalysis.minimumRuns && budget) {
    validations.push(...validatePowerVsBudget(powerAnalysis, budget))
  }

  // Validate goal alignment
  if (goal && selectedDesign) {
    validations.push(...validateGoalAlignment(goal, selectedDesign, nFactors))
  }

  // Remove duplicates and sort by severity
  return deduplicateAndSort(validations)
}

/**
 * Validate factor count appropriateness
 */
const validateFactorCount = (nFactors, goal, budget) => {
  const validations = []

  // Too many factors warning
  if (nFactors >= 6 && goal !== 'screening') {
    validations.push({
      severity: SEVERITY.WARNING,
      category: CATEGORY.FACTOR_COUNT,
      title: 'High Factor Count Detected',
      message: `${nFactors} factors require many experimental runs. Consider screening to identify important factors first, then optimize with fewer factors.`,
      recommendation: 'Use screening designs (Plackett-Burman, Fractional Factorial) to reduce to 3-4 key factors',
      icon: 'AlertTriangle'
    })
  }

  // Too few factors for goal
  if (nFactors < 2) {
    validations.push({
      severity: SEVERITY.ERROR,
      category: CATEGORY.FACTOR_COUNT,
      title: 'Insufficient Factors',
      message: 'Design of Experiments requires at least 2 factors. For single-factor studies, use simpler methods like t-tests or ANOVA.',
      recommendation: 'Add at least one more factor or use single-factor analysis methods',
      icon: 'XCircle'
    })
  }

  // Optimal range feedback
  if (nFactors >= 2 && nFactors <= 4 && goal === 'optimization') {
    validations.push({
      severity: SEVERITY.SUCCESS,
      category: CATEGORY.FACTOR_COUNT,
      title: 'Ideal Factor Count',
      message: `${nFactors} factors is an excellent range for response surface methodology and optimization studies.`,
      recommendation: 'Proceed with confidence - this is a well-sized optimization study',
      icon: 'CheckCircle'
    })
  }

  // Screening with few factors
  if (nFactors <= 3 && goal === 'screening') {
    validations.push({
      severity: SEVERITY.INFO,
      category: CATEGORY.FACTOR_COUNT,
      title: 'Consider Full Factorial',
      message: `With only ${nFactors} factors, a full factorial design may be more informative than screening designs.`,
      recommendation: 'Full factorial provides complete information with minimal runs for 3 or fewer factors',
      icon: 'Info'
    })
  }

  // Budget too small for factor count
  if (budget && nFactors >= 5) {
    const minRunsEstimate = Math.pow(2, Math.min(nFactors - 1, 5))
    if (budget < minRunsEstimate) {
      validations.push({
        severity: SEVERITY.ERROR,
        category: CATEGORY.BUDGET,
        title: 'Budget Too Low for Factor Count',
        message: `${nFactors} factors require at least ${minRunsEstimate} runs for fractional factorial screening. Your budget of ${budget} runs is insufficient.`,
        recommendation: `Increase budget to at least ${minRunsEstimate} runs or reduce to ${nFactors - 1} factors`,
        icon: 'DollarSign'
      })
    }
  }

  return validations
}

/**
 * Validate budget constraints
 */
const validateBudget = (budget, selectedDesign, nFactors) => {
  const validations = []
  const requiredRuns = selectedDesign.runs

  // Budget insufficient for selected design
  if (budget < requiredRuns) {
    const difference = requiredRuns - budget
    validations.push({
      severity: SEVERITY.ERROR,
      category: CATEGORY.BUDGET,
      title: 'Budget Exceeds Available Runs',
      message: `${selectedDesign.type} requires ${requiredRuns} runs, but your budget is only ${budget} runs (${difference} runs short).`,
      recommendation: `Either increase budget to ${requiredRuns} runs or select a more efficient design`,
      icon: 'AlertCircle'
    })
  }

  // Budget barely sufficient (within 10%)
  else if (budget >= requiredRuns && budget < requiredRuns * 1.1) {
    validations.push({
      severity: SEVERITY.SUCCESS,
      category: CATEGORY.BUDGET,
      title: 'Budget Perfectly Matched',
      message: `Your budget of ${budget} runs perfectly accommodates the ${requiredRuns} runs needed for ${selectedDesign.type}.`,
      recommendation: 'Excellent match - proceed with this design',
      icon: 'CheckCircle'
    })
  }

  // Budget has significant excess (>30% unused)
  else if (budget > requiredRuns * 1.3) {
    const excess = budget - requiredRuns
    validations.push({
      severity: SEVERITY.INFO,
      category: CATEGORY.BUDGET,
      title: 'Budget Underutilized',
      message: `Your budget allows ${budget} runs, but ${selectedDesign.type} only needs ${requiredRuns} runs (${excess} runs unused).`,
      recommendation: 'Consider adding center point replicates for better error estimation or exploring more comprehensive designs',
      icon: 'TrendingUp'
    })
  }

  // Very low budget warning
  if (budget < 10 && nFactors >= 3) {
    validations.push({
      severity: SEVERITY.WARNING,
      category: CATEGORY.BUDGET,
      title: 'Limited Budget May Restrict Analysis',
      message: `With only ${budget} runs for ${nFactors} factors, your ability to detect interactions and curvature will be limited.`,
      recommendation: 'If possible, increase budget to at least 15-20 runs for more robust analysis',
      icon: 'AlertTriangle'
    })
  }

  return validations
}

/**
 * Validate design choice appropriateness
 */
const validateDesignChoice = (selectedDesign, nFactors, goal, budget) => {
  const validations = []
  const designType = selectedDesign.type.toLowerCase()

  // Box-Behnken warnings
  if (designType.includes('box-behnken')) {
    validations.push({
      severity: SEVERITY.WARNING,
      category: CATEGORY.DESIGN_LIMITATION,
      title: 'Box-Behnken Avoids Extreme Corners',
      message: 'Box-Behnken designs do not test extreme factor combinations (all factors at high or low simultaneously). The true optimum may be at these untested corners.',
      recommendation: 'If extreme conditions are feasible and safe, consider Central Composite Design for complete coverage',
      icon: 'AlertTriangle'
    })

    if (nFactors === 2) {
      validations.push({
        severity: SEVERITY.ERROR,
        category: CATEGORY.DESIGN_LIMITATION,
        title: 'Box-Behnken Invalid for 2 Factors',
        message: 'Box-Behnken designs require at least 3 factors. For 2 factors, use Central Composite or Full Factorial designs.',
        recommendation: 'Switch to Central Composite Design (CCD) or Full Factorial',
        icon: 'XCircle'
      })
    }
  }

  // Full Factorial warnings
  if (designType.includes('full factorial')) {
    const runs = Math.pow(2, nFactors)
    if (nFactors >= 5) {
      validations.push({
        severity: SEVERITY.WARNING,
        category: CATEGORY.EFFICIENCY,
        title: 'Full Factorial May Be Inefficient',
        message: `Full factorial with ${nFactors} factors requires ${runs} runs. Fractional factorial or screening designs can provide similar insights with fewer runs.`,
        recommendation: `Consider fractional factorial (${Math.pow(2, nFactors - 1)} runs) or Plackett-Burman for initial screening`,
        icon: 'Zap'
      })
    }

    if (goal === 'optimization' && nFactors >= 3) {
      validations.push({
        severity: SEVERITY.INFO,
        category: CATEGORY.DESIGN_LIMITATION,
        title: 'Full Factorial Lacks Curvature Estimation',
        message: 'Full factorial designs (2-level) cannot detect quadratic effects or estimate optimal settings. They only test corners.',
        recommendation: 'For optimization, consider Central Composite Design which includes center and axial points',
        icon: 'Info'
      })
    }
  }

  // CCD validations
  if (designType.includes('central composite') || designType.includes('ccd')) {
    if (goal === 'screening') {
      validations.push({
        severity: SEVERITY.WARNING,
        category: CATEGORY.GOAL_MISMATCH,
        title: 'CCD Not Ideal for Screening',
        message: 'Central Composite Designs are excellent for optimization but inefficient for screening many factors.',
        recommendation: 'For screening, use Plackett-Burman or Fractional Factorial designs first',
        icon: 'Target'
      })
    } else if (goal === 'optimization') {
      validations.push({
        severity: SEVERITY.SUCCESS,
        category: CATEGORY.GOAL_MISMATCH,
        title: 'Excellent Choice for Optimization',
        message: 'Central Composite Design is the gold standard for response surface methodology and finding optimal factor settings.',
        recommendation: 'Proceed with confidence - CCD provides quadratic modeling and optimization capabilities',
        icon: 'Award'
      })
    }
  }

  // Fractional Factorial warnings
  if (designType.includes('fractional factorial')) {
    validations.push({
      severity: SEVERITY.INFO,
      category: CATEGORY.DESIGN_LIMITATION,
      title: 'Fractional Factorial Has Confounding',
      message: 'Fractional factorial designs confound some effects (aliasing). Main effects may be confounded with two-way interactions.',
      recommendation: 'Document the confounding pattern and use follow-up experiments if needed',
      icon: 'AlertCircle'
    })

    if (goal === 'optimization') {
      validations.push({
        severity: SEVERITY.WARNING,
        category: CATEGORY.GOAL_MISMATCH,
        title: 'Fractional Factorial Not for Optimization',
        message: 'Fractional factorial designs are for screening, not optimization. They lack the structure to find optimal settings.',
        recommendation: 'After screening, use Central Composite or Box-Behnken for optimization',
        icon: 'Target'
      })
    }
  }

  // Plackett-Burman validations
  if (designType.includes('plackett-burman')) {
    validations.push({
      severity: SEVERITY.INFO,
      category: CATEGORY.DESIGN_LIMITATION,
      title: 'Plackett-Burman: Screening Only',
      message: 'Plackett-Burman designs are highly efficient for screening but provide no information about interactions.',
      recommendation: 'Use for initial screening, then follow up with factorial or RSM designs for important factors',
      icon: 'Filter'
    })

    if (nFactors <= 4) {
      validations.push({
        severity: SEVERITY.WARNING,
        category: CATEGORY.EFFICIENCY,
        title: 'Plackett-Burman Unnecessary',
        message: `With only ${nFactors} factors, factorial designs provide more information for comparable run counts.`,
        recommendation: 'Consider Full or Fractional Factorial for better resolution',
        icon: 'AlertTriangle'
      })
    }
  }

  return validations
}

/**
 * Validate power analysis vs budget
 */
const validatePowerVsBudget = (powerAnalysis, budget) => {
  const validations = []
  const minRuns = powerAnalysis.minimumRuns
  const power = (powerAnalysis.desiredPower * 100).toFixed(0)

  if (budget < minRuns) {
    const shortfall = minRuns - budget
    validations.push({
      severity: SEVERITY.ERROR,
      category: CATEGORY.POWER,
      title: 'Insufficient Runs for Statistical Power',
      message: `Your budget of ${budget} runs is ${shortfall} runs short of the ${minRuns} runs needed for ${power}% statistical power.`,
      recommendation: `Increase budget to ${minRuns} runs or accept lower power (higher Type II error risk)`,
      icon: 'Zap'
    })
  } else if (budget >= minRuns && budget < minRuns * 1.2) {
    validations.push({
      severity: SEVERITY.SUCCESS,
      category: CATEGORY.POWER,
      title: 'Adequate Statistical Power',
      message: `Your budget of ${budget} runs meets the ${minRuns} runs needed for ${power}% power with ${powerAnalysis.effectSize} effect size.`,
      recommendation: 'Proceed with confidence - design has sufficient statistical power',
      icon: 'CheckCircle'
    })
  }

  return validations
}

/**
 * Validate goal alignment with design and parameters
 */
const validateGoalAlignment = (goal, selectedDesign, nFactors) => {
  const validations = []
  const designType = selectedDesign.type.toLowerCase()

  // Screening goal validations
  if (goal === 'screening') {
    if (nFactors < 5) {
      validations.push({
        severity: SEVERITY.INFO,
        category: CATEGORY.GOAL_MISMATCH,
        title: 'Screening May Be Unnecessary',
        message: `With only ${nFactors} factors, screening is typically not needed. Full factorial provides complete information efficiently.`,
        recommendation: 'Consider skipping screening and proceeding directly to optimization or full factorial',
        icon: 'Info'
      })
    }

    if ((designType.includes('central composite') || designType.includes('box-behnken')) && nFactors >= 5) {
      validations.push({
        severity: SEVERITY.WARNING,
        category: CATEGORY.GOAL_MISMATCH,
        title: 'Response Surface Design for Screening',
        message: 'Response surface designs (CCD, Box-Behnken) are inefficient for screening many factors. They\'re designed for optimization.',
        recommendation: 'Use Plackett-Burman or Fractional Factorial for initial screening',
        icon: 'Target'
      })
    }
  }

  // Optimization goal validations
  if (goal === 'optimization') {
    if (designType.includes('plackett-burman') || (designType.includes('fractional') && designType.includes('factorial'))) {
      validations.push({
        severity: SEVERITY.ERROR,
        category: CATEGORY.GOAL_MISMATCH,
        title: 'Wrong Design Type for Optimization',
        message: 'Screening designs cannot find optimal settings. They only identify important factors.',
        recommendation: 'Switch to Central Composite Design or Box-Behnken for optimization',
        icon: 'XCircle'
      })
    }

    if (designType.includes('full factorial') && !designType.includes('3-level')) {
      validations.push({
        severity: SEVERITY.WARNING,
        category: CATEGORY.DESIGN_LIMITATION,
        title: '2-Level Factorial Cannot Optimize',
        message: '2-level factorial designs cannot detect quadratic effects or identify optimal settings. They only test extremes.',
        recommendation: 'Use Central Composite Design or Box-Behnken which include center and intermediate points',
        icon: 'AlertTriangle'
      })
    }
  }

  // Modeling goal validations
  if (goal === 'modeling') {
    if (designType.includes('plackett-burman')) {
      validations.push({
        severity: SEVERITY.WARNING,
        category: CATEGORY.GOAL_MISMATCH,
        title: 'Plackett-Burman Cannot Model Interactions',
        message: 'Plackett-Burman designs cannot estimate interaction effects, which are often crucial for modeling.',
        recommendation: 'Use factorial or response surface designs for interaction modeling',
        icon: 'Network'
      })
    }
  }

  return validations
}

/**
 * Remove duplicate validations and sort by severity
 */
const deduplicateAndSort = (validations) => {
  // Remove duplicates based on title
  const unique = validations.filter((v, index, self) =>
    index === self.findIndex(t => t.title === v.title)
  )

  // Sort by severity: ERROR > WARNING > INFO > SUCCESS
  const severityOrder = {
    [SEVERITY.ERROR]: 0,
    [SEVERITY.WARNING]: 1,
    [SEVERITY.INFO]: 2,
    [SEVERITY.SUCCESS]: 3
  }

  return unique.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity])
}

/**
 * Get validation summary counts
 */
export const getValidationSummary = (validations) => {
  return {
    errors: validations.filter(v => v.severity === SEVERITY.ERROR).length,
    warnings: validations.filter(v => v.severity === SEVERITY.WARNING).length,
    info: validations.filter(v => v.severity === SEVERITY.INFO).length,
    success: validations.filter(v => v.severity === SEVERITY.SUCCESS).length,
    total: validations.length,
    hasBlockers: validations.some(v => v.severity === SEVERITY.ERROR)
  }
}

/**
 * Check if wizard can proceed (no blocking errors)
 */
export const canProceed = (validations) => {
  return !validations.some(v => v.severity === SEVERITY.ERROR)
}
