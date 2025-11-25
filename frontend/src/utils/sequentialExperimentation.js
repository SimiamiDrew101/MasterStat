// Sequential Experimentation Guide
// Provides guidance for multi-phase DOE workflows

/**
 * Phase types for sequential experimentation
 */
export const PHASE_TYPE = {
  SCREENING: 'screening',
  CHARACTERIZATION: 'characterization',
  OPTIMIZATION: 'optimization',
  VERIFICATION: 'verification'
}

/**
 * Detect if sequential experimentation is recommended
 * Returns guidance object with phase information
 */
export const detectSequentialScenario = (wizardData) => {
  const { goal, nFactors, selectedDesign, budget } = wizardData

  // Scenario 1: High factor count with any goal - should screen first
  if (nFactors >= 5 && goal !== 'screening') {
    return {
      isSequential: true,
      currentPhase: PHASE_TYPE.SCREENING,
      recommendedPhase: PHASE_TYPE.SCREENING,
      nextPhase: goal === 'optimization' ? PHASE_TYPE.OPTIMIZATION : PHASE_TYPE.CHARACTERIZATION,
      scenario: 'high_factor_count',
      warning: true,
      title: 'Sequential Experimentation Recommended',
      message: `With ${nFactors} factors, it's more efficient to screen first to identify the 2-4 most important factors, then optimize.`,
      currentPhaseGuidance: {
        title: 'Recommended: Start with Screening (Phase 1)',
        description: 'Run a screening design to identify which factors significantly affect your response.',
        designRecommendations: [
          'Plackett-Burman (most efficient)',
          `Fractional Factorial (Resolution IV or V)`,
          `2^(${nFactors}-p) design with minimal runs`
        ],
        expectedRuns: Math.ceil((nFactors + 1) / 4) * 4, // PB estimate
        actionItems: [
          'Select a screening design from the wizard',
          `Run the ${Math.ceil((nFactors + 1) / 4) * 4}-run experiment`,
          'Analyze results to identify significant factors',
          'Note which 2-4 factors have the largest effects'
        ]
      },
      nextPhaseGuidance: {
        title: goal === 'optimization' ? 'Phase 2: Optimization (After Screening)' : 'Phase 2: Characterization',
        description: goal === 'optimization'
          ? 'After screening, use Response Surface Methodology to find optimal settings for the important factors.'
          : 'After screening, characterize interactions and main effects for important factors.',
        designRecommendations: goal === 'optimization'
          ? [
              'Central Composite Design (CCD)',
              'Box-Behnken Design',
              'Optimal Design (D-optimal)'
            ]
          : [
              'Full Factorial (2-4 factors)',
              'Fractional Factorial (Resolution V)',
              'Full model with interactions'
            ],
        expectedRuns: '15-30 runs for 3-4 factors',
        actionItems: [
          'Return to this wizard after analyzing screening results',
          'Enter the 2-4 significant factors from Phase 1',
          'Select an optimization or characterization design',
          'Run the refined experiment to find optimal settings'
        ]
      },
      timeline: `Total: ${Math.ceil((nFactors + 1) / 4) * 4} runs (screening) + 20-30 runs (optimization) = ${Math.ceil((nFactors + 1) / 4) * 4 + 25} runs approximately`
    }
  }

  // Scenario 2: User selected screening design - guide them on next steps
  if (goal === 'screening' || (selectedDesign && selectedDesign.type.toLowerCase().includes('screening'))) {
    const isPlackettBurman = selectedDesign?.type.toLowerCase().includes('plackett-burman')
    const isFractional = selectedDesign?.type.toLowerCase().includes('fractional')

    return {
      isSequential: true,
      currentPhase: PHASE_TYPE.SCREENING,
      nextPhase: PHASE_TYPE.OPTIMIZATION,
      scenario: 'screening_selected',
      warning: false,
      title: 'Sequential Experimentation Workflow',
      message: 'You are designing Phase 1 (Screening). After analysis, you will design Phase 2 for the important factors.',
      currentPhaseGuidance: {
        title: 'Current: Phase 1 - Screening',
        description: `Run this ${selectedDesign?.runs || ''}-run screening design to identify which factors matter most.`,
        designRecommendations: [selectedDesign?.type || 'Screening design'],
        expectedRuns: selectedDesign?.runs || nFactors * 2,
        actionItems: [
          `Run the ${selectedDesign?.runs || nFactors * 2}-run screening experiment`,
          'Perform statistical analysis (Pareto chart, main effects plot)',
          'Identify 2-4 factors with largest effects (p < 0.05)',
          'Consider which factors show potential interactions',
          'Save your experimental data for reference'
        ],
        tips: [
          isPlackettBurman ? 'Plackett-Burman assumes no interactions - use for factor identification only' : null,
          isFractional ? 'Check your confounding pattern - some main effects may be aliased with interactions' : null,
          'Use half-normal plots to identify active factors',
          'Factors with small effects can be fixed at optimal levels or removed'
        ].filter(Boolean)
      },
      nextPhaseGuidance: {
        title: 'Phase 2: Optimization (After Screening Analysis)',
        description: 'Design a Response Surface experiment with the 2-4 most important factors from screening.',
        designRecommendations: [
          'Central Composite Design (CCD) - Best for optimization',
          'Box-Behnken Design - Good when corner points are impractical',
          'Full Factorial with center points - If only 2-3 important factors found'
        ],
        expectedRuns: '15-30 runs for 3-4 factors',
        actionItems: [
          'Analyze your screening results completely',
          'Identify the 2-4 most significant factors',
          'Return to this Experiment Wizard',
          'Click "Design Phase 2" or start a new experiment',
          'Enter only the significant factors from screening',
          'Select goal: "Optimization"',
          'Choose Central Composite Design or Box-Behnken',
          'Run the optimization experiment to find ideal settings'
        ]
      },
      timeline: `Total: ${selectedDesign?.runs || nFactors * 2} runs (Phase 1) + 20-30 runs (Phase 2) = ${(selectedDesign?.runs || nFactors * 2) + 25} runs total`
    }
  }

  // Scenario 3: Optimization goal with screening design selected (warning)
  if (goal === 'optimization' && selectedDesign) {
    const isScreeningDesign =
      selectedDesign.type.toLowerCase().includes('plackett-burman') ||
      selectedDesign.type.toLowerCase().includes('fractional factorial') ||
      selectedDesign.type.toLowerCase().includes('screening')

    if (isScreeningDesign) {
      return {
        isSequential: true,
        currentPhase: PHASE_TYPE.SCREENING,
        nextPhase: PHASE_TYPE.OPTIMIZATION,
        scenario: 'optimization_goal_screening_design',
        warning: true,
        title: 'Design-Goal Mismatch Detected',
        message: `Your goal is optimization, but ${selectedDesign.type} is a screening design. Sequential experimentation recommended.`,
        currentPhaseGuidance: {
          title: 'Step 1: Complete Screening First',
          description: 'Screening designs identify important factors but cannot find optimal settings.',
          designRecommendations: [selectedDesign.type],
          expectedRuns: selectedDesign.runs,
          actionItems: [
            'Run this screening design to identify important factors',
            'Analyze which factors significantly affect the response',
            'Proceed to Phase 2 with only the important factors'
          ]
        },
        nextPhaseGuidance: {
          title: 'Step 2: Run Optimization Design',
          description: 'After screening, use the important factors in a Response Surface design.',
          designRecommendations: [
            'Central Composite Design',
            'Box-Behnken Design',
            'Optimal Design (RSM)'
          ],
          expectedRuns: '15-30 runs',
          actionItems: [
            'Return here after analyzing screening results',
            'Design Phase 2 with 2-4 important factors',
            'Use CCD or Box-Behnken for optimization',
            'Find optimal factor settings'
          ]
        },
        timeline: `${selectedDesign.runs} runs (screening) + 20-30 runs (optimization) = ${selectedDesign.runs + 25} runs total`
      }
    }
  }

  // Scenario 4: Good single-phase design - no sequential needed
  if (nFactors <= 4 && goal === 'optimization' && selectedDesign) {
    const isRSMDesign =
      selectedDesign.type.toLowerCase().includes('central composite') ||
      selectedDesign.type.toLowerCase().includes('box-behnken') ||
      selectedDesign.type.toLowerCase().includes('response surface')

    if (isRSMDesign) {
      return {
        isSequential: false,
        currentPhase: PHASE_TYPE.OPTIMIZATION,
        scenario: 'single_phase_optimal',
        warning: false,
        title: 'Single-Phase Experiment',
        message: `With ${nFactors} factors and an optimization goal, ${selectedDesign.type} is appropriate for a single-phase experiment.`,
        currentPhaseGuidance: {
          title: 'Single-Phase Optimization',
          description: `Your ${selectedDesign.runs}-run design is sufficient to find optimal settings without screening.`,
          designRecommendations: [selectedDesign.type],
          expectedRuns: selectedDesign.runs,
          actionItems: [
            `Run the ${selectedDesign.runs}-run experiment`,
            'Fit response surface model (quadratic)',
            'Analyze contour plots and response surfaces',
            'Identify optimal factor settings',
            'Verify optimum with confirmation runs'
          ]
        },
        timeline: `${selectedDesign.runs} runs + 3-5 confirmation runs = ${selectedDesign.runs + 4} runs total`
      }
    }
  }

  // Scenario 5: Few factors with screening goal
  if (nFactors <= 4 && goal === 'screening') {
    return {
      isSequential: true,
      currentPhase: PHASE_TYPE.SCREENING,
      nextPhase: PHASE_TYPE.OPTIMIZATION,
      scenario: 'few_factors_screening',
      warning: false,
      title: 'Screening + Optimization Pathway',
      message: `With ${nFactors} factors, you can screen and then optimize the important factors.`,
      currentPhaseGuidance: {
        title: 'Phase 1: Factor Screening',
        description: 'Identify which factors significantly affect your response.',
        designRecommendations: [
          'Full Factorial (most information)',
          'Fractional Factorial (more efficient)',
          selectedDesign?.type || 'Screening design'
        ],
        expectedRuns: selectedDesign?.runs || Math.pow(2, nFactors),
        actionItems: [
          'Run the screening experiment',
          'Analyze main effects and interactions',
          'Identify significant factors (p < 0.05)',
          'Note optimal direction for each factor'
        ]
      },
      nextPhaseGuidance: {
        title: 'Phase 2: Optimization (Optional)',
        description: 'If needed, run an RSM design to find precise optimal settings.',
        designRecommendations: [
          'Central Composite Design',
          'Box-Behnken Design',
          'Steepest ascent/descent experiments'
        ],
        expectedRuns: '15-25 runs for 2-3 significant factors',
        actionItems: [
          'If screening shows clear optimal direction, may not need Phase 2',
          'If precise optimization needed, return here to design RSM',
          'Use only significant factors from screening',
          'Run optimization experiment'
        ]
      },
      timeline: `${selectedDesign?.runs || Math.pow(2, nFactors)} runs (screening) + optional 15-25 runs (optimization)`
    }
  }

  // Default: No sequential guidance needed
  return {
    isSequential: false,
    currentPhase: null,
    scenario: 'no_guidance',
    warning: false,
    title: 'Standard Experiment',
    message: 'This experiment can be completed in a single phase.'
  }
}

/**
 * Save wizard state to localStorage for multi-phase experiments
 */
export const saveWizardState = (wizardData, phaseName = 'Phase 1') => {
  const state = {
    ...wizardData,
    phaseName,
    savedAt: new Date().toISOString(),
    version: '1.0'
  }

  try {
    localStorage.setItem('masterstat_wizard_state', JSON.stringify(state))

    // Also save to history
    const history = loadWizardHistory()
    history.unshift({
      id: Date.now(),
      phaseName,
      savedAt: state.savedAt,
      nFactors: wizardData.nFactors,
      goal: wizardData.goal,
      designType: wizardData.selectedDesign?.type || 'Not selected'
    })

    // Keep only last 10 experiments
    const trimmedHistory = history.slice(0, 10)
    localStorage.setItem('masterstat_wizard_history', JSON.stringify(trimmedHistory))

    return true
  } catch (error) {
    console.error('Failed to save wizard state:', error)
    return false
  }
}

/**
 * Load wizard state from localStorage
 */
export const loadWizardState = () => {
  try {
    const stateJson = localStorage.getItem('masterstat_wizard_state')
    if (!stateJson) return null

    const state = JSON.parse(stateJson)
    return state
  } catch (error) {
    console.error('Failed to load wizard state:', error)
    return null
  }
}

/**
 * Clear wizard state
 */
export const clearWizardState = () => {
  try {
    localStorage.removeItem('masterstat_wizard_state')
    return true
  } catch (error) {
    console.error('Failed to clear wizard state:', error)
    return false
  }
}

/**
 * Load wizard history
 */
export const loadWizardHistory = () => {
  try {
    const historyJson = localStorage.getItem('masterstat_wizard_history')
    if (!historyJson) return []

    return JSON.parse(historyJson)
  } catch (error) {
    console.error('Failed to load wizard history:', error)
    return []
  }
}

/**
 * Delete a specific history entry
 */
export const deleteHistoryEntry = (id) => {
  try {
    const history = loadWizardHistory()
    const filtered = history.filter(entry => entry.id !== id)
    localStorage.setItem('masterstat_wizard_history', JSON.stringify(filtered))
    return true
  } catch (error) {
    console.error('Failed to delete history entry:', error)
    return false
  }
}

/**
 * Get phase recommendations for next phase design
 */
export const getPhase2Recommendations = (phase1Results) => {
  // This would be called after user analyzes their screening results
  // For now, provide general guidance
  return {
    title: 'Ready for Phase 2?',
    description: 'Use these guidelines to design your optimization experiment.',
    steps: [
      'Analyze your Phase 1 data completely',
      'Identify 2-4 factors with largest effects',
      'Return to Experiment Wizard',
      'Select Goal: "Optimization"',
      'Enter only the significant factors',
      'Choose Central Composite Design or Box-Behnken',
      'Review recommendations and generate design'
    ],
    tips: [
      'Fix unimportant factors at their best levels',
      'Use the same response variable',
      'Consider expanding factor ranges if edge effects seen',
      'Include center point replicates for pure error estimation'
    ]
  }
}

/**
 * Generate sequential experiment timeline
 */
export const generateTimeline = (sequentialGuidance) => {
  if (!sequentialGuidance.isSequential) {
    return {
      phases: [
        {
          phase: 'Single Phase',
          runs: sequentialGuidance.currentPhaseGuidance?.expectedRuns || 'TBD',
          description: 'Complete experiment in one phase'
        }
      ],
      totalRuns: sequentialGuidance.currentPhaseGuidance?.expectedRuns || 'TBD'
    }
  }

  return {
    phases: [
      {
        phase: 'Phase 1: Screening',
        runs: sequentialGuidance.currentPhaseGuidance?.expectedRuns || 'TBD',
        description: sequentialGuidance.currentPhaseGuidance?.description || ''
      },
      {
        phase: 'Phase 2: Optimization',
        runs: sequentialGuidance.nextPhaseGuidance?.expectedRuns || '20-30',
        description: sequentialGuidance.nextPhaseGuidance?.description || ''
      }
    ],
    totalRuns: sequentialGuidance.timeline || 'TBD'
  }
}

/**
 * Check if user should be warned about inefficient approach
 */
export const checkEfficiencyWarnings = (wizardData) => {
  const { goal, nFactors, selectedDesign, budget } = wizardData
  const warnings = []

  // Warning: Many factors + optimization goal without screening
  if (nFactors >= 5 && goal === 'optimization' && selectedDesign) {
    const isRSMDesign =
      selectedDesign.type.toLowerCase().includes('central composite') ||
      selectedDesign.type.toLowerCase().includes('box-behnken')

    if (isRSMDesign) {
      warnings.push({
        severity: 'warning',
        title: 'Consider Sequential Experimentation',
        message: `RSM designs with ${nFactors} factors require ${selectedDesign.runs} runs. Consider screening to reduce to 3-4 factors first (total ~${Math.ceil(nFactors * 1.5) + 25} runs).`,
        recommendation: 'Screen first, then optimize with fewer factors for better efficiency'
      })
    }
  }

  // Warning: Low budget + high factor count
  if (budget && nFactors >= 5 && budget < nFactors * 3) {
    warnings.push({
      severity: 'error',
      title: 'Budget Too Low for Factor Count',
      message: `${nFactors} factors with budget of ${budget} runs is severely underpowered. Consider reducing factors or using sequential approach.`,
      recommendation: `Increase budget to at least ${nFactors * 3} runs or reduce to ${Math.floor(budget / 3)} factors`
    })
  }

  return warnings
}
