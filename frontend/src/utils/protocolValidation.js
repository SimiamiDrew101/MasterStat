/**
 * Protocol Validation Utility
 * Validates protocol data and returns errors/warnings
 */

/**
 * Validate complete protocol
 * @param {Object} protocol - Protocol object to validate
 * @returns {Object} Validation result with errors and warnings
 */
export const validateProtocol = (protocol) => {
  const errors = []
  const warnings = []

  // Validate metadata
  if (!protocol.metadata?.title?.trim()) {
    errors.push('Protocol title is required')
  }
  if (!protocol.metadata?.investigator?.trim()) {
    errors.push('Principal investigator name is required')
  }
  if (!protocol.metadata?.date) {
    errors.push('Protocol date is required')
  }

  // Validate objective section
  if (!protocol.objective?.researchQuestion?.trim()) {
    errors.push('Research question is required')
  }
  if (!protocol.objective?.hypothesis?.trim()) {
    errors.push('Hypothesis is required')
  }
  if (!protocol.objective?.primaryOutcome?.trim()) {
    warnings.push('Primary outcome should be specified')
  }

  // Validate materials section
  if (!protocol.materials?.factors || protocol.materials.factors.length === 0) {
    errors.push('At least one factor must be defined')
  } else {
    // Validate individual factors
    protocol.materials.factors.forEach((factor, idx) => {
      if (!factor.name?.trim()) {
        errors.push(`Factor ${idx + 1} must have a name`)
      }
    })
  }

  if (!protocol.materials?.sampleSize || protocol.materials.sampleSize < 3) {
    errors.push('Sample size must be at least 3')
  }

  if (!protocol.materials?.experimentalUnits?.trim()) {
    warnings.push('Experimental units should be defined')
  }

  // Validate procedure section
  if (!protocol.procedure?.preparation?.trim()) {
    warnings.push('Preparation steps should be documented')
  }
  if (!protocol.procedure?.executionSteps || protocol.procedure.executionSteps.length === 0) {
    errors.push('Execution steps are required')
  }
  if (!protocol.procedure?.measurementProtocol?.trim()) {
    warnings.push('Measurement protocol should be specified')
  }

  // Validate randomization section
  if (!protocol.randomization?.seed) {
    warnings.push('No randomization seed set - protocol will not be reproducible')
  }

  if (protocol.randomization?.method === 'block') {
    if (!protocol.randomization?.blockSize || protocol.randomization.blockSize < 2) {
      errors.push('Block size must be at least 2 for block randomization')
    }
  }

  if (!protocol.randomization?.randomizedDesign || protocol.randomization.randomizedDesign.length === 0) {
    warnings.push('Design has not been randomized yet - click "Apply Randomization"')
  }

  // Validate blinding section
  if (protocol.blinding?.type !== 'none') {
    if (protocol.blinding.blindedParties.length === 0) {
      errors.push('Specify which parties are blinded (experimenter, evaluator, and/or subject)')
    }

    if (protocol.blinding.codeType === 'custom') {
      if (!protocol.blinding.customCodes || Object.keys(protocol.blinding.customCodes).length === 0) {
        errors.push('Custom blinding codes must be specified')
      }
    }

    if (!protocol.blinding?.unblindingCriteria?.trim()) {
      warnings.push('Unblinding criteria should be specified')
    }

    // Check if blinding is appropriate for design
    if (protocol.materials?.factors && protocol.materials.factors.length === 1) {
      warnings.push('Single-factor designs may have limited blinding options')
    }
  }

  // Validate data recording section
  if (!protocol.dataRecording?.responseVariables || protocol.dataRecording.responseVariables.length === 0) {
    errors.push('At least one response variable must be defined')
  }

  if (!protocol.dataRecording?.entryMethod?.trim()) {
    warnings.push('Data entry method should be specified')
  }

  if (!protocol.dataRecording?.qualityAssurance?.trim()) {
    warnings.push('Quality assurance procedures should be documented')
  }

  if (!protocol.dataRecording?.backupProcedure?.trim()) {
    warnings.push('Data backup procedure should be specified')
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  }
}

/**
 * Validate specific section
 * @param {Object} protocol - Protocol object
 * @param {string} sectionName - Section to validate
 * @returns {Object} Section validation result
 */
export const validateSection = (protocol, sectionName) => {
  const errors = []
  const warnings = []

  switch (sectionName) {
    case 'objective':
      if (!protocol.objective?.researchQuestion?.trim()) {
        errors.push('Research question is required')
      }
      if (!protocol.objective?.hypothesis?.trim()) {
        errors.push('Hypothesis is required')
      }
      if (!protocol.objective?.primaryOutcome?.trim()) {
        warnings.push('Primary outcome should be specified')
      }
      break

    case 'materials':
      if (!protocol.materials?.factors || protocol.materials.factors.length === 0) {
        errors.push('At least one factor must be defined')
      }
      if (!protocol.materials?.sampleSize || protocol.materials.sampleSize < 3) {
        errors.push('Sample size must be at least 3')
      }
      break

    case 'procedure':
      if (!protocol.procedure?.executionSteps || protocol.procedure.executionSteps.length === 0) {
        errors.push('Execution steps are required')
      }
      break

    case 'randomization':
      if (!protocol.randomization?.seed) {
        warnings.push('Random seed should be set for reproducibility')
      }
      if (protocol.randomization?.method === 'block' && !protocol.randomization?.blockSize) {
        errors.push('Block size required for block randomization')
      }
      break

    case 'blinding':
      if (protocol.blinding?.type !== 'none' && protocol.blinding.blindedParties.length === 0) {
        errors.push('Specify which parties are blinded')
      }
      break

    case 'dataRecording':
      if (!protocol.dataRecording?.responseVariables || protocol.dataRecording.responseVariables.length === 0) {
        errors.push('At least one response variable must be defined')
      }
      break

    default:
      break
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  }
}

/**
 * Check if protocol is complete enough to export
 * @param {Object} protocol - Protocol object
 * @returns {boolean} True if exportable
 */
export const isExportable = (protocol) => {
  const validation = validateProtocol(protocol)
  return validation.isValid
}

/**
 * Get validation summary message
 * @param {Object} validation - Validation result
 * @returns {string} Human-readable summary
 */
export const getValidationSummary = (validation) => {
  if (validation.isValid) {
    if (validation.warnings.length === 0) {
      return 'Protocol is complete and ready to export'
    }
    return `Protocol is ready to export (${validation.warnings.length} warning${validation.warnings.length > 1 ? 's' : ''})`
  }

  return `Protocol has ${validation.errors.length} error${validation.errors.length > 1 ? 's' : ''} that must be fixed`
}

export default {
  validateProtocol,
  validateSection,
  isExportable,
  getValidationSummary,
}
