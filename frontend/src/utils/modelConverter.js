/**
 * Convert RSM model results to Prediction Profiler format
 */
export function convertRSMToProfilerModel(modelResult, factorNames, responseName, factorRanges) {
  if (!modelResult || !modelResult.coefficients) {
    throw new Error('Invalid model result: missing coefficients')
  }

  // Extract coefficient estimates and convert parameter names
  const coefficients = {}

  for (const [paramName, paramData] of Object.entries(modelResult.coefficients)) {
    if (paramData.estimate === null || paramData.estimate === undefined) {
      continue  // Skip null/undefined coefficients
    }

    // Convert parameter names to Prediction Profiler format
    let convertedName = paramName

    // Convert "I(X1**2)" -> "X1^2"
    if (paramName.startsWith('I(') && paramName.includes('**2')) {
      const factorMatch = paramName.match(/I\(([^*]+)\*\*2\)/)
      if (factorMatch) {
        convertedName = `${factorMatch[1]}^2`
      }
    }
    // Convert "X1:X2" -> "X1*X2"
    else if (paramName.includes(':')) {
      convertedName = paramName.replace(':', '*')
    }

    coefficients[convertedName] = paramData.estimate
  }

  // Determine model type
  const hasQuadraticTerms = Object.keys(coefficients).some(k => k.includes('^2'))
  const modelType = hasQuadraticTerms ? 'rsm_quadratic' : 'rsm_linear'

  return {
    model_type: modelType,
    coefficients,
    factors: factorNames,
    factor_ranges: factorRanges,
    response_name: responseName,
    source: 'RSM Analysis',
    r_squared: modelResult.r_squared,
    adj_r_squared: modelResult.adj_r_squared,
    rmse: modelResult.rmse
  }
}

/**
 * Infer factor ranges from data
 */
export function inferFactorRanges(tableData, factorNames) {
  const ranges = {}

  for (const factor of factorNames) {
    const values = tableData.map(row => row[factor]).filter(v => v !== null && v !== undefined)
    if (values.length === 0) {
      ranges[factor] = [-1, 1]  // Default range for coded designs
      continue
    }

    const min = Math.min(...values)
    const max = Math.max(...values)

    // Add 10% padding to ranges
    const padding = (max - min) * 0.1
    ranges[factor] = [min - padding, max + padding]
  }

  return ranges
}

/**
 * Store model in sessionStorage for cross-page transfer
 */
export function saveModelToSession(model, key = 'profiler_model') {
  try {
    sessionStorage.setItem(key, JSON.stringify(model))
    return true
  } catch (e) {
    console.error('Failed to save model to session:', e)
    return false
  }
}

/**
 * Retrieve model from sessionStorage
 */
export function loadModelFromSession(key = 'profiler_model') {
  try {
    const data = sessionStorage.getItem(key)
    if (!data) return null
    return JSON.parse(data)
  } catch (e) {
    console.error('Failed to load model from session:', e)
    return null
  }
}

/**
 * Clear model from sessionStorage
 */
export function clearModelFromSession(key = 'profiler_model') {
  try {
    sessionStorage.removeItem(key)
  } catch (e) {
    console.error('Failed to clear model from session:', e)
  }
}
