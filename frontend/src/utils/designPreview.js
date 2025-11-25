// Design Preview Generator
// Generates preview data for visualizing experimental designs before generation

/**
 * Generate preview points for different design types
 * Returns array of points with coordinates and type labels
 */
export const generateDesignPreview = (designType, nFactors) => {
  const type = designType?.toLowerCase() || ''

  if (nFactors === 2) {
    return generate2DPreview(type, nFactors)
  } else if (nFactors === 3) {
    return generate3DPreview(type, nFactors)
  } else {
    return generateMultiDimensionalPreview(type, nFactors)
  }
}

/**
 * Generate 2D design preview (2 factors)
 */
const generate2DPreview = (type, nFactors) => {
  const points = []

  if (type.includes('full factorial') || type.includes('2^k')) {
    // Full Factorial: corner points only
    points.push({ x: -1, y: -1, type: 'Corner', color: '#ef4444' })
    points.push({ x: -1, y: 1, type: 'Corner', color: '#ef4444' })
    points.push({ x: 1, y: -1, type: 'Corner', color: '#ef4444' })
    points.push({ x: 1, y: 1, type: 'Corner', color: '#ef4444' })
  } else if (type.includes('central composite') || type.includes('ccd')) {
    // CCD: corners + center + axial
    // Corner points
    points.push({ x: -1, y: -1, type: 'Corner', color: '#ef4444' })
    points.push({ x: -1, y: 1, type: 'Corner', color: '#ef4444' })
    points.push({ x: 1, y: -1, type: 'Corner', color: '#ef4444' })
    points.push({ x: 1, y: 1, type: 'Corner', color: '#ef4444' })
    // Center points
    for (let i = 0; i < 3; i++) {
      points.push({ x: 0, y: 0, type: 'Center', color: '#22c55e' })
    }
    // Axial points (star points)
    const alpha = 1.414 // For 2 factors
    points.push({ x: -alpha, y: 0, type: 'Axial', color: '#3b82f6' })
    points.push({ x: alpha, y: 0, type: 'Axial', color: '#3b82f6' })
    points.push({ x: 0, y: -alpha, type: 'Axial', color: '#3b82f6' })
    points.push({ x: 0, y: alpha, type: 'Axial', color: '#3b82f6' })
  } else if (type.includes('box-behnken')) {
    // Box-Behnken: edge midpoints + center
    points.push({ x: -1, y: 0, type: 'Edge', color: '#f59e0b' })
    points.push({ x: 1, y: 0, type: 'Edge', color: '#f59e0b' })
    points.push({ x: 0, y: -1, type: 'Edge', color: '#f59e0b' })
    points.push({ x: 0, y: 1, type: 'Edge', color: '#f59e0b' })
    for (let i = 0; i < 3; i++) {
      points.push({ x: 0, y: 0, type: 'Center', color: '#22c55e' })
    }
  } else if (type.includes('fractional factorial')) {
    // Fractional Factorial: subset of corners
    points.push({ x: -1, y: -1, type: 'Corner', color: '#ef4444' })
    points.push({ x: 1, y: 1, type: 'Corner', color: '#ef4444' })
  } else if (type.includes('plackett-burman')) {
    // Plackett-Burman: efficient screening design
    points.push({ x: -1, y: -1, type: 'Screening', color: '#a855f7' })
    points.push({ x: -1, y: 1, type: 'Screening', color: '#a855f7' })
    points.push({ x: 1, y: -1, type: 'Screening', color: '#a855f7' })
    points.push({ x: 1, y: 1, type: 'Screening', color: '#a855f7' })
  } else {
    // Default: Full Factorial
    points.push({ x: -1, y: -1, type: 'Corner', color: '#ef4444' })
    points.push({ x: -1, y: 1, type: 'Corner', color: '#ef4444' })
    points.push({ x: 1, y: -1, type: 'Corner', color: '#ef4444' })
    points.push({ x: 1, y: 1, type: 'Corner', color: '#ef4444' })
  }

  return points
}

/**
 * Generate 3D design preview (3 factors)
 */
const generate3DPreview = (type, nFactors) => {
  const points = []

  if (type.includes('full factorial') || type.includes('2^k')) {
    // Full Factorial: all 8 corners of cube
    for (let i = -1; i <= 1; i += 2) {
      for (let j = -1; j <= 1; j += 2) {
        for (let k = -1; k <= 1; k += 2) {
          points.push({ x: i, y: j, z: k, type: 'Corner', color: '#ef4444' })
        }
      }
    }
  } else if (type.includes('central composite') || type.includes('ccd')) {
    // CCD: corners + center + axial
    // 8 corner points
    for (let i = -1; i <= 1; i += 2) {
      for (let j = -1; j <= 1; j += 2) {
        for (let k = -1; k <= 1; k += 2) {
          points.push({ x: i, y: j, z: k, type: 'Corner', color: '#ef4444' })
        }
      }
    }
    // Center points (3-5 replicates)
    for (let i = 0; i < 4; i++) {
      points.push({ x: 0, y: 0, z: 0, type: 'Center', color: '#22c55e' })
    }
    // 6 axial points (star points)
    const alpha = 1.682 // For 3 factors
    points.push({ x: -alpha, y: 0, z: 0, type: 'Axial', color: '#3b82f6' })
    points.push({ x: alpha, y: 0, z: 0, type: 'Axial', color: '#3b82f6' })
    points.push({ x: 0, y: -alpha, z: 0, type: 'Axial', color: '#3b82f6' })
    points.push({ x: 0, y: alpha, z: 0, type: 'Axial', color: '#3b82f6' })
    points.push({ x: 0, y: 0, z: -alpha, type: 'Axial', color: '#3b82f6' })
    points.push({ x: 0, y: 0, z: alpha, type: 'Axial', color: '#3b82f6' })
  } else if (type.includes('box-behnken')) {
    // Box-Behnken: edge midpoints + center (no corners!)
    // 12 edge midpoint combinations
    const combos = [
      [-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0],
      [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1],
      [0, -1, -1], [0, -1, 1], [0, 1, -1], [0, 1, 1]
    ]
    combos.forEach(([x, y, z]) => {
      points.push({ x, y, z, type: 'Edge', color: '#f59e0b' })
    })
    // Center points
    for (let i = 0; i < 3; i++) {
      points.push({ x: 0, y: 0, z: 0, type: 'Center', color: '#22c55e' })
    }
  } else if (type.includes('fractional factorial')) {
    // Fractional Factorial: half of the cube (4 points)
    points.push({ x: -1, y: -1, z: -1, type: 'Corner', color: '#ef4444' })
    points.push({ x: -1, y: 1, z: 1, type: 'Corner', color: '#ef4444' })
    points.push({ x: 1, y: -1, z: 1, type: 'Corner', color: '#ef4444' })
    points.push({ x: 1, y: 1, z: -1, type: 'Corner', color: '#ef4444' })
  } else if (type.includes('plackett-burman')) {
    // Plackett-Burman: efficient screening
    for (let i = -1; i <= 1; i += 2) {
      for (let j = -1; j <= 1; j += 2) {
        for (let k = -1; k <= 1; k += 2) {
          if (Math.random() > 0.5) { // Random subset
            points.push({ x: i, y: j, z: k, type: 'Screening', color: '#a855f7' })
          }
        }
      }
    }
  } else {
    // Default: Full Factorial
    for (let i = -1; i <= 1; i += 2) {
      for (let j = -1; j <= 1; j += 2) {
        for (let k = -1; k <= 1; k += 2) {
          points.push({ x: i, y: j, z: k, type: 'Corner', color: '#ef4444' })
        }
      }
    }
  }

  return points
}

/**
 * Generate preview for 4+ factors (returns multiple 2D projections)
 */
const generateMultiDimensionalPreview = (type, nFactors) => {
  // For 4+ factors, we'll return projections onto 2D planes
  // This returns an array of {factor1, factor2, points} objects
  const projections = []

  // Generate preview for first 3 factor pairs
  const maxPairs = Math.min(3, Math.floor((nFactors * (nFactors - 1)) / 2))
  let pairCount = 0

  for (let i = 0; i < nFactors && pairCount < maxPairs; i++) {
    for (let j = i + 1; j < nFactors && pairCount < maxPairs; j++) {
      const points = generate2DPreview(type, 2)
      projections.push({
        factor1: i,
        factor2: j,
        factorName1: `Factor ${i + 1}`,
        factorName2: `Factor ${j + 1}`,
        points: points
      })
      pairCount++
    }
  }

  return projections
}

/**
 * Calculate design structure statistics
 */
export const getDesignStatistics = (designType, nFactors) => {
  const type = designType?.toLowerCase() || ''
  let stats = {
    cornerPoints: 0,
    centerPoints: 0,
    axialPoints: 0,
    edgePoints: 0,
    totalRuns: 0,
    designFamily: '',
    description: ''
  }

  if (type.includes('full factorial') || type.includes('2^k')) {
    stats.cornerPoints = Math.pow(2, nFactors)
    stats.totalRuns = stats.cornerPoints
    stats.designFamily = 'Full Factorial'
    stats.description = 'Tests all combinations of factor levels at corners of design space'
  } else if (type.includes('fractional factorial')) {
    stats.cornerPoints = Math.pow(2, nFactors - 1) // Half-fraction approximation
    stats.totalRuns = stats.cornerPoints
    stats.designFamily = 'Fractional Factorial'
    stats.description = 'Tests strategically selected subset of factorial combinations'
  } else if (type.includes('central composite') || type.includes('ccd')) {
    stats.cornerPoints = Math.pow(2, nFactors)
    stats.centerPoints = Math.max(3, nFactors)
    stats.axialPoints = 2 * nFactors
    stats.totalRuns = stats.cornerPoints + stats.centerPoints + stats.axialPoints
    stats.designFamily = 'Central Composite Design (CCD)'
    stats.description = 'Combines factorial, axial, and center points for response surface modeling'
  } else if (type.includes('box-behnken')) {
    if (nFactors === 3) {
      stats.edgePoints = 12
      stats.centerPoints = 3
    } else if (nFactors === 4) {
      stats.edgePoints = 24
      stats.centerPoints = 3
    } else {
      stats.edgePoints = nFactors * (nFactors - 1) * 2
      stats.centerPoints = 3
    }
    stats.totalRuns = stats.edgePoints + stats.centerPoints
    stats.designFamily = 'Box-Behnken Design'
    stats.description = 'Uses edge midpoints only - does not test extreme corners'
  } else if (type.includes('plackett-burman')) {
    // Plackett-Burman runs are multiples of 4
    stats.totalRuns = Math.ceil((nFactors + 1) / 4) * 4
    stats.cornerPoints = stats.totalRuns
    stats.designFamily = 'Plackett-Burman'
    stats.description = 'Efficient screening design for identifying important factors'
  } else {
    stats.cornerPoints = Math.pow(2, nFactors)
    stats.totalRuns = stats.cornerPoints
    stats.designFamily = 'Full Factorial'
    stats.description = 'Tests all combinations of factor levels'
  }

  return stats
}

/**
 * Get design space coverage description
 */
export const getDesignCoverageInfo = (designType) => {
  const type = designType?.toLowerCase() || ''

  if (type.includes('central composite') || type.includes('ccd')) {
    return {
      coverage: 'Excellent',
      description: 'Covers corners, edges, center, and extends beyond the cube for curvature estimation',
      color: '#22c55e'
    }
  } else if (type.includes('box-behnken')) {
    return {
      coverage: 'Good',
      description: 'Covers edges and center but avoids extreme corners - safe for constrained regions',
      color: '#3b82f6'
    }
  } else if (type.includes('full factorial')) {
    return {
      coverage: 'Complete',
      description: 'Tests all possible combinations - thorough but expensive for many factors',
      color: '#22c55e'
    }
  } else if (type.includes('fractional factorial')) {
    return {
      coverage: 'Efficient',
      description: 'Strategic subset of factorial points - efficient screening but some confounding',
      color: '#f59e0b'
    }
  } else if (type.includes('plackett-burman')) {
    return {
      coverage: 'Screening',
      description: 'Minimal runs for factor screening - identifies important factors efficiently',
      color: '#a855f7'
    }
  } else {
    return {
      coverage: 'Standard',
      description: 'Standard design space coverage',
      color: '#6b7280'
    }
  }
}

/**
 * Get point type legend information
 */
export const getPointTypeLegend = () => {
  return [
    { type: 'Corner', color: '#ef4444', description: 'Factorial points at extremes' },
    { type: 'Center', color: '#22c55e', description: 'Center points for curvature & error' },
    { type: 'Axial', color: '#3b82f6', description: 'Star points for quadratic terms' },
    { type: 'Edge', color: '#f59e0b', description: 'Edge midpoints (Box-Behnken)' },
    { type: 'Screening', color: '#a855f7', description: 'Screening design points' }
  ]
}
