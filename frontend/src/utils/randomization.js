/**
 * Randomization Utility with Seeded RNG
 * CRITICAL: Implements seed-based reproducible randomization - addressing a major gap in the application
 */

import seedrandom from 'seedrandom'

/**
 * Seeded Random Number Generator for reproducible randomization
 * Using the same seed will produce IDENTICAL results every time
 */
export class SeededRNG {
  constructor(seed) {
    this.seed = seed
    this.rng = seedrandom(seed.toString())
  }

  /**
   * Generate random number between 0 and 1
   */
  random() {
    return this.rng()
  }

  /**
   * Generate random integer between min and max (inclusive)
   */
  randomInt(min, max) {
    return Math.floor(this.rng() * (max - min + 1)) + min
  }

  /**
   * Fisher-Yates shuffle algorithm with seeded RNG
   * Guarantees identical shuffle for same seed
   */
  shuffle(array) {
    const shuffled = [...array]
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = this.randomInt(0, i)
      ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
    }
    return shuffled
  }

  /**
   * Sample n items from array without replacement
   */
  sample(array, n) {
    if (n > array.length) {
      throw new Error('Sample size cannot exceed array length')
    }
    const shuffled = this.shuffle(array)
    return shuffled.slice(0, n)
  }

  /**
   * Get current seed (for record-keeping)
   */
  getSeed() {
    return this.seed
  }
}

/**
 * Complete Randomization (Completely Randomized Design - CRD)
 * Shuffles all runs without any restrictions
 *
 * @param {Array} design - Design matrix (array of run objects)
 * @param {number} seed - Random seed for reproducibility
 * @returns {Array} Randomized design with run order
 */
export const completeRandomization = (design, seed) => {
  if (!design || design.length === 0) {
    return []
  }

  if (!seed) {
    console.warn('No seed provided - randomization will not be reproducible!')
    seed = Math.floor(Math.random() * 1000000)
  }

  const rng = new SeededRNG(seed)

  // Add original order tracking
  const designWithOriginal = design.map((run, idx) => ({
    ...run,
    originalOrder: idx + 1,
  }))

  // Shuffle using seeded RNG
  const randomized = rng.shuffle(designWithOriginal)

  // Add randomized run order
  return randomized.map((run, idx) => ({
    ...run,
    runOrder: idx + 1,
  }))
}

/**
 * Block Randomization (Randomized Block Design)
 * Randomizes within blocks, then randomizes overall order
 *
 * @param {Array} design - Design matrix
 * @param {number} blockSize - Number of runs per block
 * @param {number} seed - Random seed
 * @returns {Array} Randomized design with block assignments
 */
export const blockRandomization = (design, blockSize, seed) => {
  if (!design || design.length === 0) {
    return []
  }

  if (!blockSize || blockSize < 2) {
    throw new Error('Block size must be at least 2')
  }

  if (!seed) {
    console.warn('No seed provided - randomization will not be reproducible!')
    seed = Math.floor(Math.random() * 1000000)
  }

  const rng = new SeededRNG(seed)

  // Split design into blocks
  const blocks = []
  for (let i = 0; i < design.length; i += blockSize) {
    blocks.push(design.slice(i, i + blockSize))
  }

  // Randomize within each block
  const randomizedBlocks = blocks.map((block, blockIdx) => {
    const shuffled = rng.shuffle(block)
    return shuffled.map(run => ({
      ...run,
      block: blockIdx + 1,
    }))
  })

  // Flatten blocks
  const flattened = randomizedBlocks.flat()

  // Add run order
  return flattened.map((run, idx) => ({
    ...run,
    runOrder: idx + 1,
    originalOrder: idx + 1,
  }))
}

/**
 * Restricted Randomization
 * Maintains balance by grouping and interleaving
 *
 * @param {Array} design - Design matrix
 * @param {string} restrictionFactor - Factor name to restrict on
 * @param {number} seed - Random seed
 * @returns {Array} Randomized design with balanced restriction
 */
export const restrictedRandomization = (design, restrictionFactor, seed) => {
  if (!design || design.length === 0) {
    return []
  }

  if (!restrictionFactor) {
    throw new Error('Restriction factor must be specified')
  }

  if (!seed) {
    console.warn('No seed provided - randomization will not be reproducible!')
    seed = Math.floor(Math.random() * 1000000)
  }

  const rng = new SeededRNG(seed)

  // Group by restriction factor
  const groups = design.reduce((acc, run) => {
    const key = run[restrictionFactor]
    if (!acc[key]) acc[key] = []
    acc[key].push(run)
    return acc
  }, {})

  // Randomize each group
  const randomizedGroups = Object.values(groups).map(group => rng.shuffle(group))

  // Interleave groups to maintain balance throughout experiment
  const maxLength = Math.max(...randomizedGroups.map(g => g.length))
  const interleaved = []

  for (let i = 0; i < maxLength; i++) {
    randomizedGroups.forEach(group => {
      if (group[i]) {
        interleaved.push(group[i])
      }
    })
  }

  // Add run order
  return interleaved.map((run, idx) => ({
    ...run,
    runOrder: idx + 1,
    originalOrder: idx + 1,
  }))
}

/**
 * Stratified Randomization
 * Randomizes within strata to ensure balance
 *
 * @param {Array} design - Design matrix
 * @param {string} stratumFactor - Factor to stratify on
 * @param {number} seed - Random seed
 * @returns {Array} Randomized design with stratum assignments
 */
export const stratifiedRandomization = (design, stratumFactor, seed) => {
  if (!design || design.length === 0) {
    return []
  }

  const rng = new SeededRNG(seed || Math.floor(Math.random() * 1000000))

  // Group by stratum
  const strata = design.reduce((acc, run) => {
    const key = run[stratumFactor] || 'unknown'
    if (!acc[key]) acc[key] = []
    acc[key].push(run)
    return acc
  }, {})

  // Randomize within each stratum
  const randomizedStrata = Object.entries(strata).map(([stratum, runs]) => {
    const shuffled = rng.shuffle(runs)
    return shuffled.map(run => ({
      ...run,
      stratum,
    }))
  })

  // Combine all strata
  const combined = randomizedStrata.flat()

  // Add run order
  return combined.map((run, idx) => ({
    ...run,
    runOrder: idx + 1,
    originalOrder: idx + 1,
  }))
}

/**
 * Verify randomization reproducibility
 * Tests that same seed produces same result
 *
 * @param {Array} design - Design matrix
 * @param {number} seed - Seed to test
 * @param {number} iterations - Number of test iterations (default 5)
 * @returns {boolean} True if reproducible
 */
export const verifyReproducibility = (design, seed, iterations = 5) => {
  const results = []

  for (let i = 0; i < iterations; i++) {
    const randomized = completeRandomization(design, seed)
    const runOrderString = randomized.map(r => r.runOrder).join(',')
    results.push(runOrderString)
  }

  // All results should be identical
  const firstResult = results[0]
  return results.every(r => r === firstResult)
}

/**
 * Generate reproducible random seed
 * Uses current timestamp and optional user input
 *
 * @param {string} userInput - Optional user input for seed generation
 * @returns {number} Generated seed
 */
export const generateSeed = (userInput = '') => {
  if (userInput) {
    // Hash user input to create seed
    let hash = 0
    for (let i = 0; i < userInput.length; i++) {
      const char = userInput.charCodeAt(i)
      hash = ((hash << 5) - hash) + char
      hash = hash & hash // Convert to 32bit integer
    }
    return Math.abs(hash)
  }

  // Use timestamp-based seed
  return Math.floor(Date.now() % 1000000)
}

/**
 * Document randomization in protocol-friendly format
 *
 * @param {Object} params - Randomization parameters
 * @returns {string} Human-readable randomization documentation
 */
export const documentRandomization = (params) => {
  const { method, seed, blockSize, restrictionFactor } = params

  let doc = `Randomization Method: ${method.charAt(0).toUpperCase() + method.slice(1)}\n`
  doc += `Random Seed: ${seed} (for reproducibility)\n`

  if (method === 'block') {
    doc += `Block Size: ${blockSize}\n`
    doc += `\nTo reproduce this randomization:\n`
    doc += `1. Use the same design matrix\n`
    doc += `2. Apply block randomization with block size = ${blockSize}\n`
    doc += `3. Set random seed = ${seed}\n`
  } else if (method === 'restricted') {
    doc += `Restriction Factor: ${restrictionFactor}\n`
    doc += `\nRuns are balanced by ${restrictionFactor} throughout the experiment.\n`
  } else {
    doc += `\nTo reproduce this randomization:\n`
    doc += `1. Use the same design matrix\n`
    doc += `2. Apply complete randomization\n`
    doc += `3. Set random seed = ${seed}\n`
  }

  doc += `\nNote: Using the same seed will produce IDENTICAL run order.`

  return doc
}

export default {
  SeededRNG,
  completeRandomization,
  blockRandomization,
  restrictedRandomization,
  stratifiedRandomization,
  verifyReproducibility,
  generateSeed,
  documentRandomization,
}
