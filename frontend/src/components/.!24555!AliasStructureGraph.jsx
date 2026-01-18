import { useMemo } from 'react'

/**
 * AliasStructureGraph component for visualizing confounding patterns
 * in fractional factorial designs
 */
const AliasStructureGraph = ({ aliasStructure }) => {
  if (!aliasStructure || !aliasStructure.aliases) return null

  // Prepare graph data
  const graphData = useMemo(() => {
    const nodes = []
    const links = []
    const nodeSet = new Set()

    // Process each effect and its aliases
    Object.entries(aliasStructure.aliases).forEach(([effect, aliasList]) => {
      // Add the main effect as a node if not already added
      if (!nodeSet.has(effect)) {
        nodeSet.add(effect)

        // Determine node type (main effect, 2-way interaction, or higher)
        const type = effect.length === 1 ? 'main' :
