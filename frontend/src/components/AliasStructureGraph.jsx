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
                     effect.includes('×') ? 'interaction' :
                     effect.length === 2 ? 'two-way' : 'higher'

        nodes.push({
          id: effect,
          type: type,
          label: effect
        })
      }

      // Add aliases as nodes and create links
      aliasList.forEach((alias, idx) => {
        if (idx === 0) return // Skip the first one as it's the effect itself

        if (!nodeSet.has(alias)) {
          nodeSet.add(alias)

          const aliasType = alias.length === 1 ? 'main' :
                           alias.includes('×') ? 'interaction' :
                           alias.length === 2 ? 'two-way' : 'higher'

          nodes.push({
            id: alias,
            type: aliasType,
            label: alias
          })
        }

        // Create link between effect and alias
        links.push({
          source: effect,
          target: alias
        })
      })
    })

    return { nodes, links }
  }, [aliasStructure])

  // Simple force-directed layout using radial positioning
  const layout = useMemo(() => {
    const width = 800
    const height = 600
    const centerX = width / 2
    const centerY = height / 2

    // Separate nodes by type
    const mainEffects = graphData.nodes.filter(n => n.type === 'main')
    const twoWayInteractions = graphData.nodes.filter(n => n.type === 'two-way' || n.type === 'interaction')
    const higherOrder = graphData.nodes.filter(n => n.type === 'higher')

    const positionedNodes = []

    // Position main effects in inner circle
    mainEffects.forEach((node, idx) => {
      const angle = (idx / mainEffects.length) * 2 * Math.PI
      const radius = 120
      positionedNodes.push({
        ...node,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle)
      })
    })

    // Position 2-way interactions in middle circle
    twoWayInteractions.forEach((node, idx) => {
      const angle = (idx / twoWayInteractions.length) * 2 * Math.PI
      const radius = 220
      positionedNodes.push({
        ...node,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle)
      })
    })

    // Position higher-order interactions in outer circle
    higherOrder.forEach((node, idx) => {
      const angle = (idx / higherOrder.length) * 2 * Math.PI
      const radius = 300
      positionedNodes.push({
        ...node,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle)
      })
    })

    return { nodes: positionedNodes, width, height, centerX, centerY }
  }, [graphData])

  // Get color based on node type
  const getNodeColor = (type) => {
    switch (type) {
      case 'main':
        return '#3b82f6' // Blue for main effects
      case 'two-way':
      case 'interaction':
        return '#10b981' // Green for 2-way interactions
      case 'higher':
        return '#ef4444' // Red for higher-order
      default:
        return '#64748b'
    }
  }

  // Get resolution color
  const getResolutionColor = (resolution) => {
    if (resolution === 'III') return '#ef4444' // Red
    if (resolution === 'IV') return '#f59e0b' // Orange
    if (resolution === 'V' || resolution.includes('V')) return '#10b981' // Green
    return '#64748b'
  }

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="mb-4">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-gray-100 font-semibold text-lg">Alias Structure Visualization</h4>
            <p className="text-gray-400 text-sm mt-1">
              Confounding patterns for {aliasStructure.n_factors}-factor {aliasStructure.fraction || 'fractional'} design
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-gray-400 text-sm">Resolution:</span>
            <span
              className="px-3 py-1 rounded-full font-mono font-bold"
              style={{
                backgroundColor: `${getResolutionColor(aliasStructure.resolution)}20`,
                color: getResolutionColor(aliasStructure.resolution),
                border: `2px solid ${getResolutionColor(aliasStructure.resolution)}`
              }}
            >
              {aliasStructure.resolution}
            </span>
          </div>
        </div>
      </div>

      {/* Defining Relations */}
      {aliasStructure.defining_relations && aliasStructure.defining_relations.length > 0 && (
        <div className="mb-4 bg-slate-800/50 rounded-lg p-4">
          <h5 className="text-gray-100 font-semibold text-sm mb-2">Defining Relations</h5>
          <div className="flex flex-wrap gap-2">
            {aliasStructure.defining_relations.map((relation, idx) => (
              <span key={idx} className="px-3 py-1 bg-purple-900/30 text-purple-200 rounded text-sm font-mono border border-purple-700/30">
                {relation}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Graph Visualization */}
      <div className="bg-slate-800/50 rounded-lg p-4 overflow-auto">
        <svg width={layout.width} height={layout.height} className="mx-auto">
          {/* Draw links first (behind nodes) */}
          {graphData.links.map((link, idx) => {
            const sourceNode = layout.nodes.find(n => n.id === link.source)
            const targetNode = layout.nodes.find(n => n.id === link.target)

            if (!sourceNode || !targetNode) return null

            return (
              <line
                key={`link-${idx}`}
                x1={sourceNode.x}
                y1={sourceNode.y}
                x2={targetNode.x}
                y2={targetNode.y}
                stroke="#64748b"
                strokeWidth={2}
                strokeOpacity={0.4}
                strokeDasharray="4,4"
              />
            )
          })}

          {/* Draw nodes */}
          {layout.nodes.map((node, idx) => {
            const color = getNodeColor(node.type)

            return (
              <g key={`node-${idx}`}>
                {/* Node circle */}
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={node.type === 'main' ? 28 : 24}
                  fill={color}
                  stroke="#1e293b"
                  strokeWidth={3}
                  opacity={0.9}
                />

                {/* Node label */}
                <text
                  x={node.x}
                  y={node.y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="#ffffff"
                  fontSize={node.type === 'main' ? "14" : "11"}
                  fontWeight="bold"
                  fontFamily="monospace"
                >
                  {node.label}
                </text>

                {/* Tooltip title (appears on hover) */}
                <title>
                  {node.label}
                  {aliasStructure.aliases[node.id] &&
                    ` ’ Aliased with: ${aliasStructure.aliases[node.id].slice(1).join(', ')}`
                  }
                </title>
              </g>
            )
          })}

          {/* Center label */}
          <text
            x={layout.centerX}
            y={layout.centerY}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="#64748b"
            fontSize="12"
            fontWeight="600"
          >
            {aliasStructure.n_runs} runs
          </text>
        </svg>
      </div>

      {/* Legend */}
      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <h5 className="text-gray-100 font-semibold text-sm mb-3">Legend</h5>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 rounded-full bg-blue-500 border-2 border-slate-900"></div>
              <span className="text-gray-300 text-sm">Main Effects (inner circle)</span>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 rounded-full bg-green-500 border-2 border-slate-900"></div>
              <span className="text-gray-300 text-sm">2-Way Interactions (middle circle)</span>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 rounded-full bg-red-500 border-2 border-slate-900"></div>
              <span className="text-gray-300 text-sm">Higher-Order (outer circle)</span>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className="w-12 h-0.5 bg-slate-500" style={{ borderTop: '2px dashed #64748b' }}></div>
              <span className="text-gray-300 text-sm">Confounded/Aliased</span>
            </div>
          </div>
        </div>
      </div>

      {/* Interpretation */}
      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <h5 className="text-gray-100 font-semibold text-sm mb-2">Interpretation</h5>
        <div className="text-gray-300 text-sm space-y-2">
          <p>
            <strong className="text-gray-100">Alias Structure:</strong> In fractional factorial designs, some effects are confounded (aliased) with others.
            Dashed lines connect effects that cannot be separated in this design.
          </p>
          <ul className="list-disc list-inside space-y-1 ml-2">
            <li>
              <strong className="text-blue-400">Resolution {aliasStructure.resolution}:</strong>{' '}
              {aliasStructure.resolution === 'III' && 'Main effects are aliased with 2-way interactions. Assume interactions are negligible.'}
              {aliasStructure.resolution === 'IV' && 'Main effects are clear, but 2-way interactions are aliased with each other.'}
              {(aliasStructure.resolution === 'V' || aliasStructure.resolution.includes('V')) && 'Main effects and 2-way interactions are clear. Very good design.'}
            </li>
            <li>
              <strong>Generators:</strong> {aliasStructure.generators?.join(', ') || 'N/A'}
            </li>
            <li>
              <strong>Total Runs:</strong> {aliasStructure.n_runs} (compared to {Math.pow(2, aliasStructure.n_factors)} for full factorial)
            </li>
          </ul>
          {aliasStructure.resolution === 'III' && (
            <p className="text-yellow-400 text-sm mt-2">
                <strong>Warning:</strong> Resolution III designs confound main effects with 2-way interactions.
              Use only when you're confident interactions are negligible, or consider a foldover design.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

export default AliasStructureGraph
