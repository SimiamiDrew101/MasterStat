const CubePlot = ({ data, factors }) => {
  if (!data || data.length === 0 || !factors || factors.length !== 3) return null

  // SVG dimensions
  const width = 500
  const height = 500
  const centerX = width / 2
  const centerY = height / 2

  // Cube dimensions (isometric projection)
  const cubeSize = 150

  // Isometric projection angles
  const angleX = Math.PI / 6  // 30 degrees
  const angleY = Math.PI / 6  // 30 degrees

  // Convert 3D coordinates to 2D isometric
  const project = (x, y, z) => {
    const isoX = centerX + (x - y) * cubeSize * Math.cos(angleX)
    const isoY = centerY + (x + y) * cubeSize * Math.sin(angleY) - z * cubeSize
    return { x: isoX, y: isoY }
  }

  // Define cube vertices (all 8 corners)
  const vertices = [
    { x: 0, y: 0, z: 0, label: '000' },
    { x: 1, y: 0, z: 0, label: '100' },
    { x: 0, y: 1, z: 0, label: '010' },
    { x: 1, y: 1, z: 0, label: '110' },
    { x: 0, y: 0, z: 1, label: '001' },
    { x: 1, y: 0, z: 1, label: '101' },
    { x: 0, y: 1, z: 1, label: '011' },
    { x: 1, y: 1, z: 1, label: '111' }
  ]

  // Define cube edges
  const edges = [
    [0, 1], [0, 2], [0, 4], // from origin
    [1, 3], [1, 5], // from 100
    [2, 3], [2, 6], // from 010
    [3, 7], // from 110
    [4, 5], [4, 6], // from 001
    [5, 7], // from 101
    [6, 7]  // from 011
  ]

  // Project all vertices
  const projectedVertices = vertices.map(v => ({
    ...v,
    projected: project(v.x, v.y, v.z)
  }))

  // Match data to vertices
  const verticesWithData = projectedVertices.map(v => {
    const dataPoint = data.find(d => d.x === v.x && d.y === v.y && d.z === v.z)
    return {
      ...v,
      response: dataPoint ? dataPoint.response : null,
      dataLabel: dataPoint ? dataPoint.label : null
    }
  })

  // Find min/max response for color scaling
  const responses = data.map(d => d.response)
  const minResponse = Math.min(...responses)
  const maxResponse = Math.max(...responses)

  // Color scale (blue to red)
  const getColor = (value) => {
    if (value === null) return '#64748b'
    const normalized = (value - minResponse) / (maxResponse - minResponse)
    const r = Math.round(normalized * 255)
    const b = Math.round((1 - normalized) * 255)
    return `rgb(${r}, 100, ${b})`
  }

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <h4 className="text-gray-100 font-semibold mb-4">Cube Plot (2³ Factorial Design)</h4>
      <p className="text-gray-300 text-sm mb-4">
        Each vertex shows the response at that factor combination. Colors indicate response magnitude.
      </p>

      <div className="flex justify-center">
        <svg width={width} height={height} className="overflow-visible">
          {/* Draw edges */}
          {edges.map((edge, i) => {
            const start = projectedVertices[edge[0]].projected
            const end = projectedVertices[edge[1]].projected
            return (
              <line
                key={`edge-${i}`}
                x1={start.x}
                y1={start.y}
                x2={end.x}
                y2={end.y}
                stroke="#64748b"
                strokeWidth={2}
                opacity={0.5}
              />
            )
          })}

          {/* Draw vertices and labels */}
          {verticesWithData.map((vertex, i) => {
            const pos = vertex.projected
            const color = getColor(vertex.response)

            return (
              <g key={i}>
                {/* Vertex circle */}
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={vertex.response !== null ? 20 : 8}
                  fill={color}
                  stroke="#1e293b"
                  strokeWidth={2}
                />

                {/* Response value */}
                {vertex.response !== null && (
                  <text
                    x={pos.x}
                    y={pos.y + 4}
                    textAnchor="middle"
                    fill="#ffffff"
                    fontSize="12"
                    fontWeight="bold"
                  >
                    {vertex.response.toFixed(1)}
                  </text>
                )}

                {/* Factor combination label */}
                {vertex.dataLabel && (
                  <text
                    x={pos.x}
                    y={pos.y + 35}
                    textAnchor="middle"
                    fill="#e2e8f0"
                    fontSize="10"
                  >
                    {vertex.dataLabel}
                  </text>
                )}
              </g>
            )
          })}

          {/* Axis labels */}
          <g>
            {/* X-axis (Factor A) */}
            <text
              x={centerX + cubeSize * 1.2}
              y={centerY + 20}
              fill="#3b82f6"
              fontSize="14"
              fontWeight="600"
            >
              {factors[0]} →
            </text>

            {/* Y-axis (Factor B) */}
            <text
              x={centerX - cubeSize * 1.2}
              y={centerY + 20}
              fill="#10b981"
              fontSize="14"
              fontWeight="600"
            >
              ← {factors[1]}
            </text>

            {/* Z-axis (Factor C) */}
            <text
              x={centerX - 20}
              y={centerY - cubeSize * 1.2}
              fill="#ef4444"
              fontSize="14"
              fontWeight="600"
            >
              ↑ {factors[2]}
            </text>
          </g>
        </svg>
      </div>

      {/* Legend */}
      <div className="mt-6 flex justify-center">
        <div className="bg-slate-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-4">
            <span className="text-gray-300 text-sm font-semibold">Response:</span>
            <div className="flex items-center space-x-2">
              <div className="w-8 h-6 rounded" style={{background: 'linear-gradient(to right, rgb(0,100,255), rgb(255,100,0))'}}></div>
              <span className="text-gray-300 text-sm">
                {minResponse.toFixed(1)} → {maxResponse.toFixed(1)}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong className="text-gray-100">Interpretation:</strong> The cube shows all 8 treatment combinations
          for a 2³ design. The response value at each vertex indicates the outcome. Look for patterns across
          edges to identify main effects and along face diagonals for interactions.
        </p>
      </div>
    </div>
  )
}

export default CubePlot
