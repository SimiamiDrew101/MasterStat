import { useRef } from 'react'
import { Download } from 'lucide-react'
import { exportSvgToPng } from '../utils/exportChart'

const ResponseSurface3D = ({ surfaceData, factor1, factor2, responseName }) => {
  const svgRef = useRef(null)

  if (!surfaceData || surfaceData.length === 0) return null

  // SVG dimensions
  const width = 800
  const height = 700
  const margin = { top: 60, right: 50, bottom: 60, left: 50 }

  // Extract unique x and y values
  const xValues = [...new Set(surfaceData.map(d => d.x))].sort((a, b) => a - b)
  const yValues = [...new Set(surfaceData.map(d => d.y))].sort((a, b) => a - b)

  // Create 2D grid of z values
  const zGrid = []
  for (let i = 0; i < yValues.length; i++) {
    zGrid[i] = []
    for (let j = 0; j < xValues.length; j++) {
      const point = surfaceData.find(d => d.x === xValues[j] && d.y === yValues[i])
      zGrid[i][j] = point ? point.z : 0
    }
  }

  // Find min/max for z values
  const allZ = surfaceData.map(d => d.z)
  const zMin = Math.min(...allZ)
  const zMax = Math.max(...allZ)
  const zRange = zMax - zMin

  // Isometric projection parameters
  const angle = Math.PI / 6 // 30 degrees
  const scale = 200

  // Project 3D point to 2D isometric
  const project = (x, y, z) => {
    // Normalize coordinates
    const normX = (x - xValues[0]) / (xValues[xValues.length - 1] - xValues[0]) - 0.5
    const normY = (y - yValues[0]) / (yValues[yValues.length - 1] - yValues[0]) - 0.5
    const normZ = zRange > 0 ? (z - zMin) / zRange : 0

    // Isometric projection
    const iso2dX = (normX - normY) * Math.cos(angle) * scale
    const iso2dY = (normX + normY) * Math.sin(angle) * scale - normZ * scale * 0.8

    return {
      x: width / 2 + iso2dX,
      y: height / 2 - iso2dY + margin.top
    }
  }

  // Color scale
  const getColor = (value) => {
    const normalized = (value - zMin) / (zMax - zMin)
    // Blue to green to red gradient
    if (normalized < 0.5) {
      const t = normalized * 2
      const r = Math.round(100 * t)
      const g = Math.round(150 + 105 * t)
      const b = Math.round(255 * (1 - t))
      return `rgb(${r}, ${g}, ${b})`
    } else {
      const t = (normalized - 0.5) * 2
      const r = Math.round(100 + 155 * t)
      const g = Math.round(255 * (1 - t))
      const b = 50
      return `rgb(${r}, ${g}, ${b})`
    }
  }

  // Generate surface mesh
  const meshPolygons = []
  for (let i = 0; i < yValues.length - 1; i++) {
    for (let j = 0; j < xValues.length - 1; j++) {
      const x1 = xValues[j]
      const x2 = xValues[j + 1]
      const y1 = yValues[i]
      const y2 = yValues[i + 1]
      const z11 = zGrid[i][j]
      const z12 = zGrid[i][j + 1]
      const z21 = zGrid[i + 1][j]
      const z22 = zGrid[i + 1][j + 1]

      // Project corners
      const p11 = project(x1, y1, z11)
      const p12 = project(x2, y1, z12)
      const p21 = project(x1, y2, z21)
      const p22 = project(x2, y2, z22)

      // Average z for coloring
      const avgZ = (z11 + z12 + z21 + z22) / 4

      // Calculate depth for sorting (painter's algorithm)
      const depth = (p11.y + p12.y + p21.y + p22.y) / 4

      // Create quad as two triangles
      meshPolygons.push({
        points: [p11, p12, p22, p21],
        color: getColor(avgZ),
        depth: depth
      })
    }
  }

  // Sort by depth (back to front)
  meshPolygons.sort((a, b) => b.depth - a.depth)

  return (
    <div className="bg-slate-700/50 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-gray-100 font-semibold">3D Response Surface</h4>
        <button
          type="button"
          onClick={() => {
            if (svgRef.current) {
              exportSvgToPng(svgRef.current, `3d-surface-${new Date().toISOString().split('T')[0]}`)
            }
          }}
          className="px-3 py-2 rounded-lg text-sm font-medium bg-slate-600/50 text-gray-300 hover:bg-slate-600 transition-all flex items-center space-x-2"
          title="Export as PNG"
        >
          <Download className="w-4 h-4" />
          <span>Export PNG</span>
        </button>
      </div>

      <div className="flex justify-center">
        <svg ref={svgRef} width={width} height={height}>
          <defs>
            <filter id="shadow">
              <feDropShadow dx="2" dy="2" stdDeviation="3" floodOpacity="0.3" />
            </filter>
          </defs>

          {/* Title */}
          <text
            x={width / 2}
            y={30}
            textAnchor="middle"
            fill="#f1f5f9"
            fontSize="18"
            fontWeight="600"
          >
            {responseName} Response Surface
          </text>

          {/* Draw surface mesh */}
          {meshPolygons.map((poly, idx) => (
            <polygon
              key={idx}
              points={poly.points.map(p => `${p.x},${p.y}`).join(' ')}
              fill={poly.color}
              stroke="#1e293b"
              strokeWidth={0.5}
              opacity={0.9}
            />
          ))}

          {/* Draw base grid */}
          {xValues.map((x, i) => {
            const p1 = project(x, yValues[0], zMin - zRange * 0.1)
            const p2 = project(x, yValues[yValues.length - 1], zMin - zRange * 0.1)
            return (
              <line
                key={`grid-x-${i}`}
                x1={p1.x}
                y1={p1.y}
                x2={p2.x}
                y2={p2.y}
                stroke="#475569"
                strokeWidth={0.5}
                strokeDasharray="2,2"
              />
            )
          })}

          {yValues.map((y, i) => {
            const p1 = project(xValues[0], y, zMin - zRange * 0.1)
            const p2 = project(xValues[xValues.length - 1], y, zMin - zRange * 0.1)
            return (
              <line
                key={`grid-y-${i}`}
                x1={p1.x}
                y1={p1.y}
                x2={p2.x}
                y2={p2.y}
                stroke="#475569"
                strokeWidth={0.5}
                strokeDasharray="2,2"
              />
            )
          })}

          {/* Axis labels */}
          <text
            x={project(xValues[xValues.length - 1], yValues[0], zMin - zRange * 0.1).x + 30}
            y={project(xValues[xValues.length - 1], yValues[0], zMin - zRange * 0.1).y}
            fill="#e2e8f0"
            fontSize="14"
            fontWeight="600"
          >
            {factor1}
          </text>

          <text
            x={project(xValues[0], yValues[yValues.length - 1], zMin - zRange * 0.1).x - 50}
            y={project(xValues[0], yValues[yValues.length - 1], zMin - zRange * 0.1).y}
            fill="#e2e8f0"
            fontSize="14"
            fontWeight="600"
          >
            {factor2}
          </text>

          <text
            x={project(xValues[0], yValues[0], zMax).x - 60}
            y={project(xValues[0], yValues[0], zMax).y}
            fill="#e2e8f0"
            fontSize="14"
            fontWeight="600"
          >
            {responseName}
          </text>

          {/* Color legend */}
          <g transform={`translate(${width - 100}, ${height - 200})`}>
            <text
              x={0}
              y={-10}
              fill="#e2e8f0"
              fontSize="12"
              fontWeight="600"
            >
              Response
            </text>
            {[0, 0.25, 0.5, 0.75, 1].map((frac, i) => {
              const value = zMin + frac * zRange
              const y = i * 35
              return (
                <g key={i}>
                  <rect
                    x={0}
                    y={y}
                    width={25}
                    height={30}
                    fill={getColor(value)}
                    stroke="#1e293b"
                    strokeWidth={1}
                  />
                  <text
                    x={30}
                    y={y + 20}
                    fill="#94a3b8"
                    fontSize="11"
                  >
                    {value.toFixed(1)}
                  </text>
                </g>
              )
            })}
          </g>
        </svg>
      </div>

      <div className="mt-4 bg-slate-800/50 rounded-lg p-4">
        <p className="text-gray-300 text-sm">
          <strong className="text-gray-100">Interpretation:</strong> This 3D surface plot visualizes the response as a function of two factors. The color gradient represents the response magnitude - red indicates higher values, blue indicates lower values. The shape of the surface reveals the nature of the optimization problem (convex, concave, or saddle point).
        </p>
      </div>
    </div>
  )
}

export default ResponseSurface3D
