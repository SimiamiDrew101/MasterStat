import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { Sliders, Target } from 'lucide-react'
import ResponseSurface3D from './ResponseSurface3D'
import ContourPlot from './ContourPlot'

const SlicedVisualization = ({
  factorNames,
  responseName,
  coefficients,
  experimentalData = null,
  optimizationResult = null,
  canonicalResult = null
}) => {
  // State for which factors to visualize
  const [xFactor, setXFactor] = useState(factorNames[0])
  const [yFactor, setYFactor] = useState(factorNames[1])

  // State for fixed factor values
  const [fixedValues, setFixedValues] = useState(() => {
    const initial = {}
    factorNames.forEach(factor => {
      if (factor !== xFactor && factor !== yFactor) {
        initial[factor] = 0 // Center point
      }
    })
    return initial
  })

  // Surface data for the selected slice
  const [surfaceData, setSurfaceData] = useState(null)

  // Update fixed values when factors change
  useEffect(() => {
    const newFixed = {}
    factorNames.forEach(factor => {
      if (factor !== xFactor && factor !== yFactor) {
        newFixed[factor] = fixedValues[factor] !== undefined ? fixedValues[factor] : 0
      }
    })
    setFixedValues(newFixed)
  }, [xFactor, yFactor, factorNames])

  // Generate surface data when factors or fixed values change
  useEffect(() => {
    generateSlicedSurface()
  }, [xFactor, yFactor, fixedValues, coefficients])

  const generateSlicedSurface = () => {
    if (!coefficients) return

    const points = []
    const steps = 20
    const coefObj = Object.fromEntries(
      Object.entries(coefficients).map(([k, v]) => [k, v.estimate || v])
    )

    for (let i = 0; i <= steps; i++) {
      for (let j = 0; j <= steps; j++) {
        const x = -2 + (4 * i) / steps
        const y = -2 + (4 * j) / steps

        // Build factor values object
        const factorValues = { ...fixedValues }
        factorValues[xFactor] = x
        factorValues[yFactor] = y

        // Calculate z using second-order model
        let z = coefObj['Intercept'] || 0

        // Linear terms
        factorNames.forEach(factor => {
          z += (coefObj[factor] || 0) * factorValues[factor]
        })

        // Quadratic terms
        factorNames.forEach(factor => {
          z += (coefObj[`I(${factor}**2)`] || 0) * factorValues[factor] * factorValues[factor]
        })

        // Interaction terms
        for (let m = 0; m < factorNames.length; m++) {
          for (let n = m + 1; n < factorNames.length; n++) {
            const factor1 = factorNames[m]
            const factor2 = factorNames[n]
            z += (coefObj[`${factor1}:${factor2}`] || 0) * factorValues[factor1] * factorValues[factor2]
          }
        }

        points.push({ x, y, z })
      }
    }

    setSurfaceData(points)
  }

  const handleFactorChange = (type, value) => {
    if (type === 'x') {
      if (value === yFactor) {
        // Swap
        setYFactor(xFactor)
      }
      setXFactor(value)
    } else {
      if (value === xFactor) {
        // Swap
        setXFactor(yFactor)
      }
      setYFactor(value)
    }
  }

  const handleFixedValueChange = (factor, value) => {
    setFixedValues(prev => ({
      ...prev,
      [factor]: parseFloat(value) || 0
    }))
  }

  const otherFactors = factorNames.filter(f => f !== xFactor && f !== yFactor)

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-2 mb-4">
          <Sliders className="w-5 h-5 text-orange-400" />
          <h3 className="text-xl font-bold text-gray-100">Slice Controls</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Select factors to visualize */}
          <div className="space-y-4">
            <h4 className="text-gray-200 font-semibold">Factors to Visualize</h4>

            <div>
              <label className="block text-gray-300 text-sm mb-2">X-Axis Factor</label>
              <select
                value={xFactor}
                onChange={(e) => handleFactorChange('x', e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
              >
                {factorNames.map(factor => (
                  <option key={factor} value={factor}>{factor}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-gray-300 text-sm mb-2">Y-Axis Factor</label>
              <select
                value={yFactor}
                onChange={(e) => handleFactorChange('y', e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
              >
                {factorNames.map(factor => (
                  <option key={factor} value={factor}>{factor}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Fixed factor values */}
          {otherFactors.length > 0 && (
            <div className="space-y-4">
              <h4 className="text-gray-200 font-semibold">Fixed Factor Values</h4>

              {otherFactors.map(factor => (
                <div key={factor}>
                  <label className="block text-gray-300 text-sm mb-2">
                    {factor} = {fixedValues[factor]?.toFixed(2) || '0.00'}
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min="-2"
                      max="2"
                      step="0.1"
                      value={fixedValues[factor] || 0}
                      onChange={(e) => handleFixedValueChange(factor, e.target.value)}
                      className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
                    />
                    <input
                      type="number"
                      min="-2"
                      max="2"
                      step="0.1"
                      value={fixedValues[factor] || 0}
                      onChange={(e) => handleFixedValueChange(factor, e.target.value)}
                      className="w-20 px-2 py-1 rounded bg-slate-700/50 text-gray-100 border border-slate-600 text-sm focus:outline-none focus:ring-2 focus:ring-orange-500"
                    />
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Low (-2)</span>
                    <span>Center (0)</span>
                    <span>High (+2)</span>
                  </div>
                </div>
              ))}

              <div className="mt-4 p-3 bg-blue-900/20 rounded-lg border border-blue-700/30">
                <p className="text-blue-200 text-xs">
                  <strong>Note:</strong> Adjust sliders to see how the response surface changes at different settings of the fixed factors.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Visualizations */}
      {surfaceData && (
        <div className="space-y-6">
          <ResponseSurface3D
            surfaceData={surfaceData}
            factor1={xFactor}
            factor2={yFactor}
            responseName={responseName}
          />
          <ContourPlot
            surfaceData={surfaceData}
            factor1={xFactor}
            factor2={yFactor}
            responseName={responseName}
            experimentalData={experimentalData ? experimentalData.filter(d =>
              // Only show points that match the current fixed values (within tolerance)
              otherFactors.every(factor =>
                Math.abs(d[factor] - fixedValues[factor]) < 0.3
              )
            ) : null}
            optimizationResult={optimizationResult}
            canonicalResult={canonicalResult}
          />
        </div>
      )}
    </div>
  )
}

export default SlicedVisualization
