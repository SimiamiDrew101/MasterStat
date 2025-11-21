import { useState, useEffect } from 'react'
import axios from 'axios'
import { Mountain, Plus, Trash2, Target, TrendingUp } from 'lucide-react'
import ResultCard from '../components/ResultCard'
import ResponseSurface3D from '../components/ResponseSurface3D'
import ContourPlot from '../components/ContourPlot'
import SlicedVisualization from '../components/SlicedVisualization'
import ResidualAnalysis from '../components/ResidualAnalysis'
import EnhancedANOVA from '../components/EnhancedANOVA'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const RSM = () => {
  // Design configuration
  const [designType, setDesignType] = useState('ccd')  // 'ccd' or 'box-behnken'
  const [ccdType, setCCDType] = useState('face-centered')  // 'face-centered', 'rotatable', 'inscribed'
  const [numFactors, setNumFactors] = useState(2)
  const [numCenterPoints, setNumCenterPoints] = useState(4)
  const [factorNames, setFactorNames] = useState(['X1', 'X2'])
  const [responseName, setResponseName] = useState('Y')
  const [alpha, setAlpha] = useState(0.05)

  // Data and results
  const [designData, setDesignData] = useState(null)
  const [tableData, setTableData] = useState([])
  const [modelResult, setModelResult] = useState(null)
  const [canonicalResult, setCanonicalResult] = useState(null)
  const [optimizationResult, setOptimizationResult] = useState(null)
  const [steepestAscentResult, setSteepestAscentResult] = useState(null)
  const [surfaceData, setSurfaceData] = useState(null)

  // UI state
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('design')  // 'design', 'model', 'optimize', 'visualize'

  // Update factor names when number changes
  useEffect(() => {
    const newFactorNames = Array.from({ length: numFactors }, (_, i) => `X${i + 1}`)
    console.log('Setting factor names:', newFactorNames)
    setFactorNames(newFactorNames)
  }, [numFactors])

  // Debug table data
  useEffect(() => {
    console.log('TableData updated:', tableData.length, 'rows')
    console.log('Factor names:', factorNames)
    console.log('Response name:', responseName)
  }, [tableData, factorNames, responseName])

  // Debug surface data
  useEffect(() => {
    console.log('surfaceData state changed:', surfaceData ? `${surfaceData.length} points` : 'null')
  }, [surfaceData])

  // Generate design
  const handleGenerateDesign = async () => {
    setLoading(true)
    setError(null)

    try {
      let response
      if (designType === 'ccd') {
        response = await axios.post(`${API_URL}/api/rsm/ccd/generate`, {
          n_factors: numFactors,
          design_type: ccdType,
          n_center: numCenterPoints
        })
      } else {
        response = await axios.post(`${API_URL}/api/rsm/box-behnken/generate`, {
          n_factors: numFactors,
          n_center: numCenterPoints
        })
      }

      setDesignData(response.data)

      // Convert to table format with empty response column
      const table = response.data.design_matrix.map(row => {
        const tableRow = factorNames.map(factor => row[factor] || 0)
        tableRow.push('')  // Empty response
        return tableRow
      })

      console.log('Generated table data:', table)
      console.log('Factor names:', factorNames)
      console.log('Table row example:', table[0])
      setTableData(table)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate design')
    } finally {
      setLoading(false)
    }
  }

  // Fit RSM model
  const handleFitModel = async () => {
    setLoading(true)
    setError(null)

    try {
      // Validate data - check if response column (last column) is filled
      console.log('Table data:', tableData)
      const validRows = tableData.filter(row => {
        const responseValue = row[row.length - 1]
        const isValid = responseValue !== '' && responseValue !== null && responseValue !== undefined
        console.log('Row:', row, 'Response:', responseValue, 'Valid:', isValid)
        return isValid
      })

      console.log('Valid rows:', validRows.length, 'Total rows:', tableData.length)

      if (validRows.length < 5) {
        throw new Error(`Need at least 5 complete data points for RSM (found ${validRows.length} rows with response values)`)
      }

      // Convert to API format
      const data = validRows.map(row => {
        const point = {}
        factorNames.forEach((factor, i) => {
          point[factor] = parseFloat(row[i])
        })
        point[responseName] = parseFloat(row[row.length - 1])
        return point
      })

      const response = await axios.post(`${API_URL}/api/rsm/fit-model`, {
        data: data,
        factors: factorNames,
        response: responseName,
        alpha: alpha
      })

      setModelResult(response.data)

      // Automatically perform canonical analysis
      if (response.data.coefficients) {
        const canonicalResponse = await axios.post(`${API_URL}/api/rsm/canonical-analysis`, {
          coefficients: Object.fromEntries(
            Object.entries(response.data.coefficients).map(([k, v]) => [k, v.estimate])
          ),
          factors: factorNames
        })
        setCanonicalResult(canonicalResponse.data)
      }

      // Generate surface data for visualization (2 factors for simple viz)
      if (numFactors === 2) {
        console.log('Generating surface data for 2-factor visualization...')
        console.log('Coefficients:', response.data.coefficients)
        try {
          generateSurfaceData(response.data.coefficients)
          console.log('Surface data generated successfully')
        } catch (surfaceError) {
          console.error('Error generating surface data:', surfaceError)
          setError('Visualization data generation failed: ' + surfaceError.message)
        }
      } else {
        console.log('Multi-factor design - will use sliced visualization')
      }

      setActiveTab('model')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to fit model')
    } finally {
      setLoading(false)
    }
  }

  // Generate surface data for visualization
  const generateSurfaceData = (coefficients) => {
    console.log('generateSurfaceData called with coefficients:', coefficients)
    const points = []
    const steps = 20
    const coefObj = Object.fromEntries(
      Object.entries(coefficients).map(([k, v]) => [k, v.estimate || v])
    )
    console.log('Coefficient object:', coefObj)
    console.log('Factor names:', factorNames)

    for (let i = 0; i <= steps; i++) {
      for (let j = 0; j <= steps; j++) {
        const x = -2 + (4 * i) / steps
        const y = -2 + (4 * j) / steps

        // Calculate z using second-order model
        let z = coefObj['Intercept'] || 0
        z += (coefObj[factorNames[0]] || 0) * x
        z += (coefObj[factorNames[1]] || 0) * y
        z += (coefObj[`I(${factorNames[0]}**2)`] || 0) * x * x
        z += (coefObj[`I(${factorNames[1]}**2)`] || 0) * y * y
        z += (coefObj[`${factorNames[0]}:${factorNames[1]}`] || 0) * x * y

        points.push({ x, y, z })
      }
    }

    console.log('Generated surface points:', points.length, 'points')
    console.log('First few points:', points.slice(0, 3))
    setSurfaceData(points)
    console.log('surfaceData state updated')
  }

  // Optimize response
  const handleOptimize = async (target) => {
    setLoading(true)
    setError(null)

    try {
      if (!modelResult || !modelResult.coefficients) {
        throw new Error('Fit a model first')
      }

      const response = await axios.post(`${API_URL}/api/rsm/optimize`, {
        coefficients: Object.fromEntries(
          Object.entries(modelResult.coefficients).map(([k, v]) => [k, v.estimate])
        ),
        factors: factorNames,
        target: target
      })

      setOptimizationResult(response.data)
      setActiveTab('optimize')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Optimization failed')
    } finally {
      setLoading(false)
    }
  }

  // Calculate steepest ascent
  const handleSteepestAscent = async () => {
    setLoading(true)
    setError(null)

    try {
      if (!modelResult || !modelResult.coefficients) {
        throw new Error('Fit a model first')
      }

      // Use linear coefficients only for steepest ascent
      const linearCoefs = {}
      factorNames.forEach(factor => {
        if (modelResult.coefficients[factor]) {
          linearCoefs[factor] = modelResult.coefficients[factor].estimate
        }
      })

      // Use origin as current point
      const currentPoint = {}
      factorNames.forEach(factor => {
        currentPoint[factor] = 0
      })

      const response = await axios.post(`${API_URL}/api/rsm/steepest-ascent`, {
        current_point: currentPoint,
        coefficients: linearCoefs,
        step_size: 0.5,
        n_steps: 10
      })

      setSteepestAscentResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Steepest ascent failed')
    } finally {
      setLoading(false)
    }
  }

  // Table editing
  const handleCellChange = (rowIndex, colIndex, value) => {
    const newData = [...tableData]
    newData[rowIndex][colIndex] = value
    setTableData(newData)
  }

  const addRow = () => {
    const newRow = Array(factorNames.length + 1).fill('')
    setTableData([...tableData, newRow])
  }

  const removeRow = (rowIndex) => {
    if (tableData.length > 1) {
      setTableData(tableData.filter((_, idx) => idx !== rowIndex))
    }
  }

  // Fill test data for demonstration
  const fillTestData = () => {
    const newData = tableData.map(row => {
      // Keep factor values, generate random response
      const newRow = [...row]
      const factors = row.slice(0, -1) // All except last column
      // Generate response based on a simple quadratic model for testing
      let response = 10
      factors.forEach((factor, i) => {
        response += 5 * factor + 2 * factor * factor
      })
      // Add some noise
      response += (Math.random() - 0.5) * 5
      newRow[newRow.length - 1] = response.toFixed(2)
      return newRow
    })
    setTableData(newData)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-900/30 to-red-900/30 backdrop-blur-lg rounded-2xl p-8 border border-orange-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <Mountain className="w-10 h-10 text-orange-400" />
          <h2 className="text-4xl font-bold text-gray-100">Response Surface Methodology</h2>
        </div>
        <p className="text-gray-300 text-lg">
          Design experiments, fit second-order models, and optimize responses using Central Composite and Box-Behnken designs.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-2 bg-slate-800/50 p-2 rounded-lg">
        <button
          onClick={() => setActiveTab('design')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'design'
              ? 'bg-orange-600 text-white'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
        >
          1. Design
        </button>
        <button
          onClick={() => setActiveTab('model')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'model'
              ? 'bg-orange-600 text-white'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
          disabled={!modelResult}
        >
          2. Model & Analysis
        </button>
        <button
          onClick={() => setActiveTab('optimize')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'optimize'
              ? 'bg-orange-600 text-white'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
          disabled={!modelResult}
        >
          3. Optimization
        </button>
        <button
          onClick={() => setActiveTab('visualize')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'visualize'
              ? 'bg-orange-600 text-white'
              : !modelResult
              ? 'bg-slate-700/50 text-gray-500 cursor-not-allowed'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
          disabled={!modelResult}
          title={!modelResult ? 'Fit a model first' : 'View response surface visualization'}
        >
          4. Visualize
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200 font-medium">Error: {error}</p>
        </div>
      )}

      {/* Tab Content */}
      {activeTab === 'design' && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-2xl font-bold text-gray-100 mb-6">Experimental Design</h3>

          <div className="space-y-6">
            {/* Design Type Selection */}
            <div>
              <label className="block text-gray-100 font-medium mb-2">Design Type</label>
              <select
                value={designType}
                onChange={(e) => setDesignType(e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
              >
                <option value="ccd">Central Composite Design (CCD)</option>
                <option value="box-behnken">Box-Behnken Design</option>
              </select>
            </div>

            {/* CCD Type Selection */}
            {designType === 'ccd' && (
              <div>
                <label className="block text-gray-100 font-medium mb-2">CCD Type</label>
                <select
                  value={ccdType}
                  onChange={(e) => setCCDType(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
                >
                  <option value="face-centered">Face-Centered (α = 1)</option>
                  <option value="rotatable">Rotatable (α = 2^(k/4))</option>
                  <option value="inscribed">Inscribed (α = 1)</option>
                </select>
                <p className="text-gray-400 text-xs mt-1">
                  {ccdType === 'face-centered' && 'Axial points on the faces of the factorial cube'}
                  {ccdType === 'rotatable' && 'Equal prediction variance at all points equidistant from center'}
                  {ccdType === 'inscribed' && 'All points within the original cube'}
                </p>
              </div>
            )}

            {/* Number of Factors */}
            <div>
              <label className="block text-gray-100 font-medium mb-2">Number of Factors</label>
              <input
                type="number"
                min={designType === 'box-behnken' ? 3 : 2}
                max={designType === 'box-behnken' ? 7 : 6}
                value={numFactors}
                onChange={(e) => setNumFactors(parseInt(e.target.value) || 2)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
              <p className="text-gray-400 text-xs mt-1">
                {designType === 'box-behnken'
                  ? 'Box-Behnken designs require 3-7 factors'
                  : 'CCD supports 2-6 factors'}
              </p>
            </div>

            {/* Center Points */}
            <div>
              <label className="block text-gray-100 font-medium mb-2">Number of Center Points</label>
              <input
                type="number"
                min={1}
                max={10}
                value={numCenterPoints}
                onChange={(e) => setNumCenterPoints(parseInt(e.target.value) || 4)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
              <p className="text-gray-400 text-xs mt-1">
                Center points enable curvature detection and pure error estimation
              </p>
            </div>

            {/* Response Name */}
            <div>
              <label className="block text-gray-100 font-medium mb-2">Response Variable Name</label>
              <input
                type="text"
                value={responseName}
                onChange={(e) => setResponseName(e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-orange-500"
              />
            </div>

            {/* Generate Design Button */}
            <button
              onClick={handleGenerateDesign}
              disabled={loading}
              className="w-full bg-orange-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-orange-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Generating...' : 'Generate Design'}
            </button>

            {/* Design Information */}
            {designData && (
              <div className="bg-gradient-to-r from-orange-900/20 to-red-900/20 rounded-lg p-5 border border-orange-700/30">
                <h4 className="text-orange-200 font-semibold mb-3">Design Properties</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Design Type:</span>
                    <span className="text-gray-100 ml-2 font-medium">{designData.design_type}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Total Runs:</span>
                    <span className="text-gray-100 ml-2 font-medium">{designData.n_runs.total}</span>
                  </div>
                  {designType === 'ccd' && (
                    <>
                      <div>
                        <span className="text-gray-400">Factorial Points:</span>
                        <span className="text-gray-100 ml-2">{designData.n_runs.factorial}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Axial Points:</span>
                        <span className="text-gray-100 ml-2">{designData.n_runs.axial}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Alpha (α):</span>
                        <span className="text-gray-100 ml-2">{designData.alpha}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Rotatable:</span>
                        <span className="text-gray-100 ml-2">{designData.properties.rotatable ? 'Yes' : 'No'}</span>
                      </div>
                    </>
                  )}
                  {designType === 'box-behnken' && (
                    <div>
                      <span className="text-gray-400">Edge Points:</span>
                      <span className="text-gray-100 ml-2">{designData.n_runs.edge_points}</span>
                    </div>
                  )}
                  <div className="col-span-2">
                    <span className="text-gray-400">Description:</span>
                    <span className="text-gray-100 ml-2">{designData.properties.description || designData.properties.alpha_description}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Data Entry Table */}
            {tableData.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-gray-100 font-semibold text-lg">Experimental Data</h4>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={fillTestData}
                      disabled={!designData}
                      className="flex items-center space-x-1 px-3 py-1 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                      title="Fill response column with test data"
                    >
                      <Target className="w-4 h-4" />
                      <span>Fill Test Data</span>
                    </button>
                    <button
                      onClick={addRow}
                      className="flex items-center space-x-1 px-3 py-1 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors text-sm"
                    >
                      <Plus className="w-4 h-4" />
                      <span>Add Row</span>
                    </button>
                  </div>
                </div>

                <div className="overflow-x-auto bg-slate-700/30 rounded-lg border-2 border-slate-600">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-slate-700/70">
                        <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 w-14">
                          #
                        </th>
                        {factorNames.map((factor, idx) => (
                          <th
                            key={idx}
                            className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[80px]"
                          >
                            {factor}
                          </th>
                        ))}
                        <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[100px] bg-orange-900/20">
                          {responseName}
                        </th>
                        <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-slate-600 w-16"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {tableData.map((row, rowIndex) => (
                        <tr key={rowIndex} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                          <td className="px-3 py-2 text-center text-gray-300 text-sm font-medium bg-slate-700/30 border-r border-slate-600">
                            {rowIndex + 1}
                          </td>
                          {row.map((cell, colIndex) => (
                            <td key={colIndex} className="px-1 py-1 border-r border-slate-700/20">
                              <input
                                type="text"
                                value={cell}
                                onChange={(e) => handleCellChange(rowIndex, colIndex, e.target.value)}
                                disabled={colIndex < factorNames.length && designData}
                                className="w-full px-2 py-1.5 bg-slate-800/50 text-gray-100 border border-slate-600/50 focus:border-orange-500 focus:bg-slate-700/50 hover:border-slate-500 rounded-sm focus:outline-none focus:ring-2 focus:ring-orange-500/50 text-sm transition-all disabled:opacity-60"
                                placeholder={colIndex === row.length - 1 ? '0.0' : ''}
                              />
                            </td>
                          ))}
                          <td className="px-2 py-2 text-center">
                            <button
                              onClick={() => removeRow(rowIndex)}
                              disabled={tableData.length === 1}
                              className="p-1 text-red-400 hover:text-red-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <p className="text-gray-400 text-xs mt-2">
                  Enter response values for each run. Factor levels from the design cannot be modified.
                </p>

                <button
                  onClick={handleFitModel}
                  disabled={loading}
                  className="w-full mt-4 bg-green-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Fitting Model...' : 'Fit Response Surface Model'}
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'model' && modelResult && (
        <div className="space-y-6">
          {/* Model Summary */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-4">Second-Order Model</h3>
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-700/30">
                <p className="text-blue-200 text-sm">R-Squared</p>
                <p className="text-2xl font-bold text-blue-100">{modelResult.r_squared}</p>
              </div>
              <div className="bg-green-900/20 rounded-lg p-4 border border-green-700/30">
                <p className="text-green-200 text-sm">Adj R-Squared</p>
                <p className="text-2xl font-bold text-green-100">{modelResult.adj_r_squared}</p>
              </div>
              <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-700/30">
                <p className="text-purple-200 text-sm">RMSE</p>
                <p className="text-2xl font-bold text-purple-100">{modelResult.rmse}</p>
              </div>
            </div>

            {/* Coefficients Table */}
            <h4 className="text-gray-100 font-semibold mb-3">Model Coefficients</h4>
            <div className="overflow-x-auto bg-slate-700/30 rounded-lg">
              <table className="w-full">
                <thead>
                  <tr className="bg-slate-700/70">
                    <th className="px-4 py-2 text-left text-gray-100 font-semibold border-b border-slate-600">Term</th>
                    <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">Estimate</th>
                    <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">Std Error</th>
                    <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">t-value</th>
                    <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">p-value</th>
                    <th className="px-4 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">Sig.</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(modelResult.coefficients).map(([term, values], idx) => (
                    <tr key={idx} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                      <td className="px-4 py-2 text-gray-100 font-mono text-sm">{term}</td>
                      <td className="px-4 py-2 text-right text-gray-100">{values.estimate}</td>
                      <td className="px-4 py-2 text-right text-gray-300">{values.std_error}</td>
                      <td className="px-4 py-2 text-right text-gray-100">{values.t_value}</td>
                      <td className="px-4 py-2 text-right text-gray-100">{values.p_value}</td>
                      <td className="px-4 py-2 text-center">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          values.p_value < alpha ? 'bg-green-900/50 text-green-200' : 'bg-slate-700 text-gray-400'
                        }`}>
                          {values.p_value < alpha ? '***' : 'ns'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Canonical Analysis */}
          {canonicalResult && !canonicalResult.error && (
            <div className="bg-gradient-to-r from-purple-900/30 to-indigo-900/30 backdrop-blur-lg rounded-2xl p-6 border border-purple-700/50">
              <h3 className="text-2xl font-bold text-gray-100 mb-4">Canonical Analysis</h3>

              <div className="mb-6">
                <div className={`inline-block px-4 py-2 rounded-lg font-bold text-lg ${
                  canonicalResult.surface_type === 'Maximum' ? 'bg-green-900/50 text-green-200'
                  : canonicalResult.surface_type === 'Minimum' ? 'bg-blue-900/50 text-blue-200'
                  : 'bg-orange-900/50 text-orange-200'
                }`}>
                  Surface Type: {canonicalResult.surface_type}
                </div>
              </div>

              <h4 className="text-gray-100 font-semibold mb-2">Stationary Point:</h4>
              <div className="grid grid-cols-3 gap-3 mb-6">
                {Object.entries(canonicalResult.stationary_point).map(([factor, value]) => (
                  <div key={factor} className="bg-slate-700/50 rounded-lg p-3">
                    <p className="text-gray-400 text-sm">{factor}</p>
                    <p className="text-xl font-bold text-gray-100">{value}</p>
                  </div>
                ))}
              </div>

              <h4 className="text-gray-100 font-semibold mb-2">Eigenvalues:</h4>
              <div className="flex flex-wrap gap-2">
                {canonicalResult.eigenvalues.map((eigenvalue, idx) => (
                  <div key={idx} className={`px-3 py-1 rounded-lg ${
                    eigenvalue < 0 ? 'bg-red-900/50 text-red-200' : 'bg-green-900/50 text-green-200'
                  }`}>
                    λ{idx + 1} = {eigenvalue}
                  </div>
                ))}
              </div>

              <p className="text-gray-300 text-sm mt-4">
                <strong>Interpretation:</strong> {canonicalResult.interpretation}
              </p>
            </div>
          )}

          {/* Curvature Test */}
          {modelResult.curvature_test && (
            <div className="bg-slate-800/50 rounded-lg p-5 border border-slate-700/50">
              <h4 className="text-gray-100 font-semibold mb-3">Curvature Detection</h4>
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-slate-700/50 rounded-lg p-3">
                  <p className="text-gray-400 text-sm">F-Statistic</p>
                  <p className="text-xl font-bold text-gray-100">{modelResult.curvature_test.f_statistic}</p>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-3">
                  <p className="text-gray-400 text-sm">p-value</p>
                  <p className="text-xl font-bold text-gray-100">{modelResult.curvature_test.p_value}</p>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-3">
                  <p className="text-gray-400 text-sm">Significant Curvature?</p>
                  <p className={`text-xl font-bold ${modelResult.curvature_test.significant_curvature ? 'text-green-400' : 'text-gray-400'}`}>
                    {modelResult.curvature_test.significant_curvature ? 'Yes' : 'No'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Enhanced ANOVA Table */}
          {modelResult.enhanced_anova && (
            <EnhancedANOVA
              enhancedAnova={modelResult.enhanced_anova}
              lackOfFitTest={modelResult.lack_of_fit_test}
              alpha={alpha}
            />
          )}

          {/* Residual Analysis */}
          {modelResult.diagnostics && (
            <ResidualAnalysis
              diagnostics={modelResult.diagnostics}
              responseName={responseName}
            />
          )}
        </div>
      )}

      {activeTab === 'optimize' && modelResult && (
        <div className="space-y-6">
          {/* Optimization Controls */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-6">Response Optimization</h3>

            <div className="grid grid-cols-2 gap-4 mb-6">
              <button
                onClick={() => handleOptimize('maximize')}
                disabled={loading}
                className="flex items-center justify-center space-x-2 bg-green-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
              >
                <Target className="w-5 h-5" />
                <span>Maximize Response</span>
              </button>
              <button
                onClick={() => handleOptimize('minimize')}
                disabled={loading}
                className="flex items-center justify-center space-x-2 bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                <Target className="w-5 h-5" />
                <span>Minimize Response</span>
              </button>
            </div>

            <button
              onClick={handleSteepestAscent}
              disabled={loading}
              className="w-full flex items-center justify-center space-x-2 bg-purple-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50"
            >
              <TrendingUp className="w-5 h-5" />
              <span>Calculate Steepest Ascent Path</span>
            </button>
          </div>

          {/* Optimization Results */}
          {optimizationResult && (
            <div className="bg-gradient-to-r from-green-900/30 to-blue-900/30 backdrop-blur-lg rounded-2xl p-6 border border-green-700/50">
              <h3 className="text-2xl font-bold text-gray-100 mb-4">
                Optimization Results ({optimizationResult.target})
              </h3>

              <div className="mb-6">
                <div className="bg-slate-700/50 rounded-lg p-4 inline-block">
                  <p className="text-gray-400 text-sm">Predicted Response</p>
                  <p className="text-3xl font-bold text-green-300">{optimizationResult.predicted_response}</p>
                </div>
              </div>

              <h4 className="text-gray-100 font-semibold mb-3">Optimal Factor Settings:</h4>
              <div className="grid grid-cols-3 gap-4">
                {Object.entries(optimizationResult.optimal_point).map(([factor, value]) => (
                  <div key={factor} className="bg-slate-700/50 rounded-lg p-4">
                    <p className="text-gray-400 text-sm">{factor}</p>
                    <p className="text-2xl font-bold text-gray-100">{value}</p>
                  </div>
                ))}
              </div>

              <p className="text-gray-300 text-sm mt-4">
                <strong>Method:</strong> {optimizationResult.method}
              </p>
            </div>
          )}

          {/* Steepest Ascent Results */}
          {steepestAscentResult && (
            <div className="bg-gradient-to-r from-purple-900/30 to-pink-900/30 backdrop-blur-lg rounded-2xl p-6 border border-purple-700/50">
              <h3 className="text-2xl font-bold text-gray-100 mb-4">Steepest Ascent Path</h3>

              <div className="mb-4">
                <p className="text-gray-300">
                  <strong>Gradient Magnitude:</strong> {steepestAscentResult.gradient_magnitude}
                </p>
              </div>

              <h4 className="text-gray-100 font-semibold mb-3">Path of Steepest Ascent:</h4>
              <div className="overflow-x-auto bg-slate-700/30 rounded-lg">
                <table className="w-full">
                  <thead>
                    <tr className="bg-slate-700/70">
                      <th className="px-4 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">Step</th>
                      {factorNames.map((factor, idx) => (
                        <th key={idx} className="px-4 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">
                          {factor}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {steepestAscentResult.path.map((point, idx) => (
                      <tr key={idx} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                        <td className="px-4 py-2 text-center text-gray-100 font-bold">{point.step}</td>
                        {factorNames.map((factor, fIdx) => (
                          <td key={fIdx} className="px-4 py-2 text-center text-gray-100">
                            {point[factor]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <p className="text-gray-300 text-sm mt-4">
                <strong>Interpretation:</strong> Follow this path from the current design center to move in the direction of maximum increase in the response. Conduct experiments at these points to verify improvement.
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'visualize' && modelResult && (
        <div className="space-y-6">
          {numFactors === 2 && surfaceData ? (
            // Simple 2-factor visualization
            <>
              <ResponseSurface3D
                surfaceData={surfaceData}
                factor1={factorNames[0]}
                factor2={factorNames[1]}
                responseName={responseName}
              />
              <ContourPlot
                surfaceData={surfaceData}
                factor1={factorNames[0]}
                factor2={factorNames[1]}
                responseName={responseName}
                experimentalData={(() => {
                  // Convert tableData to experimental data format
                  const validRows = tableData.filter(row => {
                    const responseValue = row[row.length - 1]
                    return responseValue !== '' && responseValue !== null && responseValue !== undefined
                  })
                  return validRows.map(row => {
                    const point = {}
                    factorNames.forEach((factor, i) => {
                      point[factor] = parseFloat(row[i])
                    })
                    point[responseName] = parseFloat(row[row.length - 1])
                    return point
                  })
                })()}
                optimizationResult={optimizationResult}
                canonicalResult={canonicalResult}
                steepestAscentResult={steepestAscentResult}
              />
            </>
          ) : (
            // Sliced visualization for any number of factors
            <SlicedVisualization
              factorNames={factorNames}
              responseName={responseName}
              coefficients={modelResult.coefficients}
              experimentalData={(() => {
                // Convert tableData to experimental data format
                const validRows = tableData.filter(row => {
                  const responseValue = row[row.length - 1]
                  return responseValue !== '' && responseValue !== null && responseValue !== undefined
                })
                return validRows.map(row => {
                  const point = {}
                  factorNames.forEach((factor, i) => {
                    point[factor] = parseFloat(row[i])
                  })
                  point[responseName] = parseFloat(row[row.length - 1])
                  return point
                })
              })()}
              optimizationResult={optimizationResult}
              canonicalResult={canonicalResult}
            />
          )}
        </div>
      )}
    </div>
  )
}

export default RSM
