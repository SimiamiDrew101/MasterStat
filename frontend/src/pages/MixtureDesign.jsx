import { useState, useEffect } from 'react'
import axios from 'axios'
import { Droplet, Plus, Trash2, Target } from 'lucide-react'
import ResultCard from '../components/ResultCard'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const MixtureDesign = () => {
  // Design configuration
  const [designType, setDesignType] = useState('simplex-lattice')
  const [numComponents, setNumComponents] = useState(3)
  const [latticeDegree, setLatticeDegree] = useState(2)
  const [componentNames, setComponentNames] = useState(['X1', 'X2', 'X3'])
  const [responseName, setResponseName] = useState('Y')

  // Data and results
  const [designData, setDesignData] = useState(null)
  const [tableData, setTableData] = useState([])
  const [modelResult, setModelResult] = useState(null)
  const [modelDegree, setModelDegree] = useState(2)
  const [optimizationResult, setOptimizationResult] = useState(null)
  const [componentBounds, setComponentBounds] = useState({})

  // UI state
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('design')

  // Update component names when number changes
  useEffect(() => {
    const newNames = Array.from({ length: numComponents }, (_, i) => `X${i + 1}`)
    setComponentNames(newNames)
  }, [numComponents])

  // Generate mixture design
  const handleGenerateDesign = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post(`${API_URL}/api/rsm/mixture-design/generate`, {
        n_components: numComponents,
        design_type: designType,
        degree: designType === 'simplex-lattice' ? latticeDegree : undefined,
        component_names: componentNames
      })

      setDesignData(response.data)

      // Convert to table format with empty response column
      const table = response.data.design_matrix.map(row => {
        const tableRow = componentNames.map(comp => row[comp] || 0)
        tableRow.push('') // Empty response
        return tableRow
      })

      setTableData(table)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate design')
    } finally {
      setLoading(false)
    }
  }

  // Fit mixture model
  const handleFitModel = async () => {
    setLoading(true)
    setError(null)

    try {
      // Validate data
      const validRows = tableData.filter(row => {
        const responseValue = row[row.length - 1]
        return responseValue !== '' && responseValue !== null && responseValue !== undefined
      })

      if (validRows.length < 5) {
        throw new Error(`Need at least 5 complete data points (found ${validRows.length})`)
      }

      // Convert to API format
      const data = validRows.map(row => {
        const point = {}
        componentNames.forEach((comp, i) => {
          point[comp] = parseFloat(row[i])
        })
        point[responseName] = parseFloat(row[row.length - 1])
        return point
      })

      const response = await axios.post(`${API_URL}/api/rsm/mixture-design/fit-model`, {
        data: data,
        components: componentNames,
        response: responseName,
        model_type: 'scheffe',
        degree: modelDegree
      })

      setModelResult(response.data)
      setActiveTab('model')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to fit model')
    } finally {
      setLoading(false)
    }
  }

  // Optimize mixture
  const handleOptimize = async (target) => {
    setLoading(true)
    setError(null)

    try {
      if (!modelResult || !modelResult.coefficients) {
        throw new Error('Fit a model first')
      }

      // Build coefficients from model
      const coefficients = {}
      Object.entries(modelResult.coefficients).forEach(([term, values]) => {
        coefficients[term] = values.estimate
      })

      const response = await axios.post(`${API_URL}/api/rsm/mixture-design/optimize`, {
        coefficients: coefficients,
        components: componentNames,
        target: target,
        lower_bounds: componentBounds.lower || null,
        upper_bounds: componentBounds.upper || null
      })

      setOptimizationResult(response.data)
      setActiveTab('optimize')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Optimization failed')
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
    const newRow = Array(componentNames.length + 1).fill('')
    setTableData([...tableData, newRow])
  }

  const removeRow = (rowIndex) => {
    if (tableData.length > 1) {
      setTableData(tableData.filter((_, idx) => idx !== rowIndex))
    }
  }

  // Fill test data
  const fillTestData = () => {
    const newData = tableData.map(row => {
      const newRow = [...row]
      const components = row.slice(0, -1)
      // Generate response based on a simple mixture model
      let response = 0
      components.forEach((comp, i) => {
        response += (10 + i * 5) * parseFloat(comp)
      })
      // Add some interaction effect
      if (components.length >= 2) {
        response += 15 * parseFloat(components[0]) * parseFloat(components[1])
      }
      // Add noise
      response += (Math.random() - 0.5) * 2
      newRow[newRow.length - 1] = response.toFixed(2)
      return newRow
    })
    setTableData(newData)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-cyan-900/30 to-blue-900/30 backdrop-blur-lg rounded-2xl p-8 border border-cyan-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <Droplet className="w-10 h-10 text-cyan-400" />
          <h2 className="text-4xl font-bold text-gray-100">Mixture Designs</h2>
        </div>
        <p className="text-gray-300 text-lg">
          Optimize formulations where components must sum to 100%. Design experiments for blends, recipes, and mixtures using Simplex designs and Scheffé models.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-2 bg-slate-800/50 p-2 rounded-lg">
        <button
          onClick={() => setActiveTab('design')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'design'
              ? 'bg-cyan-600 text-white'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
        >
          1. Design
        </button>
        <button
          onClick={() => setActiveTab('model')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'model'
              ? 'bg-cyan-600 text-white'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
          disabled={!modelResult}
        >
          2. Model Analysis
        </button>
        <button
          onClick={() => setActiveTab('optimize')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'optimize'
              ? 'bg-cyan-600 text-white'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
          disabled={!modelResult}
        >
          3. Optimization
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200 font-medium">Error: {error}</p>
        </div>
      )}

      {/* Design Tab */}
      {activeTab === 'design' && (
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
          <h3 className="text-2xl font-bold text-gray-100 mb-6">Mixture Design Configuration</h3>

          <div className="space-y-6">
            {/* Design Type */}
            <div>
              <label className="block text-gray-100 font-medium mb-2">Design Type</label>
              <select
                value={designType}
                onChange={(e) => setDesignType(e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              >
                <option value="simplex-lattice">Simplex-Lattice</option>
                <option value="simplex-centroid">Simplex-Centroid</option>
              </select>
              <p className="text-gray-400 text-xs mt-1">
                {designType === 'simplex-lattice'
                  ? 'Evenly spaced proportions across the simplex'
                  : 'Vertices, edge centers, face centers, and overall centroid'}
              </p>
            </div>

            {/* Degree (for lattice) */}
            {designType === 'simplex-lattice' && (
              <div>
                <label className="block text-gray-100 font-medium mb-2">Lattice Degree</label>
                <select
                  value={latticeDegree}
                  onChange={(e) => setLatticeDegree(parseInt(e.target.value))}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                >
                  <option value={2}>2 (0, 0.5, 1)</option>
                  <option value={3}>3 (0, 0.33, 0.67, 1)</option>
                  <option value={4}>4 (0, 0.25, 0.5, 0.75, 1)</option>
                </select>
                <p className="text-gray-400 text-xs mt-1">
                  Higher degrees provide more design points for better model fit
                </p>
              </div>
            )}

            {/* Number of Components */}
            <div>
              <label className="block text-gray-100 font-medium mb-2">Number of Components</label>
              <input
                type="number"
                min={2}
                max={6}
                value={numComponents}
                onChange={(e) => setNumComponents(parseInt(e.target.value) || 2)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              />
              <p className="text-gray-400 text-xs mt-1">
                Number of ingredients/components in your mixture (2-6)
              </p>
            </div>

            {/* Component Names */}
            <div>
              <label className="block text-gray-100 font-medium mb-2">Component Names</label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {componentNames.map((name, idx) => (
                  <input
                    key={idx}
                    type="text"
                    value={name}
                    onChange={(e) => {
                      const newNames = [...componentNames]
                      newNames[idx] = e.target.value
                      setComponentNames(newNames)
                    }}
                    className="px-3 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                    placeholder={`Component ${idx + 1}`}
                  />
                ))}
              </div>
            </div>

            {/* Response Name */}
            <div>
              <label className="block text-gray-100 font-medium mb-2">Response Variable Name</label>
              <input
                type="text"
                value={responseName}
                onChange={(e) => setResponseName(e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              />
            </div>

            {/* Generate Button */}
            <button
              onClick={handleGenerateDesign}
              disabled={loading}
              className="w-full bg-cyan-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-cyan-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Generating...' : 'Generate Mixture Design'}
            </button>

            {/* Design Properties */}
            {designData && (
              <div className="bg-gradient-to-r from-cyan-900/20 to-blue-900/20 rounded-lg p-5 border border-cyan-700/30">
                <h4 className="text-cyan-200 font-semibold mb-3">Design Properties</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Design Type:</span>
                    <span className="text-gray-100 ml-2 font-medium">{designData.design_type}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Total Runs:</span>
                    <span className="text-gray-100 ml-2 font-medium">{designData.n_runs}</span>
                  </div>
                  <div className="col-span-2">
                    <span className="text-gray-400">Constraint:</span>
                    <span className="text-gray-100 ml-2">{designData.properties.constraint}</span>
                  </div>
                  <div className="col-span-2">
                    <span className="text-gray-400">Description:</span>
                    <span className="text-gray-100 ml-2">{designData.properties.description}</span>
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
                      className="flex items-center space-x-1 px-3 py-1 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm disabled:opacity-50"
                    >
                      <Target className="w-4 h-4" />
                      <span>Fill Test Data</span>
                    </button>
                    <button
                      onClick={addRow}
                      className="flex items-center space-x-1 px-3 py-1 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors text-sm"
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
                        {componentNames.map((comp, idx) => (
                          <th
                            key={idx}
                            className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[80px]"
                          >
                            {comp}
                          </th>
                        ))}
                        <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[100px] bg-cyan-900/20">
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
                                disabled={colIndex < componentNames.length && designData}
                                className="w-full px-2 py-1.5 bg-slate-800/50 text-gray-100 border border-slate-600/50 focus:border-cyan-500 focus:bg-slate-700/50 hover:border-slate-500 rounded-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/50 text-sm transition-all disabled:opacity-60"
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
                  Component proportions from the design are fixed and sum to 1.0. Enter response values for each mixture.
                </p>

                {/* Model Degree Selection */}
                <div className="mt-4">
                  <label className="block text-gray-100 font-medium mb-2">Scheffé Model Degree</label>
                  <select
                    value={modelDegree}
                    onChange={(e) => setModelDegree(parseInt(e.target.value))}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  >
                    <option value={1}>Linear (Component Effects Only)</option>
                    <option value={2}>Quadratic (+ Pairwise Interactions)</option>
                    <option value={3}>Cubic (+ Three-Way Interactions)</option>
                  </select>
                </div>

                <button
                  onClick={handleFitModel}
                  disabled={loading}
                  className="w-full mt-4 bg-green-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Fitting Model...' : 'Fit Mixture Model'}
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Model Tab */}
      {activeTab === 'model' && modelResult && (
        <div className="space-y-6">
          {/* Model Summary */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-4">{modelResult.model_type}</h3>
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

            {/* Formula */}
            <div className="bg-slate-700/30 rounded-lg p-4 mb-4">
              <p className="text-gray-400 text-sm mb-2">Model Formula:</p>
              <p className="text-gray-100 font-mono text-sm">{modelResult.formula}</p>
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
                          values.p_value < 0.05 ? 'bg-green-900/50 text-green-200' : 'bg-slate-700 text-gray-400'
                        }`}>
                          {values.p_value < 0.05 ? '***' : 'ns'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* ANOVA Table */}
          {modelResult.anova && (
            <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700/50">
              <h4 className="text-gray-100 font-semibold mb-3">ANOVA Table</h4>
              <div className="overflow-x-auto bg-slate-700/30 rounded-lg">
                <table className="w-full">
                  <thead>
                    <tr className="bg-slate-700/70">
                      <th className="px-4 py-2 text-left text-gray-100 font-semibold border-b border-slate-600">Source</th>
                      <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">DF</th>
                      <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">Sum Sq</th>
                      <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">F</th>
                      <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">p-value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(modelResult.anova).map(([source, values], idx) => (
                      <tr key={idx} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                        <td className="px-4 py-2 text-gray-100 font-medium">{source}</td>
                        <td className="px-4 py-2 text-right text-gray-100">{values.df}</td>
                        <td className="px-4 py-2 text-right text-gray-100">{values.sum_sq}</td>
                        <td className="px-4 py-2 text-right text-gray-100">{values.F || '-'}</td>
                        <td className="px-4 py-2 text-right text-gray-100">{values['PR(>F)'] || '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Optimize Tab */}
      {activeTab === 'optimize' && modelResult && (
        <div className="space-y-6">
          {/* Optimization Controls */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-6">Mixture Optimization</h3>

            {/* Component Bounds */}
            <div className="mb-6">
              <h4 className="text-gray-100 font-semibold mb-3">Component Constraints (Optional)</h4>
              <p className="text-gray-400 text-sm mb-4">
                Set minimum and maximum proportions for each component. Components will still sum to 1.0.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {componentNames.map((comp, idx) => (
                  <div key={idx} className="bg-slate-700/30 rounded-lg p-3">
                    <p className="text-gray-300 font-medium mb-2">{comp}</p>
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="1"
                        placeholder="Min"
                        value={componentBounds.lower?.[comp] || ''}
                        onChange={(e) => {
                          const newBounds = {...componentBounds}
                          if (!newBounds.lower) newBounds.lower = {}
                          newBounds.lower[comp] = e.target.value ? parseFloat(e.target.value) : undefined
                          setComponentBounds(newBounds)
                        }}
                        className="w-full px-3 py-1.5 bg-slate-800/50 text-gray-100 rounded border border-slate-600 focus:border-cyan-500 focus:outline-none text-sm"
                      />
                      <span className="text-gray-400">≤</span>
                      <span className="text-gray-100 font-medium">{comp}</span>
                      <span className="text-gray-400">≤</span>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="1"
                        placeholder="Max"
                        value={componentBounds.upper?.[comp] || ''}
                        onChange={(e) => {
                          const newBounds = {...componentBounds}
                          if (!newBounds.upper) newBounds.upper = {}
                          newBounds.upper[comp] = e.target.value ? parseFloat(e.target.value) : undefined
                          setComponentBounds(newBounds)
                        }}
                        className="w-full px-3 py-1.5 bg-slate-800/50 text-gray-100 rounded border border-slate-600 focus:border-cyan-500 focus:outline-none text-sm"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Optimization Buttons */}
            <div className="grid grid-cols-2 gap-4">
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

              <h4 className="text-gray-100 font-semibold mb-3">Optimal Mixture Proportions:</h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
                {Object.entries(optimizationResult.optimal_point).map(([comp, value]) => (
                  <div key={comp} className="bg-slate-700/50 rounded-lg p-4">
                    <p className="text-gray-400 text-sm">{comp}</p>
                    <p className="text-2xl font-bold text-gray-100">{value}</p>
                    <p className="text-gray-400 text-xs">{(value * 100).toFixed(2)}%</p>
                  </div>
                ))}
              </div>

              <div className="bg-slate-700/30 rounded-lg p-4">
                <p className="text-gray-300 text-sm">
                  <strong>Method:</strong> {optimizationResult.method}
                </p>
                <p className="text-gray-300 text-sm mt-2">
                  <strong>Verification:</strong> Components sum to {optimizationResult.verification.components_sum}
                  {optimizationResult.verification.meets_constraint &&
                    <span className="text-green-400 ml-2">✓ Mixture constraint satisfied</span>
                  }
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default MixtureDesign
