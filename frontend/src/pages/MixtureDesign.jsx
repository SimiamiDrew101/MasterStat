import { useState, useEffect } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'
import { Droplet, Plus, Trash2, Target, Triangle, TrendingUp, Settings, Layers, ChevronDown, ChevronUp, Info } from 'lucide-react'
import ResultCard from '../components/ResultCard'

const API_URL = import.meta.env.VITE_API_URL || ''

const MixtureDesign = () => {
  // Design configuration
  const [designType, setDesignType] = useState('simplex-lattice')
  const [numComponents, setNumComponents] = useState(3)
  const [latticeDegree, setLatticeDegree] = useState(2)
  const [componentNames, setComponentNames] = useState(['X1', 'X2', 'X3'])
  const [responseName, setResponseName] = useState('Y')

  // Extreme Vertices configuration
  const [useConstraints, setUseConstraints] = useState(false)
  const [componentConstraints, setComponentConstraints] = useState([])
  const [includeEVCentroids, setIncludeEVCentroids] = useState(true)
  const [includeEVAxial, setIncludeEVAxial] = useState(false)
  const [evCenterPoints, setEvCenterPoints] = useState(1)

  // Mixture + Process configuration
  const [processFactors, setProcessFactors] = useState([
    { name: 'Temperature', levels: [-1, 1] }
  ])
  const [processDesignType, setProcessDesignType] = useState('full-factorial')

  // Data and results
  const [designData, setDesignData] = useState(null)
  const [tableData, setTableData] = useState([])
  const [modelResult, setModelResult] = useState(null)
  const [modelDegree, setModelDegree] = useState(2)
  const [optimizationResult, setOptimizationResult] = useState(null)
  const [componentBounds, setComponentBounds] = useState({})

  // Visualization results
  const [ternaryContour, setTernaryContour] = useState(null)
  const [tracePlotData, setTracePlotData] = useState(null)
  const [referenceBlend, setReferenceBlend] = useState(null)

  // UI state
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('design')
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Update component names and constraints when number changes
  useEffect(() => {
    const newNames = Array.from({ length: numComponents }, (_, i) =>
      componentNames[i] || `X${i + 1}`
    )
    setComponentNames(newNames)

    // Update constraints
    const newConstraints = Array.from({ length: numComponents }, (_, i) => ({
      name: newNames[i],
      min_prop: componentConstraints[i]?.min_prop ?? 0,
      max_prop: componentConstraints[i]?.max_prop ?? 1
    }))
    setComponentConstraints(newConstraints)
  }, [numComponents])

  // Update constraint names when component names change
  useEffect(() => {
    if (componentConstraints.length === componentNames.length) {
      const newConstraints = componentConstraints.map((c, i) => ({
        ...c,
        name: componentNames[i]
      }))
      setComponentConstraints(newConstraints)
    }
  }, [componentNames])

  // Generate standard mixture design
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

  // Generate Extreme Vertices design
  const handleGenerateExtremeVertices = async () => {
    setLoading(true)
    setError(null)

    try {
      // Validate constraints
      const minSum = componentConstraints.reduce((sum, c) => sum + (c.min_prop || 0), 0)
      const maxSum = componentConstraints.reduce((sum, c) => sum + (c.max_prop || 1), 0)

      if (minSum > 1.01) {
        throw new Error(`Minimum proportions sum to ${minSum.toFixed(3)}, exceeding 1.0`)
      }
      if (maxSum < 0.99) {
        throw new Error(`Maximum proportions sum to ${maxSum.toFixed(3)}, less than 1.0`)
      }

      const response = await axios.post(`${API_URL}/api/mixture/extreme-vertices/generate`, {
        components: componentConstraints.map(c => ({
          name: c.name,
          min_prop: c.min_prop || 0,
          max_prop: c.max_prop || 1
        })),
        include_centroids: includeEVCentroids,
        include_axial: includeEVAxial,
        n_center_points: evCenterPoints
      })

      setDesignData({
        ...response.data,
        design_type: 'Extreme Vertices',
        design_matrix: response.data.design,
        properties: {
          constraint: 'Sum = 1.0 with component bounds',
          description: response.data.interpretation
        }
      })

      // Convert to table format
      const table = response.data.design.map(row => {
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

  // Generate Mixture + Process design
  const handleGenerateMixtureProcess = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post(`${API_URL}/api/mixture/mixture-process/generate`, {
        n_components: numComponents,
        component_names: componentNames,
        mixture_design_type: designType,
        lattice_degree: latticeDegree,
        process_factors: processFactors,
        process_design_type: processDesignType
      })

      const allColumns = [...componentNames, ...response.data.process_factor_names]

      setDesignData({
        ...response.data,
        design_type: 'Mixture + Process',
        design_matrix: response.data.design,
        properties: {
          constraint: 'Mixture sum = 1.0 with process factors',
          description: response.data.interpretation
        },
        all_columns: allColumns
      })

      // Convert to table format
      const table = response.data.design.map(row => {
        const tableRow = allColumns.map(col => row[col] ?? 0)
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

      // Set reference blend to centroid for trace plots
      const centroid = componentNames.map(() => 1 / componentNames.length)
      setReferenceBlend(centroid)

      setActiveTab('model')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to fit model')
    } finally {
      setLoading(false)
    }
  }

  // Generate ternary contour
  const handleGenerateTernaryContour = async () => {
    if (!modelResult || componentNames.length !== 3) return

    setLoading(true)
    setError(null)

    try {
      // Build coefficients from model
      const coefficients = {}
      Object.entries(modelResult.coefficients).forEach(([term, values]) => {
        coefficients[term] = values.estimate
      })

      const response = await axios.post(`${API_URL}/api/mixture/ternary-contour`, {
        component_names: componentNames,
        model_coefficients: coefficients,
        model_type: modelDegree === 1 ? 'linear' : modelDegree === 2 ? 'quadratic' : 'cubic',
        grid_resolution: 50,
        constraints: useConstraints ? componentConstraints : null
      })

      setTernaryContour(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate contour')
    } finally {
      setLoading(false)
    }
  }

  // Generate trace plot
  const handleGenerateTracePlot = async () => {
    if (!modelResult || !referenceBlend) return

    setLoading(true)
    setError(null)

    try {
      // Build coefficients from model
      const coefficients = {}
      Object.entries(modelResult.coefficients).forEach(([term, values]) => {
        coefficients[term] = values.estimate
      })

      const response = await axios.post(`${API_URL}/api/mixture/trace-plot`, {
        component_names: componentNames,
        reference_blend: referenceBlend,
        model_coefficients: coefficients,
        model_type: modelDegree === 1 ? 'linear' : modelDegree === 2 ? 'quadratic' : 'cubic',
        n_points: 50
      })

      setTracePlotData(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate trace plot')
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
    const numCols = designData?.all_columns?.length || componentNames.length
    const newRow = Array(numCols + 1).fill('')
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
      const components = row.slice(0, componentNames.length)
      // Generate response based on a simple mixture model
      let response = 0
      components.forEach((comp, i) => {
        response += (10 + i * 5) * parseFloat(comp || 0)
      })
      // Add some interaction effect
      if (components.length >= 2) {
        response += 15 * parseFloat(components[0] || 0) * parseFloat(components[1] || 0)
      }
      // Add noise
      response += (Math.random() - 0.5) * 2
      newRow[newRow.length - 1] = response.toFixed(2)
      return newRow
    })
    setTableData(newData)
  }

  // Add process factor
  const addProcessFactor = () => {
    setProcessFactors([...processFactors, { name: `P${processFactors.length + 1}`, levels: [-1, 1] }])
  }

  // Remove process factor
  const removeProcessFactor = (idx) => {
    if (processFactors.length > 1) {
      setProcessFactors(processFactors.filter((_, i) => i !== idx))
    }
  }

  // Render ternary plot
  const renderTernaryPlot = () => {
    if (!ternaryContour || componentNames.length !== 3) return null

    const { contour_data, optimal_blend } = ternaryContour

    return (
      <Plot
        data={[
          {
            type: 'scatterternary',
            mode: 'markers',
            a: contour_data.proportions.map(p => p[0]),
            b: contour_data.proportions.map(p => p[1]),
            c: contour_data.proportions.map(p => p[2]),
            marker: {
              size: 6,
              color: contour_data.z,
              colorscale: 'Viridis',
              showscale: true,
              colorbar: {
                title: { text: responseName, font: { color: '#e2e8f0' } },
                tickfont: { color: '#e2e8f0' }
              }
            },
            text: contour_data.z.map((z, i) =>
              `${componentNames[0]}: ${(contour_data.proportions[i][0] * 100).toFixed(1)}%<br>` +
              `${componentNames[1]}: ${(contour_data.proportions[i][1] * 100).toFixed(1)}%<br>` +
              `${componentNames[2]}: ${(contour_data.proportions[i][2] * 100).toFixed(1)}%<br>` +
              `${responseName}: ${z.toFixed(3)}`
            ),
            hoverinfo: 'text'
          },
          // Optimal point
          optimal_blend && {
            type: 'scatterternary',
            mode: 'markers',
            a: [optimal_blend[componentNames[0]]],
            b: [optimal_blend[componentNames[1]]],
            c: [optimal_blend[componentNames[2]]],
            marker: {
              size: 15,
              color: '#f59e0b',
              symbol: 'star',
              line: { color: '#1e293b', width: 2 }
            },
            name: 'Optimal',
            text: [`Optimal: ${optimal_blend.predicted_response?.toFixed(3)}`],
            hoverinfo: 'text'
          }
        ].filter(Boolean)}
        layout={{
          ternary: {
            sum: 1,
            aaxis: {
              title: { text: componentNames[0], font: { color: '#e2e8f0' } },
              tickfont: { color: '#94a3b8' },
              linecolor: '#475569',
              gridcolor: '#475569'
            },
            baxis: {
              title: { text: componentNames[1], font: { color: '#e2e8f0' } },
              tickfont: { color: '#94a3b8' },
              linecolor: '#475569',
              gridcolor: '#475569'
            },
            caxis: {
              title: { text: componentNames[2], font: { color: '#e2e8f0' } },
              tickfont: { color: '#94a3b8' },
              linecolor: '#475569',
              gridcolor: '#475569'
            },
            bgcolor: '#0f172a'
          },
          paper_bgcolor: '#1e293b',
          font: { color: '#e2e8f0' },
          showlegend: true,
          legend: { font: { color: '#e2e8f0' } },
          margin: { t: 50, b: 50, l: 50, r: 50 }
        }}
        config={{ responsive: true }}
        style={{ width: '100%', height: '500px' }}
      />
    )
  }

  // Render trace plot
  const renderTracePlot = () => {
    if (!tracePlotData) return null

    const colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899']

    return (
      <Plot
        data={Object.entries(tracePlotData.traces).map(([name, trace], i) => ({
          type: 'scatter',
          mode: 'lines',
          x: trace.x.map(v => v * 100),
          y: trace.y,
          name: name,
          line: { color: colors[i % colors.length], width: 2 },
          hovertemplate: `${name}: %{x:.1f}%<br>${responseName}: %{y:.3f}<extra></extra>`
        }))}
        layout={{
          title: {
            text: 'Component Trace Plot (Cox Direction)',
            font: { color: '#e2e8f0' }
          },
          xaxis: {
            title: { text: 'Component Proportion (%)', font: { color: '#e2e8f0' } },
            tickfont: { color: '#94a3b8' },
            gridcolor: '#475569',
            zerolinecolor: '#475569',
            range: [0, 100]
          },
          yaxis: {
            title: { text: responseName, font: { color: '#e2e8f0' } },
            tickfont: { color: '#94a3b8' },
            gridcolor: '#475569',
            zerolinecolor: '#475569'
          },
          paper_bgcolor: '#1e293b',
          plot_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' },
          showlegend: true,
          legend: { font: { color: '#e2e8f0' } },
          shapes: [
            // Reference line
            {
              type: 'line',
              x0: 0,
              x1: 100,
              y0: tracePlotData.reference_response,
              y1: tracePlotData.reference_response,
              line: { color: '#94a3b8', dash: 'dash', width: 1 }
            }
          ],
          annotations: [
            {
              x: 95,
              y: tracePlotData.reference_response,
              text: 'Reference',
              showarrow: false,
              font: { color: '#94a3b8', size: 10 }
            }
          ]
        }}
        config={{ responsive: true }}
        style={{ width: '100%', height: '400px' }}
      />
    )
  }

  // Get table columns based on design type
  const getTableColumns = () => {
    if (designData?.all_columns) {
      return designData.all_columns
    }
    return componentNames
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
          Optimize formulations where components must sum to 100%. Design experiments for blends, recipes, and mixtures using Simplex designs, Extreme Vertices, and Scheffe models.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex flex-wrap gap-2 bg-slate-800/50 p-2 rounded-lg">
        {['design', 'model', 'visualize', 'optimize'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 min-w-[120px] px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === tab
                ? 'bg-cyan-600 text-white'
                : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
            disabled={tab !== 'design' && !tableData.length}
          >
            {tab === 'design' && '1. Design'}
            {tab === 'model' && '2. Model'}
            {tab === 'visualize' && '3. Visualize'}
            {tab === 'optimize' && '4. Optimize'}
          </button>
        ))}
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
            {/* Design Type Selection */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <button
                onClick={() => { setDesignType('simplex-lattice'); setUseConstraints(false) }}
                className={`p-4 rounded-lg border-2 transition-all ${
                  designType === 'simplex-lattice' && !useConstraints
                    ? 'border-cyan-500 bg-cyan-900/30'
                    : 'border-slate-600 bg-slate-700/30 hover:border-slate-500'
                }`}
              >
                <Triangle className="w-8 h-8 text-cyan-400 mb-2" />
                <h4 className="text-gray-100 font-semibold">Simplex-Lattice</h4>
                <p className="text-gray-400 text-sm">Evenly spaced points on simplex</p>
              </button>

              <button
                onClick={() => { setDesignType('simplex-centroid'); setUseConstraints(false) }}
                className={`p-4 rounded-lg border-2 transition-all ${
                  designType === 'simplex-centroid' && !useConstraints
                    ? 'border-cyan-500 bg-cyan-900/30'
                    : 'border-slate-600 bg-slate-700/30 hover:border-slate-500'
                }`}
              >
                <Target className="w-8 h-8 text-green-400 mb-2" />
                <h4 className="text-gray-100 font-semibold">Simplex-Centroid</h4>
                <p className="text-gray-400 text-sm">Vertices, edges, faces, centroid</p>
              </button>

              <button
                onClick={() => setUseConstraints(true)}
                className={`p-4 rounded-lg border-2 transition-all ${
                  useConstraints
                    ? 'border-cyan-500 bg-cyan-900/30'
                    : 'border-slate-600 bg-slate-700/30 hover:border-slate-500'
                }`}
              >
                <Layers className="w-8 h-8 text-purple-400 mb-2" />
                <h4 className="text-gray-100 font-semibold">Extreme Vertices</h4>
                <p className="text-gray-400 text-sm">Constrained mixture region</p>
              </button>
            </div>

            {/* Lattice Degree (for simplex-lattice) */}
            {designType === 'simplex-lattice' && !useConstraints && (
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

            {/* Extreme Vertices Constraints */}
            {useConstraints && (
              <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-700/30">
                <h4 className="text-purple-200 font-semibold mb-4">Component Constraints</h4>
                <div className="space-y-3">
                  {componentConstraints.map((constraint, idx) => (
                    <div key={idx} className="flex items-center gap-3">
                      <span className="text-gray-300 w-20">{constraint.name}</span>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="1"
                        value={constraint.min_prop}
                        onChange={(e) => {
                          const newConstraints = [...componentConstraints]
                          newConstraints[idx].min_prop = parseFloat(e.target.value) || 0
                          setComponentConstraints(newConstraints)
                        }}
                        className="w-24 px-3 py-1.5 bg-slate-800/50 text-gray-100 rounded border border-slate-600"
                        placeholder="Min"
                      />
                      <span className="text-gray-400">to</span>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="1"
                        value={constraint.max_prop}
                        onChange={(e) => {
                          const newConstraints = [...componentConstraints]
                          newConstraints[idx].max_prop = parseFloat(e.target.value) || 1
                          setComponentConstraints(newConstraints)
                        }}
                        className="w-24 px-3 py-1.5 bg-slate-800/50 text-gray-100 rounded border border-slate-600"
                        placeholder="Max"
                      />
                    </div>
                  ))}
                </div>

                <div className="mt-4 grid grid-cols-3 gap-4">
                  <label className="flex items-center gap-2 text-gray-300">
                    <input
                      type="checkbox"
                      checked={includeEVCentroids}
                      onChange={(e) => setIncludeEVCentroids(e.target.checked)}
                      className="rounded bg-slate-700 border-slate-600"
                    />
                    Include Centroids
                  </label>
                  <label className="flex items-center gap-2 text-gray-300">
                    <input
                      type="checkbox"
                      checked={includeEVAxial}
                      onChange={(e) => setIncludeEVAxial(e.target.checked)}
                      className="rounded bg-slate-700 border-slate-600"
                    />
                    Include Axial
                  </label>
                  <div className="flex items-center gap-2">
                    <span className="text-gray-300">Center Points:</span>
                    <input
                      type="number"
                      min="0"
                      max="5"
                      value={evCenterPoints}
                      onChange={(e) => setEvCenterPoints(parseInt(e.target.value) || 0)}
                      className="w-16 px-2 py-1 bg-slate-800/50 text-gray-100 rounded border border-slate-600"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Advanced: Mixture + Process */}
            <div>
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-2 text-cyan-400 hover:text-cyan-300"
              >
                {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                <span>Advanced: Mixture + Process Factors</span>
              </button>

              {showAdvanced && (
                <div className="mt-4 bg-slate-700/30 rounded-lg p-4 border border-slate-600">
                  <p className="text-gray-400 text-sm mb-4">
                    Add process factors (like temperature, time) that are crossed with the mixture design.
                  </p>

                  {processFactors.map((factor, idx) => (
                    <div key={idx} className="flex items-center gap-3 mb-3">
                      <input
                        type="text"
                        value={factor.name}
                        onChange={(e) => {
                          const newFactors = [...processFactors]
                          newFactors[idx].name = e.target.value
                          setProcessFactors(newFactors)
                        }}
                        className="w-32 px-3 py-1.5 bg-slate-800/50 text-gray-100 rounded border border-slate-600"
                        placeholder="Name"
                      />
                      <input
                        type="text"
                        value={factor.levels.join(', ')}
                        onChange={(e) => {
                          const newFactors = [...processFactors]
                          newFactors[idx].levels = e.target.value.split(',').map(v => parseFloat(v.trim()) || 0)
                          setProcessFactors(newFactors)
                        }}
                        className="flex-1 px-3 py-1.5 bg-slate-800/50 text-gray-100 rounded border border-slate-600"
                        placeholder="Levels (comma separated)"
                      />
                      <button
                        onClick={() => removeProcessFactor(idx)}
                        disabled={processFactors.length === 1}
                        className="p-1.5 text-red-400 hover:text-red-300 disabled:opacity-30"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  ))}

                  <button
                    onClick={addProcessFactor}
                    className="flex items-center gap-1 text-cyan-400 hover:text-cyan-300 text-sm"
                  >
                    <Plus className="w-4 h-4" />
                    Add Process Factor
                  </button>

                  <button
                    onClick={handleGenerateMixtureProcess}
                    disabled={loading}
                    className="mt-4 w-full bg-purple-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50"
                  >
                    {loading ? 'Generating...' : 'Generate Mixture + Process Design'}
                  </button>
                </div>
              )}
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
            {!showAdvanced && (
              <button
                onClick={useConstraints ? handleGenerateExtremeVertices : handleGenerateDesign}
                disabled={loading}
                className="w-full bg-cyan-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-cyan-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Generating...' : `Generate ${useConstraints ? 'Extreme Vertices' : 'Mixture'} Design`}
              </button>
            )}

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
                  {designData.design_summary && (
                    <>
                      <div>
                        <span className="text-gray-400">Vertices:</span>
                        <span className="text-gray-100 ml-2">{designData.design_summary.vertices}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Centroids:</span>
                        <span className="text-gray-100 ml-2">
                          {(designData.design_summary.edge_centroids || 0) +
                           (designData.design_summary.face_centroids || 0) +
                           (designData.design_summary.overall_centroids || 0)}
                        </span>
                      </div>
                    </>
                  )}
                  <div className="col-span-2">
                    <span className="text-gray-400">Description:</span>
                    <span className="text-gray-100 ml-2">{designData.properties?.description || designData.interpretation}</span>
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
                      className="flex items-center space-x-1 px-3 py-1 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
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
                        {getTableColumns().map((col, idx) => (
                          <th
                            key={idx}
                            className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[80px]"
                          >
                            {col}
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
                                disabled={colIndex < getTableColumns().length && designData}
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

                {/* Model Degree Selection */}
                <div className="mt-4">
                  <label className="block text-gray-100 font-medium mb-2">Scheffe Model Degree</label>
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

      {/* Visualize Tab */}
      {activeTab === 'visualize' && modelResult && (
        <div className="space-y-6">
          {/* Ternary Contour (3 components only) */}
          {componentNames.length === 3 && (
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-2xl font-bold text-gray-100">Ternary Contour Plot</h3>
                <button
                  onClick={handleGenerateTernaryContour}
                  disabled={loading}
                  className="px-4 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors disabled:opacity-50"
                >
                  {loading ? 'Generating...' : 'Generate Contour'}
                </button>
              </div>

              {ternaryContour ? (
                <>
                  {renderTernaryPlot()}

                  {ternaryContour.optimal_blend && (
                    <div className="mt-4 bg-gradient-to-r from-amber-900/20 to-orange-900/20 rounded-lg p-4 border border-amber-700/30">
                      <h4 className="text-amber-200 font-semibold mb-2">Optimal Blend (Maximum)</h4>
                      <div className="grid grid-cols-4 gap-4">
                        {componentNames.map(name => (
                          <div key={name}>
                            <p className="text-gray-400 text-sm">{name}</p>
                            <p className="text-gray-100 font-bold">{(ternaryContour.optimal_blend[name] * 100).toFixed(1)}%</p>
                          </div>
                        ))}
                        <div>
                          <p className="text-gray-400 text-sm">Predicted {responseName}</p>
                          <p className="text-amber-300 font-bold">{ternaryContour.optimal_blend.predicted_response?.toFixed(4)}</p>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="bg-slate-700/30 rounded-lg p-8 text-center">
                  <Triangle className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                  <p className="text-gray-400">Click "Generate Contour" to visualize the response surface</p>
                </div>
              )}
            </div>
          )}

          {componentNames.length !== 3 && (
            <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700/50">
              <div className="flex items-center gap-3 text-amber-400">
                <Info className="w-6 h-6" />
                <p>Ternary contour plots are only available for 3-component mixtures.</p>
              </div>
            </div>
          )}

          {/* Trace Plot */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-2xl font-bold text-gray-100">Component Trace Plot</h3>
              <button
                onClick={handleGenerateTracePlot}
                disabled={loading || !referenceBlend}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
              >
                {loading ? 'Generating...' : 'Generate Trace Plot'}
              </button>
            </div>

            {/* Reference Blend Input */}
            <div className="mb-4 bg-slate-700/30 rounded-lg p-4">
              <h4 className="text-gray-100 font-medium mb-2">Reference Blend</h4>
              <p className="text-gray-400 text-sm mb-3">
                Set the starting blend for the trace plot. Proportions must sum to 1.0.
              </p>
              <div className="flex flex-wrap gap-3">
                {componentNames.map((name, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <label className="text-gray-300">{name}:</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={referenceBlend?.[idx] ?? (1 / componentNames.length)}
                      onChange={(e) => {
                        const newRef = [...(referenceBlend || componentNames.map(() => 1/componentNames.length))]
                        newRef[idx] = parseFloat(e.target.value) || 0
                        setReferenceBlend(newRef)
                      }}
                      className="w-20 px-2 py-1 bg-slate-800/50 text-gray-100 rounded border border-slate-600"
                    />
                  </div>
                ))}
                <span className="text-gray-400 self-center">
                  Sum: {referenceBlend?.reduce((a, b) => a + b, 0)?.toFixed(3) || '1.000'}
                </span>
              </div>
            </div>

            {tracePlotData ? (
              <>
                {renderTracePlot()}

                <div className="mt-4 bg-slate-700/30 rounded-lg p-4">
                  <h4 className="text-gray-100 font-medium mb-2">Reference Blend Response</h4>
                  <p className="text-2xl font-bold text-cyan-300">{tracePlotData.reference_response?.toFixed(4)}</p>
                  <p className="text-gray-400 text-sm mt-1">
                    Each trace shows the predicted response as that component varies from 0% to 100%,
                    while other components adjust proportionally.
                  </p>
                </div>
              </>
            ) : (
              <div className="bg-slate-700/30 rounded-lg p-8 text-center">
                <TrendingUp className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                <p className="text-gray-400">Click "Generate Trace Plot" to see component effects</p>
              </div>
            )}
          </div>
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
                    <span className="text-green-400 ml-2">Mixture constraint satisfied</span>
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
