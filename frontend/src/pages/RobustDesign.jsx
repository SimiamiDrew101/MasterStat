import { useState, useEffect } from 'react'
import axios from 'axios'
import { Shield, Plus, Trash2, Target, TrendingUp } from 'lucide-react'
import Plot from 'react-plotly.js'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const RobustDesign = () => {
  // Design configuration
  const [numControlFactors, setNumControlFactors] = useState(3)
  const [numNoiseFactors, setNumNoiseFactors] = useState(2)
  const [controlDesignType, setControlDesignType] = useState('orthogonal_array')
  const [noiseDesignType, setNoiseDesignType] = useState('full_factorial')
  const [controlFactorNames, setControlFactorNames] = useState(['C1', 'C2', 'C3'])
  const [noiseFactorNames, setNoiseFactorNames] = useState(['N1', 'N2'])
  const [responseName, setResponseName] = useState('Y')
  const [qualityCharacteristic, setQualityCharacteristic] = useState('smaller-is-better')

  // Data and results
  const [designData, setDesignData] = useState(null)
  const [tableData, setTableData] = useState([])
  const [analysisResult, setAnalysisResult] = useState(null)

  // UI state
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('design')

  // Update factor names when numbers change
  useEffect(() => {
    const newControlNames = Array.from({ length: numControlFactors }, (_, i) => `C${i + 1}`)
    setControlFactorNames(newControlNames)
  }, [numControlFactors])

  useEffect(() => {
    const newNoiseNames = Array.from({ length: numNoiseFactors }, (_, i) => `N${i + 1}`)
    setNoiseFactorNames(newNoiseNames)
  }, [numNoiseFactors])

  // Generate robust design
  const handleGenerateDesign = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await axios.post(`${API_URL}/api/rsm/robust-design/generate`, {
        n_control_factors: numControlFactors,
        n_noise_factors: numNoiseFactors,
        control_design_type: controlDesignType,
        noise_design_type: noiseDesignType,
        control_factor_names: controlFactorNames,
        noise_factor_names: noiseFactorNames
      })

      setDesignData(response.data)

      // Convert to table format with empty response column
      const table = response.data.design_matrix.map(row => {
        const tableRow = []
        // Add control factors
        controlFactorNames.forEach(cf => tableRow.push(row[cf] || 0))
        // Add noise factors
        noiseFactorNames.forEach(nf => tableRow.push(row[nf] || 0))
        // Add empty response
        tableRow.push('')
        // Store control and noise run numbers for reference
        tableRow.controlRun = row.control_run
        tableRow.noiseRun = row.noise_run
        return tableRow
      })

      setTableData(table)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate design')
    } finally {
      setLoading(false)
    }
  }

  // Analyze robust design
  const handleAnalyze = async () => {
    setLoading(true)
    setError(null)

    try {
      // Validate data
      const validRows = tableData.filter(row => {
        const responseValue = row[row.length - 1]
        return responseValue !== '' && responseValue !== null && responseValue !== undefined
      })

      if (validRows.length < numControlFactors * numNoiseFactors) {
        throw new Error(`Need complete data for all ${designData.total_runs} runs`)
      }

      // Convert to API format
      const data = validRows.map(row => {
        const point = {}
        // Add control factors
        controlFactorNames.forEach((cf, i) => {
          point[cf] = parseFloat(row[i])
        })
        // Add noise factors
        noiseFactorNames.forEach((nf, i) => {
          point[nf] = parseFloat(row[numControlFactors + i])
        })
        point[responseName] = parseFloat(row[row.length - 1])
        return point
      })

      const response = await axios.post(`${API_URL}/api/rsm/robust-design/analyze`, {
        data: data,
        control_factors: controlFactorNames,
        noise_factors: noiseFactorNames,
        response: responseName,
        quality_characteristic: qualityCharacteristic
      })

      setAnalysisResult(response.data)
      setActiveTab('analysis')
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed')
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

  // Fill test data
  const fillTestData = () => {
    const newData = tableData.map(row => {
      const newRow = [...row]
      // Get control and noise factor values
      const controlValues = row.slice(0, numControlFactors)
      const noiseValues = row.slice(numControlFactors, numControlFactors + numNoiseFactors)

      // Generate response based on control and noise
      let response = 50 // baseline

      // Control factors contribute to mean
      controlValues.forEach((val, i) => {
        response += (5 + i * 2) * parseFloat(val)
      })

      // Noise factors add variability
      noiseValues.forEach((val, i) => {
        response += (2 + i) * parseFloat(val) * (Math.random() - 0.5) * 5
      })

      // Additional random noise
      response += (Math.random() - 0.5) * 3

      newRow[newRow.length - 1] = response.toFixed(2)
      return newRow
    })
    setTableData(newData)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-900/30 to-teal-900/30 backdrop-blur-lg rounded-2xl p-8 border border-emerald-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <Shield className="w-10 h-10 text-emerald-400" />
          <h2 className="text-4xl font-bold text-gray-100">Robust Parameter Design</h2>
        </div>
        <p className="text-gray-300 text-lg">
          Taguchi-style designs to identify control factor settings that minimize sensitivity to noise factors.
          Optimize process robustness using Signal-to-Noise ratios.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-2 bg-slate-800/50 p-2 rounded-lg">
        <button
          onClick={() => setActiveTab('design')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'design'
              ? 'bg-emerald-600 text-white'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
        >
          1. Design
        </button>
        <button
          onClick={() => setActiveTab('analysis')}
          className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'analysis'
              ? 'bg-emerald-600 text-white'
              : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
          }`}
          disabled={!analysisResult}
        >
          2. Analysis & Results
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
          <h3 className="text-2xl font-bold text-gray-100 mb-6">Robust Design Configuration</h3>

          <div className="space-y-6">
            {/* Control Factors */}
            <div className="bg-gradient-to-r from-blue-900/20 to-indigo-900/20 rounded-lg p-5 border border-blue-700/30">
              <h4 className="text-blue-200 font-semibold mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Control Factors (Inner Array)
              </h4>
              <p className="text-gray-300 text-sm mb-4">
                Factors you can control and want to optimize
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-100 font-medium mb-2">Number of Control Factors</label>
                  <input
                    type="number"
                    min={1}
                    max={7}
                    value={numControlFactors}
                    onChange={(e) => setNumControlFactors(parseInt(e.target.value) || 1)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  />
                </div>

                <div>
                  <label className="block text-gray-100 font-medium mb-2">Inner Array Design</label>
                  <select
                    value={controlDesignType}
                    onChange={(e) => setControlDesignType(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  >
                    <option value="orthogonal_array">Orthogonal Array (L4/L8)</option>
                    <option value="full_factorial">Full Factorial</option>
                  </select>
                </div>
              </div>

              <div className="mt-4">
                <label className="block text-gray-100 font-medium mb-2">Control Factor Names</label>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {controlFactorNames.map((name, idx) => (
                    <input
                      key={idx}
                      type="text"
                      value={name}
                      onChange={(e) => {
                        const newNames = [...controlFactorNames]
                        newNames[idx] = e.target.value
                        setControlFactorNames(newNames)
                      }}
                      className="px-3 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                      placeholder={`Control ${idx + 1}`}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* Noise Factors */}
            <div className="bg-gradient-to-r from-amber-900/20 to-orange-900/20 rounded-lg p-5 border border-amber-700/30">
              <h4 className="text-amber-200 font-semibold mb-4 flex items-center gap-2">
                <Shield className="w-5 h-5" />
                Noise Factors (Outer Array)
              </h4>
              <p className="text-gray-300 text-sm mb-4">
                Uncontrollable factors that cause variability (environmental, wear, etc.)
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-100 font-medium mb-2">Number of Noise Factors</label>
                  <input
                    type="number"
                    min={1}
                    max={7}
                    value={numNoiseFactors}
                    onChange={(e) => setNumNoiseFactors(parseInt(e.target.value) || 1)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  />
                </div>

                <div>
                  <label className="block text-gray-100 font-medium mb-2">Outer Array Design</label>
                  <select
                    value={noiseDesignType}
                    onChange={(e) => setNoiseDesignType(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  >
                    <option value="full_factorial">Full Factorial</option>
                    <option value="orthogonal_array">Orthogonal Array (L4/L8)</option>
                  </select>
                </div>
              </div>

              <div className="mt-4">
                <label className="block text-gray-100 font-medium mb-2">Noise Factor Names</label>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {noiseFactorNames.map((name, idx) => (
                    <input
                      key={idx}
                      type="text"
                      value={name}
                      onChange={(e) => {
                        const newNames = [...noiseFactorNames]
                        newNames[idx] = e.target.value
                        setNoiseFactorNames(newNames)
                      }}
                      className="px-3 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                      placeholder={`Noise ${idx + 1}`}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* Response and Quality Characteristic */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-100 font-medium mb-2">Response Variable Name</label>
                <input
                  type="text"
                  value={responseName}
                  onChange={(e) => setResponseName(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                />
              </div>

              <div>
                <label className="block text-gray-100 font-medium mb-2">Quality Characteristic</label>
                <select
                  value={qualityCharacteristic}
                  onChange={(e) => setQualityCharacteristic(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                >
                  <option value="smaller-is-better">Smaller is Better</option>
                  <option value="larger-is-better">Larger is Better</option>
                  <option value="nominal-is-best">Nominal is Best</option>
                </select>
                <p className="text-gray-400 text-xs mt-1">
                  {qualityCharacteristic === 'smaller-is-better' && 'Minimize response (e.g., defects, cost)'}
                  {qualityCharacteristic === 'larger-is-better' && 'Maximize response (e.g., strength, yield)'}
                  {qualityCharacteristic === 'nominal-is-best' && 'Hit target with minimum variation'}
                </p>
              </div>
            </div>

            {/* Generate Button */}
            <button
              onClick={handleGenerateDesign}
              disabled={loading}
              className="w-full bg-emerald-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-emerald-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Generating...' : 'Generate Robust Design'}
            </button>

            {/* Design Properties */}
            {designData && (
              <div className="bg-gradient-to-r from-emerald-900/20 to-teal-900/20 rounded-lg p-5 border border-emerald-700/30">
                <h4 className="text-emerald-200 font-semibold mb-3">Design Properties</h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Design Type:</span>
                    <span className="text-gray-100 ml-2 font-medium">{designData.design_type}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Control Array:</span>
                    <span className="text-gray-100 ml-2 font-medium">{designData.control_array_size} runs</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Noise Array:</span>
                    <span className="text-gray-100 ml-2 font-medium">{designData.noise_array_size} runs</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Total Runs:</span>
                    <span className="text-gray-100 ml-2 font-medium">{designData.total_runs}</span>
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
                  <button
                    onClick={fillTestData}
                    disabled={!designData}
                    className="flex items-center space-x-1 px-3 py-1 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm disabled:opacity-50"
                  >
                    <Target className="w-4 h-4" />
                    <span>Fill Test Data</span>
                  </button>
                </div>

                <div className="overflow-x-auto bg-slate-700/30 rounded-lg border-2 border-slate-600">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-slate-700/70">
                        <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 w-14">
                          #
                        </th>
                        {controlFactorNames.map((cf, idx) => (
                          <th
                            key={`c-${idx}`}
                            className="px-3 py-2 text-center text-blue-200 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[70px] bg-blue-900/20"
                          >
                            {cf}
                          </th>
                        ))}
                        {noiseFactorNames.map((nf, idx) => (
                          <th
                            key={`n-${idx}`}
                            className="px-3 py-2 text-center text-amber-200 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[70px] bg-amber-900/20"
                          >
                            {nf}
                          </th>
                        ))}
                        <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-slate-600 min-w-[100px] bg-emerald-900/20">
                          {responseName}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {tableData.map((row, rowIndex) => (
                        <tr key={rowIndex} className="border-b border-slate-700/30 hover:bg-slate-600/10">
                          <td className="px-3 py-2 text-center text-gray-300 text-sm font-medium bg-slate-700/30 border-r border-slate-600">
                            {rowIndex + 1}
                          </td>
                          {row.slice(0, -1).map((cell, colIndex) => (
                            <td key={colIndex} className="px-1 py-1 border-r border-slate-700/20">
                              <input
                                type="text"
                                value={cell}
                                onChange={(e) => handleCellChange(rowIndex, colIndex, e.target.value)}
                                disabled={colIndex < numControlFactors + numNoiseFactors && designData}
                                className="w-full px-2 py-1.5 bg-slate-800/50 text-gray-100 border border-slate-600/50 focus:border-emerald-500 focus:bg-slate-700/50 rounded-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/50 text-sm transition-all disabled:opacity-60"
                                placeholder={colIndex === row.length - 2 ? '0.0' : ''}
                              />
                            </td>
                          ))}
                          <td className="px-1 py-1">
                            <input
                              type="text"
                              value={row[row.length - 1]}
                              onChange={(e) => handleCellChange(rowIndex, row.length - 1, e.target.value)}
                              className="w-full px-2 py-1.5 bg-slate-800/50 text-gray-100 border border-slate-600/50 focus:border-emerald-500 focus:bg-slate-700/50 rounded-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/50 text-sm transition-all"
                              placeholder="0.0"
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <p className="text-gray-400 text-xs mt-2">
                  <span className="text-blue-300">Control factors (blue)</span> are from inner array.
                  <span className="text-amber-300 ml-2">Noise factors (amber)</span> are from outer array.
                  Enter response values for each run.
                </p>

                <button
                  onClick={handleAnalyze}
                  disabled={loading}
                  className="w-full mt-4 bg-green-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Analyzing...' : 'Analyze Robust Design'}
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Analysis Tab */}
      {activeTab === 'analysis' && analysisResult && (
        <div className="space-y-6">
          {/* Optimal Settings */}
          <div className="bg-gradient-to-r from-green-900/30 to-emerald-900/30 backdrop-blur-lg rounded-2xl p-6 border border-green-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-4">Optimal Robust Settings</h3>

            <div className="mb-6">
              <div className="bg-slate-700/50 rounded-lg p-4 inline-block">
                <p className="text-gray-400 text-sm">Signal-to-Noise Ratio</p>
                <p className="text-3xl font-bold text-green-300">{analysisResult.optimal_settings.sn_ratio} dB</p>
                <p className="text-gray-400 text-xs mt-1">Higher is better (more robust)</p>
              </div>
            </div>

            <h4 className="text-gray-100 font-semibold mb-3">Optimal Control Factor Settings:</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              {Object.entries(analysisResult.optimal_settings.control_factors).map(([factor, value]) => (
                <div key={factor} className="bg-slate-700/50 rounded-lg p-4">
                  <p className="text-gray-400 text-sm">{factor}</p>
                  <p className="text-2xl font-bold text-gray-100">{value}</p>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-700/30 rounded-lg p-4">
                <p className="text-gray-400 text-sm">Predicted Mean</p>
                <p className="text-xl font-bold text-gray-100">{analysisResult.optimal_settings.predicted_mean}</p>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-4">
                <p className="text-gray-400 text-sm">Predicted Std Dev</p>
                <p className="text-xl font-bold text-gray-100">{analysisResult.optimal_settings.predicted_std_dev}</p>
              </div>
            </div>

            <div className="mt-4 bg-slate-700/30 rounded-lg p-4">
              <p className="text-gray-300 text-sm">
                <strong>Interpretation:</strong> {analysisResult.interpretation}
              </p>
            </div>
          </div>

          {/* Main Effects for SN Ratios */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-2xl font-bold text-gray-100 mb-4">Main Effects on S/N Ratio</h3>
            <p className="text-gray-300 text-sm mb-6">
              Shows how each control factor level affects robustness. Choose levels with highest S/N ratio.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {Object.entries(analysisResult.main_effects).map(([factor, data]) => (
                <div key={factor} className="bg-slate-700/30 rounded-lg p-4">
                  <h5 className="text-gray-100 font-semibold mb-3">{factor}</h5>
                  <div className="space-y-2">
                    {Object.entries(data.level_means).map(([level, snMean]) => (
                      <div key={level} className="flex items-center justify-between">
                        <span className="text-gray-300">Level {level}:</span>
                        <span className={`font-bold ${
                          snMean === Math.max(...Object.values(data.level_means))
                            ? 'text-green-400'
                            : 'text-gray-100'
                        }`}>
                          {snMean} dB
                        </span>
                      </div>
                    ))}
                  </div>
                  <div className="mt-3 pt-3 border-t border-slate-600">
                    <p className="text-gray-400 text-sm">
                      Effect Size: <span className="text-gray-100 font-semibold">{data.effect_size} dB</span>
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Main Effects Plot */}
          {analysisResult.main_effects && (
            <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700/50">
              <h4 className="text-gray-100 font-semibold mb-4">Main Effects Plot</h4>
              <Plot
                data={Object.entries(analysisResult.main_effects).map(([factor, data]) => ({
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: factor,
                  x: Object.keys(data.level_means).map(l => `Level ${l}`),
                  y: Object.values(data.level_means),
                  marker: { size: 10 },
                  line: { width: 3 }
                }))}
                layout={{
                  title: {
                    text: 'Main Effects on S/N Ratio',
                    font: { size: 18, color: '#f1f5f9' }
                  },
                  xaxis: {
                    title: 'Factor Level',
                    gridcolor: '#475569'
                  },
                  yaxis: {
                    title: 'Mean S/N Ratio (dB)',
                    gridcolor: '#475569'
                  },
                  paper_bgcolor: '#334155',
                  plot_bgcolor: '#1e293b',
                  font: { color: '#e2e8f0' },
                  showlegend: true,
                  legend: {
                    bgcolor: 'rgba(30, 41, 59, 0.8)',
                    bordercolor: '#475569',
                    borderwidth: 1
                  }
                }}
                config={{
                  responsive: true,
                  displayModeBar: true,
                  displaylogo: false
                }}
                style={{ width: '100%', height: '500px' }}
                useResizeHandler={true}
              />
            </div>
          )}

          {/* All SN Ratios Table */}
          <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700/50">
            <h4 className="text-gray-100 font-semibold mb-4">All Control Factor Combinations</h4>
            <div className="overflow-x-auto bg-slate-700/30 rounded-lg">
              <table className="w-full">
                <thead>
                  <tr className="bg-slate-700/70">
                    <th className="px-4 py-2 text-left text-gray-100 font-semibold border-b border-slate-600">Rank</th>
                    {controlFactorNames.map((cf, idx) => (
                      <th key={idx} className="px-4 py-2 text-center text-gray-100 font-semibold border-b border-slate-600">
                        {cf}
                      </th>
                    ))}
                    <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">Mean</th>
                    <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">Std Dev</th>
                    <th className="px-4 py-2 text-right text-gray-100 font-semibold border-b border-slate-600">S/N Ratio (dB)</th>
                  </tr>
                </thead>
                <tbody>
                  {analysisResult.sn_ratios.map((entry, idx) => (
                    <tr
                      key={idx}
                      className={`border-b border-slate-700/30 hover:bg-slate-600/10 ${
                        idx === 0 ? 'bg-green-900/20' : ''
                      }`}
                    >
                      <td className="px-4 py-2 text-gray-100 font-bold">
                        {idx === 0 && <span className="text-green-400">★ </span>}
                        #{idx + 1}
                      </td>
                      {controlFactorNames.map((cf, cfIdx) => (
                        <td key={cfIdx} className="px-4 py-2 text-center text-gray-100">
                          {entry.control_setting[cf]}
                        </td>
                      ))}
                      <td className="px-4 py-2 text-right text-gray-100">{entry.mean_response}</td>
                      <td className="px-4 py-2 text-right text-gray-100">{entry.std_dev}</td>
                      <td className="px-4 py-2 text-right text-gray-100 font-bold">
                        {entry.sn_ratio}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Quality Characteristic Info */}
          <div className="bg-slate-700/30 rounded-lg p-4">
            <p className="text-gray-300 text-sm">
              <strong>Quality Characteristic:</strong> {analysisResult.quality_characteristic}
              <br />
              <strong>Analysis Method:</strong> Taguchi Signal-to-Noise (S/N) ratio analysis
              <br />
              <strong>Recommendations:</strong> Use the optimal control factor settings (rank #1) to achieve robust performance across varying noise conditions.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default RobustDesign
