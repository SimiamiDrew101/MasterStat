import { useState } from 'react'
import { Layers, Download, PieChart, BarChart3, Table2 } from 'lucide-react'
import axios from 'axios'
import InteractionPlot from '../components/InteractionPlot'
import MainEffectsPlot from '../components/MainEffectsPlot'
import ResidualPlots from '../components/ResidualPlots'
import BoxPlot from '../components/BoxPlot'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const MixedModels = () => {
  const [analysisType, setAnalysisType] = useState('mixed-anova')

  // Mixed ANOVA state
  const [mixedTableData, setMixedTableData] = useState(
    Array(15).fill(null).map(() => Array(3).fill('')) // Factor A, Factor B, Response
  )
  const [mixedFactorNames, setMixedFactorNames] = useState(['Treatment', 'Subject'])
  const [factorTypes, setFactorTypes] = useState({
    'Treatment': 'fixed',
    'Subject': 'random'
  })

  // Split-Plot state
  const [splitPlotTableData, setSplitPlotTableData] = useState(
    Array(15).fill(null).map(() => Array(4).fill('')) // Block, Whole-plot, Sub-plot, Response
  )
  const [splitPlotFactorNames, setSplitPlotFactorNames] = useState(['Irrigation', 'Variety'])
  const [blockName, setBlockName] = useState('Block')
  const [includeBlocks, setIncludeBlocks] = useState(true)

  // Shared state
  const [responseName, setResponseName] = useState('Response')
  const [includeInteractions, setIncludeInteractions] = useState(true)
  const [alpha, setAlpha] = useState(0.05)

  // Results
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Get current data based on analysis type
  const tableData = analysisType === 'mixed-anova' ? mixedTableData : splitPlotTableData
  const setTableData = analysisType === 'mixed-anova' ? setMixedTableData : setSplitPlotTableData
  const factorNames = analysisType === 'mixed-anova' ? mixedFactorNames : splitPlotFactorNames
  const setFactorNames = analysisType === 'mixed-anova' ? setMixedFactorNames : setSplitPlotFactorNames
  const numColumns = analysisType === 'mixed-anova' ? 3 : 4

  // Handle cell changes
  const handleCellChange = (rowIndex, colIndex, value) => {
    const newData = [...tableData]
    newData[rowIndex][colIndex] = value
    setTableData(newData)

    // Auto-add row if typing in last row
    if (rowIndex === tableData.length - 1 && value.trim() !== '') {
      setTableData([...newData, Array(numColumns).fill('')])
    }
  }

  // Handle factor name change
  const handleFactorNameChange = (index, value) => {
    const newFactorNames = [...factorNames]
    const oldName = newFactorNames[index]
    newFactorNames[index] = value
    setFactorNames(newFactorNames)

    // Update factor types with new name
    if (factorTypes[oldName] !== undefined) {
      const newFactorTypes = { ...factorTypes }
      newFactorTypes[value] = newFactorTypes[oldName]
      delete newFactorTypes[oldName]
      setFactorTypes(newFactorTypes)
    } else {
      setFactorTypes({ ...factorTypes, [value]: 'fixed' })
    }
  }

  // Handle factor type change
  const handleFactorTypeChange = (factorName, type) => {
    setFactorTypes({ ...factorTypes, [factorName]: type })
  }

  // Convert table data to API format
  const prepareData = () => {
    if (analysisType === 'mixed-anova') {
      return tableData
        .filter(row => row[0] && row[1] && row[2])
        .map(row => ({
          [factorNames[0]]: row[0],
          [factorNames[1]]: row[1],
          [responseName]: parseFloat(row[2])
        }))
    } else {
      // Split-plot
      return tableData
        .filter(row => {
          if (includeBlocks) {
            return row[0] && row[1] && row[2] && row[3]
          } else {
            return row[1] && row[2] && row[3]
          }
        })
        .map(row => {
          const dataRow = {
            [factorNames[0]]: row[1], // Whole-plot factor
            [factorNames[1]]: row[2], // Sub-plot factor
            [responseName]: parseFloat(row[3])
          }
          if (includeBlocks) {
            dataRow[blockName] = row[0]
          }
          return dataRow
        })
    }
  }

  // Run analysis
  const runAnalysis = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = prepareData()

      if (data.length === 0) {
        throw new Error('Please enter data in the table')
      }

      let response
      if (analysisType === 'mixed-anova') {
        // Separate fixed and random factors
        const fixedFactors = factorNames.filter(f => factorTypes[f] === 'fixed')
        const randomFactors = factorNames.filter(f => factorTypes[f] === 'random')

        const payload = {
          data: data,
          fixed_factors: fixedFactors,
          random_factors: randomFactors,
          response: responseName,
          alpha: alpha,
          include_interactions: includeInteractions
        }

        response = await axios.post(`${API_URL}/api/mixed/mixed-model-anova`, payload)
      } else {
        // Split-plot
        const payload = {
          data: data,
          whole_plot_factor: factorNames[0],
          subplot_factor: factorNames[1],
          block: includeBlocks ? blockName : null,
          response: responseName,
          alpha: alpha
        }

        response = await axios.post(`${API_URL}/api/mixed/split-plot`, payload)
      }

      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  // Load example data
  const loadExampleData = () => {
    if (analysisType === 'mixed-anova') {
      const exampleData = [
        ['A1', 'S1', '12.5'],
        ['A1', 'S1', '13.2'],
        ['A1', 'S2', '14.8'],
        ['A1', 'S2', '15.1'],
        ['A1', 'S3', '11.9'],
        ['A1', 'S3', '12.3'],
        ['A2', 'S1', '18.4'],
        ['A2', 'S1', '19.1'],
        ['A2', 'S2', '20.2'],
        ['A2', 'S2', '19.8'],
        ['A2', 'S3', '17.6'],
        ['A2', 'S3', '18.2']
      ]

      const newTableData = [...exampleData, ...Array(Math.max(0, 15 - exampleData.length)).fill(null).map(() => Array(3).fill(''))]
      setMixedTableData(newTableData)
      setMixedFactorNames(['Treatment', 'Subject'])
      setFactorTypes({ 'Treatment': 'fixed', 'Subject': 'random' })
      setResponseName('Response')
    } else {
      // Split-plot example data
      const exampleData = [
        ['B1', 'I1', 'V1', '45.2'],
        ['B1', 'I1', 'V2', '48.5'],
        ['B1', 'I1', 'V3', '42.8'],
        ['B1', 'I2', 'V1', '52.3'],
        ['B1', 'I2', 'V2', '55.7'],
        ['B1', 'I2', 'V3', '49.1'],
        ['B2', 'I1', 'V1', '43.9'],
        ['B2', 'I1', 'V2', '47.2'],
        ['B2', 'I1', 'V3', '41.5'],
        ['B2', 'I2', 'V1', '51.6'],
        ['B2', 'I2', 'V2', '54.9'],
        ['B2', 'I2', 'V3', '48.3'],
        ['B3', 'I1', 'V1', '44.5'],
        ['B3', 'I1', 'V2', '47.9'],
        ['B3', 'I1', 'V3', '42.1'],
        ['B3', 'I2', 'V1', '52.0'],
        ['B3', 'I2', 'V2', '55.3'],
        ['B3', 'I2', 'V3', '48.7']
      ]

      const newTableData = [...exampleData, ...Array(Math.max(0, 15 - exampleData.length)).fill(null).map(() => Array(4).fill(''))]
      setSplitPlotTableData(newTableData)
      setSplitPlotFactorNames(['Irrigation', 'Variety'])
      setBlockName('Block')
      setIncludeBlocks(true)
      setResponseName('Yield')
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center space-x-3 mb-4">
          <Layers className="w-8 h-8 text-indigo-400" />
          <h2 className="text-3xl font-bold text-gray-100">Mixed Models</h2>
        </div>
        <p className="text-gray-300">
          Analyze designs with both fixed and random effects. Includes Expected Mean Squares (EMS) and variance component estimation.
        </p>
      </div>

      {/* Analysis Type Selection */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-gray-100 mb-4">Analysis Type</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <button
            onClick={() => {
              setAnalysisType('mixed-anova')
              setResult(null)
              setError(null)
            }}
            className={`p-4 rounded-lg border-2 transition ${
              analysisType === 'mixed-anova'
                ? 'border-indigo-500 bg-indigo-500/20'
                : 'border-slate-600 hover:border-indigo-400'
            }`}
          >
            <h4 className="text-lg font-semibold text-gray-100 mb-2">Mixed Model ANOVA</h4>
            <p className="text-sm text-gray-300">
              Analyze designs with fixed and random factors. Tests each factor with appropriate error terms.
            </p>
          </button>
          <button
            onClick={() => {
              setAnalysisType('split-plot')
              setResult(null)
              setError(null)
            }}
            className={`p-4 rounded-lg border-2 transition ${
              analysisType === 'split-plot'
                ? 'border-indigo-500 bg-indigo-500/20'
                : 'border-slate-600 hover:border-indigo-400'
            }`}
          >
            <h4 className="text-lg font-semibold text-gray-100 mb-2">Split-Plot Design</h4>
            <p className="text-sm text-gray-300">
              Hierarchical design with whole-plot and sub-plot factors. Uses two error terms.
            </p>
          </button>
        </div>
      </div>

      {/* Data Entry Section */}
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <h3 className="text-xl font-bold text-gray-100 mb-4">Data Entry</h3>

        {/* Factor Configuration */}
        <div className="mb-6 space-y-4">
          {analysisType === 'mixed-anova' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Factor A */}
              <div className="bg-slate-700/30 rounded-lg p-4">
                <label className="block text-gray-200 font-medium mb-2">Factor A</label>
              <input
                type="text"
                value={factorNames[0]}
                onChange={(e) => handleFactorNameChange(0, e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 mb-3"
                placeholder="e.g., Treatment"
              />
              <div className="flex space-x-4">
                <label className="flex items-center space-x-2 text-gray-300">
                  <input
                    type="radio"
                    checked={factorTypes[factorNames[0]] === 'fixed'}
                    onChange={() => handleFactorTypeChange(factorNames[0], 'fixed')}
                    className="w-4 h-4"
                  />
                  <span>Fixed Effect</span>
                </label>
                <label className="flex items-center space-x-2 text-gray-300">
                  <input
                    type="radio"
                    checked={factorTypes[factorNames[0]] === 'random'}
                    onChange={() => handleFactorTypeChange(factorNames[0], 'random')}
                    className="w-4 h-4"
                  />
                  <span>Random Effect</span>
                </label>
              </div>
              <p className="text-xs text-gray-400 mt-2">
                {factorTypes[factorNames[0]] === 'fixed'
                  ? 'Fixed: Specific chosen levels'
                  : 'Random: Sampled from population'}
              </p>
            </div>

            {/* Factor B */}
            <div className="bg-slate-700/30 rounded-lg p-4">
              <label className="block text-gray-200 font-medium mb-2">Factor B</label>
              <input
                type="text"
                value={factorNames[1]}
                onChange={(e) => handleFactorNameChange(1, e.target.value)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 mb-3"
                placeholder="e.g., Subject"
              />
              <div className="flex space-x-4">
                <label className="flex items-center space-x-2 text-gray-300">
                  <input
                    type="radio"
                    checked={factorTypes[factorNames[1]] === 'fixed'}
                    onChange={() => handleFactorTypeChange(factorNames[1], 'fixed')}
                    className="w-4 h-4"
                  />
                  <span>Fixed Effect</span>
                </label>
                <label className="flex items-center space-x-2 text-gray-300">
                  <input
                    type="radio"
                    checked={factorTypes[factorNames[1]] === 'random'}
                    onChange={() => handleFactorTypeChange(factorNames[1], 'random')}
                    className="w-4 h-4"
                  />
                  <span>Random Effect</span>
                </label>
              </div>
              <p className="text-xs text-gray-400 mt-2">
                {factorTypes[factorNames[1]] === 'fixed'
                  ? 'Fixed: Specific chosen levels'
                  : 'Random: Sampled from population'}
              </p>
            </div>
          </div>
          ) : (
            /* Split-Plot Configuration */
            <div className="space-y-4">
              {/* Include Blocks */}
              <div className="bg-slate-700/30 rounded-lg p-4">
                <label className="flex items-center space-x-2 text-gray-200">
                  <input
                    type="checkbox"
                    checked={includeBlocks}
                    onChange={(e) => setIncludeBlocks(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <span className="font-medium">Include Blocks/Replicates</span>
                </label>
                <p className="text-xs text-gray-400 mt-2">
                  Check if you have blocks (replicates) in your design. This uses RCBD at whole-plot level.
                </p>
                {includeBlocks && (
                  <input
                    type="text"
                    value={blockName}
                    onChange={(e) => setBlockName(e.target.value)}
                    className="w-full mt-3 px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    placeholder="Block name (e.g., Block, Replicate)"
                  />
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Whole-plot Factor */}
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <label className="block text-gray-200 font-medium mb-2">Whole-Plot Factor</label>
                  <input
                    type="text"
                    value={factorNames[0]}
                    onChange={(e) => handleFactorNameChange(0, e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    placeholder="e.g., Irrigation"
                  />
                  <p className="text-xs text-gray-400 mt-2">
                    Applied to large experimental units (hard to change)
                  </p>
                </div>

                {/* Sub-plot Factor */}
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <label className="block text-gray-200 font-medium mb-2">Sub-Plot Factor</label>
                  <input
                    type="text"
                    value={factorNames[1]}
                    onChange={(e) => handleFactorNameChange(1, e.target.value)}
                    className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    placeholder="e.g., Variety"
                  />
                  <p className="text-xs text-gray-400 mt-2">
                    Applied within whole-plots (easy to change)
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Response Variable Name */}
          <div>
            <label className="block text-gray-200 font-medium mb-2">Response Variable Name</label>
            <input
              type="text"
              value={responseName}
              onChange={(e) => setResponseName(e.target.value)}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder="e.g., Yield"
            />
          </div>
        </div>

        {/* Excel-like Table */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-gray-200 font-medium">Data Table</label>
            <button
              onClick={loadExampleData}
              className="px-3 py-1 text-sm bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition"
            >
              Load Example Data
            </button>
          </div>
          <div className="overflow-x-auto bg-slate-900/50 rounded-lg border border-slate-600">
            <table className="w-full">
              <thead>
                <tr className="bg-slate-700/50">
                  <th className="px-4 py-2 text-left text-gray-200 font-medium">#</th>
                  {analysisType === 'split-plot' && includeBlocks && (
                    <th className="px-4 py-2 text-left text-gray-200 font-medium">{blockName}</th>
                  )}
                  <th className="px-4 py-2 text-left text-gray-200 font-medium">
                    {analysisType === 'split-plot' ? `${factorNames[0]} (WP)` : factorNames[0]}
                  </th>
                  <th className="px-4 py-2 text-left text-gray-200 font-medium">
                    {analysisType === 'split-plot' ? `${factorNames[1]} (SP)` : factorNames[1]}
                  </th>
                  <th className="px-4 py-2 text-left text-gray-200 font-medium">{responseName}</th>
                </tr>
              </thead>
              <tbody>
                {tableData.map((row, rowIndex) => (
                  <tr key={rowIndex} className="border-t border-slate-700 hover:bg-slate-700/30">
                    <td className="px-4 py-2 text-gray-400">{rowIndex + 1}</td>
                    {analysisType === 'split-plot' && includeBlocks && (
                      <td className="px-2 py-1">
                        <input
                          type="text"
                          value={row[0]}
                          onChange={(e) => handleCellChange(rowIndex, 0, e.target.value)}
                          className="w-full px-2 py-1 bg-slate-700/50 text-gray-100 border border-slate-600 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
                          placeholder="B1"
                        />
                      </td>
                    )}
                    <td className="px-2 py-1">
                      <input
                        type="text"
                        value={analysisType === 'split-plot' && includeBlocks ? row[1] : row[0]}
                        onChange={(e) => handleCellChange(rowIndex, analysisType === 'split-plot' && includeBlocks ? 1 : 0, e.target.value)}
                        className="w-full px-2 py-1 bg-slate-700/50 text-gray-100 border border-slate-600 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
                        placeholder={analysisType === 'split-plot' ? 'I1' : 'A1'}
                      />
                    </td>
                    <td className="px-2 py-1">
                      <input
                        type="text"
                        value={analysisType === 'split-plot' && includeBlocks ? row[2] : row[1]}
                        onChange={(e) => handleCellChange(rowIndex, analysisType === 'split-plot' && includeBlocks ? 2 : 1, e.target.value)}
                        className="w-full px-2 py-1 bg-slate-700/50 text-gray-100 border border-slate-600 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
                        placeholder="e.g., S1"
                      />
                    </td>
                    <td className="px-2 py-1">
                      <input
                        type="number"
                        step="0.01"
                        value={analysisType === 'split-plot' && includeBlocks ? row[3] : row[2]}
                        onChange={(e) => handleCellChange(rowIndex, analysisType === 'split-plot' && includeBlocks ? 3 : 2, e.target.value)}
                        className="w-full px-2 py-1 bg-slate-700/50 text-gray-100 border border-slate-600 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
                        placeholder="Value"
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-gray-400 text-xs mt-2">
            Enter data directly in the table. Rows will be added automatically as you type.
          </p>
        </div>

        {/* Analysis Options */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {analysisType === 'mixed-anova' && (
            <div>
              <label className="flex items-center space-x-2 text-gray-200">
                <input
                  type="checkbox"
                  checked={includeInteractions}
                  onChange={(e) => setIncludeInteractions(e.target.checked)}
                  className="w-4 h-4"
                />
                <span>Include Interaction Effect</span>
              </label>
              <p className="text-gray-400 text-xs mt-1">
                Test for {factorNames[0]} × {factorNames[1]} interaction
              </p>
            </div>
          )}

          <div>
            <label className="block text-gray-200 font-medium mb-2">Significance Level (α)</label>
            <select
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="0.001">0.001</option>
              <option value="0.01">0.01</option>
              <option value="0.05">0.05</option>
              <option value="0.10">0.10</option>
            </select>
          </div>
        </div>

        {/* Analyze Button */}
        <button
          onClick={runAnalysis}
          disabled={loading}
          className="w-full bg-gradient-to-r from-indigo-500 to-purple-500 text-white font-bold py-3 px-6 rounded-lg hover:from-indigo-600 hover:to-purple-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Analyzing...' : analysisType === 'mixed-anova' ? 'Run Mixed Model ANOVA' : 'Run Split-Plot Analysis'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <h4 className="text-red-200 font-semibold mb-2">Error</h4>
          <p className="text-red-100">{error}</p>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="space-y-6">
          {/* Model Summary */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-xl font-bold text-gray-100 mb-4">Model Summary</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-lg p-4">
                <p className="text-gray-400 text-sm">R²</p>
                <p className="text-2xl font-bold text-indigo-400">{result.model_summary.r_squared}</p>
              </div>
              <div className="bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-lg p-4">
                <p className="text-gray-400 text-sm">Adj. R²</p>
                <p className="text-2xl font-bold text-indigo-400">{result.model_summary.adj_r_squared}</p>
              </div>
              <div className="bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-lg p-4">
                <p className="text-gray-400 text-sm">F-statistic</p>
                <p className="text-2xl font-bold text-indigo-400">{result.model_summary.f_statistic}</p>
              </div>
            </div>
          </div>

          {/* ANOVA Table with EMS */}
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-xl font-bold text-gray-100 mb-4">ANOVA Table</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-700/50">
                    <th className="px-4 py-3 text-left text-gray-200 font-medium">Source</th>
                    <th className="px-4 py-3 text-right text-gray-200 font-medium">Sum of Squares</th>
                    <th className="px-4 py-3 text-right text-gray-200 font-medium">df</th>
                    <th className="px-4 py-3 text-right text-gray-200 font-medium">Mean Square</th>
                    <th className="px-4 py-3 text-left text-gray-200 font-medium">Expected Mean Squares</th>
                    <th className="px-4 py-3 text-right text-gray-200 font-medium">F</th>
                    <th className="px-4 py-3 text-right text-gray-200 font-medium">p-value</th>
                    <th className="px-4 py-3 text-left text-gray-200 font-medium">Error Term</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.anova_table).map(([source, data], idx) => {
                    const isSignificant = data.p_value_corrected
                      ? data.p_value_corrected < alpha
                      : data.p_value
                      ? data.p_value < alpha
                      : false

                    return (
                      <tr
                        key={idx}
                        className={`border-t border-slate-700 ${
                          isSignificant && source !== 'Residual'
                            ? 'bg-green-900/20'
                            : 'hover:bg-slate-700/30'
                        }`}
                      >
                        <td className="px-4 py-3 text-gray-200 font-medium">{source}</td>
                        <td className="px-4 py-3 text-right text-gray-300">{data.sum_sq}</td>
                        <td className="px-4 py-3 text-right text-gray-300">{data.df}</td>
                        <td className="px-4 py-3 text-right text-gray-300">{data.mean_sq}</td>
                        <td className="px-4 py-3 text-left text-gray-300 font-mono text-xs">
                          {data.ems || '-'}
                        </td>
                        <td className="px-4 py-3 text-right text-gray-300">
                          {data.F_corrected !== undefined ? data.F_corrected : data.F || '-'}
                        </td>
                        <td className="px-4 py-3 text-right text-gray-300">
                          {data.p_value_corrected !== undefined
                            ? data.p_value_corrected < 0.001
                              ? '<0.001'
                              : data.p_value_corrected.toFixed(4)
                            : data.p_value
                            ? data.p_value < 0.001
                              ? '<0.001'
                              : data.p_value.toFixed(4)
                            : '-'}
                          {isSignificant && source !== 'Residual' && ' *'}
                        </td>
                        <td className="px-4 py-3 text-left text-gray-400 text-xs">
                          {data.error_term || '-'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="text-gray-400 text-xs mt-4">
              * Significant at α = {alpha}. Expected Mean Squares (EMS) show what each MS estimates.
            </p>
          </div>

          {/* Variance Components */}
          {result.variance_components && (
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
              <h3 className="text-xl font-bold text-gray-100 mb-4">Variance Components</h3>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Variance Components Table */}
                <div>
                  <h4 className="text-lg font-semibold text-gray-200 mb-3">Estimates</h4>
                  <div className="overflow-x-auto bg-slate-900/50 rounded-lg border border-slate-600">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="bg-slate-700/50">
                          <th className="px-4 py-2 text-left text-gray-200 font-medium">Component</th>
                          <th className="px-4 py-2 text-right text-gray-200 font-medium">Estimate</th>
                          <th className="px-4 py-2 text-right text-gray-200 font-medium">% of Total</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(result.variance_components)
                          .filter(([_, value]) => value !== null)
                          .map(([component, value], idx) => (
                            <tr key={idx} className="border-t border-slate-700">
                              <td className="px-4 py-2 text-gray-300 font-mono text-xs">{component}</td>
                              <td className="px-4 py-2 text-right text-gray-300">{value.toFixed(6)}</td>
                              <td className="px-4 py-2 text-right text-gray-300">
                                {result.variance_percentages[component]
                                  ? result.variance_percentages[component].toFixed(1) + '%'
                                  : '-'}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-gray-400 text-xs mt-2">
                    Variance components partition total variability into sources
                  </p>
                </div>

                {/* Variance Percentage Visualization */}
                <div>
                  <h4 className="text-lg font-semibold text-gray-200 mb-3">Distribution</h4>
                  <div className="space-y-3">
                    {Object.entries(result.variance_percentages || {}).map(([component, percentage], idx) => {
                      const colors = [
                        'bg-indigo-500',
                        'bg-purple-500',
                        'bg-pink-500',
                        'bg-blue-500',
                        'bg-cyan-500'
                      ]
                      const color = colors[idx % colors.length]

                      return (
                        <div key={idx}>
                          <div className="flex justify-between text-sm text-gray-300 mb-1">
                            <span className="font-mono text-xs">{component}</span>
                            <span className="font-semibold">{percentage.toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-slate-700/50 rounded-full h-6 overflow-hidden">
                            <div
                              className={`${color} h-full flex items-center justify-center text-white text-xs font-bold transition-all duration-500`}
                              style={{ width: `${percentage}%` }}
                            >
                              {percentage > 10 && `${percentage.toFixed(1)}%`}
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                  <p className="text-gray-400 text-xs mt-4">
                    Bar chart shows relative contribution of each variance component to total variability
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Interpretation */}
          {result.interpretation && result.interpretation.length > 0 && (
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
              <h3 className="text-xl font-bold text-gray-100 mb-4">Interpretation</h3>
              <div className="space-y-2">
                {result.interpretation.map((line, idx) => (
                  <p key={idx} className="text-gray-300">
                    {line}
                  </p>
                ))}
              </div>
            </div>
          )}

          {/* Visualizations */}
          {result.plot_data && (
            <>
              {/* Interaction Plot */}
              {result.plot_data.cell_means && result.plot_data.cell_means.length > 0 && (() => {
                // Transform cell_means data to match InteractionPlot expected format
                const factors = analysisType === 'mixed-anova'
                  ? factorNames
                  : [result.whole_plot_factor, result.subplot_factor]

                const interactionData = {}
                result.plot_data.cell_means.forEach(cell => {
                  const key = `${cell[factors[0]]}, ${cell[factors[1]]}`
                  interactionData[key] = cell.mean
                })

                return (
                  <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
                    <h3 className="text-xl font-bold text-gray-100 mb-4">Interaction Plot</h3>
                    <p className="text-gray-400 text-sm mb-4">
                      Shows cell means for all factor combinations. Non-parallel lines indicate interaction.
                    </p>
                    <InteractionPlot
                      data={interactionData}
                      factorAName={factors[0]}
                      factorBName={factors[1]}
                    />
                  </div>
                )
              })()}

              {/* Main Effects Plots */}
              {result.plot_data.marginal_means && (() => {
                // Transform marginal_means to match MainEffectsPlot expected format
                const transformedData = {}
                Object.entries(result.plot_data.marginal_means).forEach(([factor, data]) => {
                  transformedData[factor] = {
                    levels: data.map(d => d.level),
                    means: data.map(d => d.mean)
                  }
                })

                return (
                  <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
                    <h3 className="text-xl font-bold text-gray-100 mb-4">Main Effects Plots</h3>
                    <p className="text-gray-400 text-sm mb-4">
                      Shows marginal means for each factor level. Steeper slopes indicate stronger effects.
                    </p>
                    <MainEffectsPlot
                      data={transformedData}
                      responseName={responseName}
                    />
                  </div>
                )
              })()}

              {/* Residual Diagnostic Plots */}
              {result.plot_data.residuals && result.plot_data.fitted_values && (() => {
                // Calculate standardized residuals
                const residuals = result.plot_data.residuals.filter(r => r !== null)
                const mean = residuals.reduce((sum, r) => sum + r, 0) / residuals.length
                const variance = residuals.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / residuals.length
                const stdDev = Math.sqrt(variance)
                const standardizedResiduals = residuals.map(r => r / stdDev)

                return (
                  <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
                    <h3 className="text-xl font-bold text-gray-100 mb-4">Residual Diagnostic Plots</h3>
                    <p className="text-gray-400 text-sm mb-4">
                      Check model assumptions: normality (Q-Q plot) and homogeneity of variance (residuals vs fitted).
                    </p>
                    <ResidualPlots
                      residuals={residuals}
                      fittedValues={result.plot_data.fitted_values.filter(f => f !== null)}
                      standardizedResiduals={standardizedResiduals}
                    />
                  </div>
                )
              })()}

              {/* Box Plots by Factor */}
              {result.plot_data.box_plot_data && (() => {
                // Helper function to calculate box plot statistics
                const calculateBoxStats = (values) => {
                  const cleanValues = values.filter(v => v !== null).sort((a, b) => a - b)
                  if (cleanValues.length === 0) return null

                  const q1Index = Math.floor(cleanValues.length * 0.25)
                  const medianIndex = Math.floor(cleanValues.length * 0.5)
                  const q3Index = Math.floor(cleanValues.length * 0.75)

                  const q1 = cleanValues[q1Index]
                  const median = cleanValues[medianIndex]
                  const q3 = cleanValues[q3Index]
                  const iqr = q3 - q1

                  const lowerFence = q1 - 1.5 * iqr
                  const upperFence = q3 + 1.5 * iqr

                  const outliers = cleanValues.filter(v => v < lowerFence || v > upperFence)
                  const nonOutliers = cleanValues.filter(v => v >= lowerFence && v <= upperFence)

                  return {
                    min: nonOutliers.length > 0 ? Math.min(...nonOutliers) : cleanValues[0],
                    q1,
                    median,
                    q3,
                    max: nonOutliers.length > 0 ? Math.max(...nonOutliers) : cleanValues[cleanValues.length - 1],
                    outliers
                  }
                }

                return (
                  <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
                    <h3 className="text-xl font-bold text-gray-100 mb-4">Distribution by Factor</h3>
                    <p className="text-gray-400 text-sm mb-4">
                      Shows the distribution of response values for each level of each factor.
                    </p>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {Object.entries(result.plot_data.box_plot_data).map(([factor, data]) => {
                        // Transform data to match BoxPlot expected format
                        const transformedData = data
                          .map(d => {
                            const stats = calculateBoxStats(d.values)
                            if (!stats) return null
                            return {
                              label: d.level,
                              ...stats
                            }
                          })
                          .filter(d => d !== null)

                        return (
                          <div key={factor}>
                            <h4 className="text-lg font-semibold text-gray-200 mb-3">{factor}</h4>
                            <BoxPlot
                              data={transformedData}
                              factorName={factor}
                              responseName={responseName}
                            />
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )
              })()}
            </>
          )}

          {/* Export Button */}
          <div className="flex justify-end">
            <button
              onClick={() => {
                const exportData = {
                  model_type: result.model_type,
                  fixed_factors: result.fixed_factors,
                  random_factors: result.random_factors,
                  anova_table: result.anova_table,
                  variance_components: result.variance_components,
                  variance_percentages: result.variance_percentages,
                  model_summary: result.model_summary,
                  interpretation: result.interpretation
                }
                const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = `mixed-model-anova-${new Date().toISOString().slice(0, 10)}.json`
                a.click()
                URL.revokeObjectURL(url)
              }}
              className="px-6 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition flex items-center space-x-2"
            >
              <Download className="w-4 h-4" />
              <span>Export Results (JSON)</span>
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default MixedModels
