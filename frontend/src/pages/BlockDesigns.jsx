import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import ResultCard from '../components/ResultCard'
import { Grid, Plus, Trash2, Shuffle } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const BlockDesigns = () => {
  const [designType, setDesignType] = useState('rcbd')
  const [nTreatments, setNTreatments] = useState(4)
  const [nBlocks, setNBlocks] = useState(3)
  const [squareSize, setSquareSize] = useState(4)
  const [responseName, setResponseName] = useState('Response')
  const [alpha, setAlpha] = useState(0.05)
  const [randomize, setRandomize] = useState(true)
  const [randomBlocks, setRandomBlocks] = useState(false)
  const [tableData, setTableData] = useState([])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [generatedDesign, setGeneratedDesign] = useState(null)

  // Generate random normal value
  const randomNormal = (mean = 100, stdDev = 15) => {
    const u1 = Math.random()
    const u2 = Math.random()
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
    return mean + z0 * stdDev
  }

  // Generate design based on type
  const handleGenerateDesign = async () => {
    setLoading(true)
    setError(null)

    try {
      let response
      if (designType === 'rcbd') {
        response = await axios.post(`${API_URL}/api/block-designs/generate/rcbd`, {
          n_treatments: nTreatments,
          n_blocks: nBlocks,
          randomize: randomize
        })
      } else if (designType === 'latin') {
        response = await axios.post(`${API_URL}/api/block-designs/generate/latin-square`, {
          size: squareSize
        })
      } else if (designType === 'graeco') {
        response = await axios.post(`${API_URL}/api/block-designs/generate/graeco-latin`, {
          size: squareSize
        })
      }

      setGeneratedDesign(response.data)

      // Convert design to table data and populate with random responses
      const design = response.data.design_table
      const tableRows = design.map(row => {
        const response = (Math.round(randomNormal(100, 15) * 10) / 10).toFixed(1)
        if (designType === 'rcbd') {
          return [row.run_order, row.block, row.treatment, response]
        } else if (designType === 'latin') {
          return [row.row, row.column, row.treatment, response]
        } else {
          return [row.row, row.column, row.latin_treatment, row.greek_treatment, response]
        }
      })

      setTableData(tableRows)
      setResult(null)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  // Auto-generate design on mount or when parameters change
  useEffect(() => {
    if (designType && ((designType === 'rcbd' && nTreatments >= 2 && nBlocks >= 2) ||
        ((designType === 'latin' || designType === 'graeco') && squareSize >= 3))) {
      handleGenerateDesign()
    }
  }, [designType, nTreatments, nBlocks, squareSize, randomize])

  const handleCellChange = (rowIndex, colIndex, value) => {
    const newData = [...tableData]
    newData[rowIndex][colIndex] = value
    setTableData(newData)
  }

  // Excel-like keyboard navigation
  const handleKeyDown = (e, rowIndex, colIndex) => {
    const maxRow = tableData.length - 1
    const maxCol = tableData[0]?.length - 1 || 0

    switch (e.key) {
      case 'ArrowUp':
        e.preventDefault()
        if (rowIndex > 0) {
          document.getElementById(`cell-${rowIndex - 1}-${colIndex}`)?.focus()
        }
        break
      case 'ArrowDown':
        e.preventDefault()
        if (rowIndex < maxRow) {
          document.getElementById(`cell-${rowIndex + 1}-${colIndex}`)?.focus()
        }
        break
      case 'ArrowLeft':
        if (e.target.selectionStart === 0 && colIndex > 0) {
          e.preventDefault()
          document.getElementById(`cell-${rowIndex}-${colIndex - 1}`)?.focus()
        }
        break
      case 'ArrowRight':
        if (e.target.selectionStart === e.target.value.length && colIndex < maxCol) {
          e.preventDefault()
          document.getElementById(`cell-${rowIndex}-${colIndex + 1}`)?.focus()
        }
        break
      case 'Enter':
        e.preventDefault()
        if (rowIndex < maxRow) {
          document.getElementById(`cell-${rowIndex + 1}-${colIndex}`)?.focus()
        }
        break
      case 'Tab':
        if (!e.shiftKey && colIndex < maxCol) {
          e.preventDefault()
          document.getElementById(`cell-${rowIndex}-${colIndex + 1}`)?.focus()
        } else if (e.shiftKey && colIndex > 0) {
          e.preventDefault()
          document.getElementById(`cell-${rowIndex}-${colIndex - 1}`)?.focus()
        }
        break
      default:
        break
    }
  }

  const handleCellClick = (e) => {
    e.target.select()
  }

  const addRow = () => {
    const numCols = tableData[0]?.length || (designType === 'graeco' ? 5 : 4)
    const newRow = Array(numCols).fill('')
    setTableData([...tableData, newRow])
  }

  const removeRow = (rowIndex) => {
    if (tableData.length > 1) {
      setTableData(tableData.filter((_, idx) => idx !== rowIndex))
    }
  }

  const handleAnalyze = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      if (tableData.length < 2) {
        throw new Error('Please provide at least 2 rows of data')
      }

      let payload, endpoint

      if (designType === 'rcbd') {
        const data = tableData
          .map((row, idx) => ({
            run_order: parseInt(row[0]) || idx + 1,
            block: String(row[1] || '').trim(),
            treatment: String(row[2] || '').trim(),
            [responseName]: parseFloat(row[3])
          }))
          .filter(row => {
            // Filter out rows with missing or invalid data
            const isValid = row.block && row.treatment && !isNaN(row[responseName])
            return isValid
          })

        if (data.length < 2) {
          throw new Error('Please provide at least 2 complete rows of data with valid response values')
        }

        payload = {
          data: data,
          treatment: 'treatment',
          block: 'block',
          response: responseName,
          alpha: alpha,
          random_blocks: randomBlocks
        }
        endpoint = `${API_URL}/api/block-designs/rcbd`

      } else if (designType === 'latin') {
        const data = tableData
          .map((row) => ({
            row: parseInt(row[0]),
            column: parseInt(row[1]),
            treatment: String(row[2] || '').trim(),
            [responseName]: parseFloat(row[3])
          }))
          .filter(row => {
            const isValid = !isNaN(row.row) && !isNaN(row.column) && row.treatment && !isNaN(row[responseName])
            return isValid
          })

        if (data.length < 2) {
          throw new Error('Please provide at least 2 complete rows of data with valid response values')
        }

        payload = {
          data: data,
          treatment: 'treatment',
          row_block: 'row',
          col_block: 'column',
          response: responseName,
          alpha: alpha,
          random_blocks: randomBlocks
        }
        endpoint = `${API_URL}/api/block-designs/latin-square`

      } else if (designType === 'graeco') {
        const data = tableData
          .map((row) => ({
            row: parseInt(row[0]),
            column: parseInt(row[1]),
            latin_treatment: String(row[2] || '').trim(),
            greek_treatment: String(row[3] || '').trim(),
            [responseName]: parseFloat(row[4])
          }))
          .filter(row => {
            const isValid = !isNaN(row.row) && !isNaN(row.column) &&
                           row.latin_treatment && row.greek_treatment &&
                           !isNaN(row[responseName])
            return isValid
          })

        if (data.length < 2) {
          throw new Error('Please provide at least 2 complete rows of data with valid response values')
        }

        payload = {
          data: data,
          latin_treatment: 'latin_treatment',
          greek_treatment: 'greek_treatment',
          row_block: 'row',
          col_block: 'column',
          response: responseName,
          alpha: alpha
        }
        endpoint = `${API_URL}/api/block-designs/graeco-latin`
      }

      const response = await axios.post(endpoint, payload)
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  // Get column headers based on design type
  const getHeaders = () => {
    if (designType === 'rcbd') {
      return ['Run', 'Block', 'Treatment', responseName]
    } else if (designType === 'latin') {
      return ['Row', 'Column', 'Treatment', responseName]
    } else {
      return ['Row', 'Column', 'Latin', 'Greek', responseName]
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center space-x-3 mb-6">
          <Grid className="w-8 h-8 text-pink-400" />
          <h2 className="text-3xl font-bold text-gray-100">Block Designs</h2>
        </div>

        <form onSubmit={handleAnalyze} className="space-y-6">
          {/* Design Type */}
          <div>
            <label className="block text-gray-100 font-medium mb-2">Design Type</label>
            <select
              value={designType}
              onChange={(e) => setDesignType(e.target.value)}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-pink-500"
            >
              <option value="rcbd">Randomized Complete Block Design (RCBD)</option>
              <option value="latin">Latin Square Design</option>
              <option value="graeco">Graeco-Latin Square Design</option>
            </select>
            <p className="text-gray-400 text-xs mt-1">
              RCBD: 1 blocking factor • Latin: 2 blocking factors • Graeco-Latin: 2 blocking + 2 treatment factors
            </p>
          </div>

          {/* Design Parameters */}
          {designType === 'rcbd' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-100 font-medium mb-2">
                  Number of Treatments
                </label>
                <input
                  type="number"
                  min="2"
                  max="20"
                  value={nTreatments}
                  onChange={(e) => setNTreatments(parseInt(e.target.value) || 2)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-pink-500"
                />
              </div>
              <div>
                <label className="block text-gray-100 font-medium mb-2">
                  Number of Blocks
                </label>
                <input
                  type="number"
                  min="2"
                  max="20"
                  value={nBlocks}
                  onChange={(e) => setNBlocks(parseInt(e.target.value) || 2)}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-pink-500"
                />
              </div>
            </div>
          ) : (
            <div>
              <label className="block text-gray-100 font-medium mb-2">
                Square Size (n × n)
              </label>
              <input
                type="number"
                min={designType === 'graeco' ? 3 : 2}
                max="12"
                value={squareSize}
                onChange={(e) => setSquareSize(parseInt(e.target.value) || 3)}
                className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-pink-500"
              />
              <p className="text-gray-400 text-xs mt-1">
                {designType === 'graeco'
                  ? 'Graeco-Latin squares do not exist for n=2 or n=6. Size must be 3-12.'
                  : 'Latin square size (treatments = rows = columns). Total runs = n²'}
              </p>
            </div>
          )}

          {/* Response Variable Name */}
          <div>
            <label className="block text-gray-100 font-medium mb-2">
              Response Variable Name
            </label>
            <input
              type="text"
              value={responseName}
              onChange={(e) => setResponseName(e.target.value)}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-pink-500"
              placeholder="e.g., Yield, Strength, Quality"
              required
            />
          </div>

          {/* Randomization Option (RCBD only) */}
          {designType === 'rcbd' && (
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="randomize"
                checked={randomize}
                onChange={(e) => setRandomize(e.target.checked)}
                className="w-4 h-4 text-pink-600 bg-slate-700 border-slate-600 rounded focus:ring-pink-500"
              />
              <label htmlFor="randomize" className="text-gray-100 font-medium">
                Randomize Run Order
              </label>
              <Shuffle className="w-4 h-4 text-pink-400" />
            </div>
          )}

          {/* Random Blocks Option (RCBD and Latin Square) */}
          {(designType === 'rcbd' || designType === 'latin') && (
            <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-700/30">
              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id="randomBlocks"
                  checked={randomBlocks}
                  onChange={(e) => setRandomBlocks(e.target.checked)}
                  className="w-4 h-4 text-purple-600 bg-slate-700 border-slate-600 rounded focus:ring-purple-500 mt-0.5"
                />
                <div className="flex-1">
                  <label htmlFor="randomBlocks" className="text-gray-100 font-medium block mb-1">
                    Treat Blocks as Random Effects
                  </label>
                  <p className="text-gray-400 text-xs">
                    <strong>Fixed blocks:</strong> Block levels are the only ones of interest (standard ANOVA).<br />
                    <strong>Random blocks:</strong> Blocks are a random sample from a larger population (mixed model with variance components).
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Generated Design Info */}
          {generatedDesign && (
            <div className="bg-pink-900/20 rounded-lg p-4 border border-pink-700/30">
              <h4 className="text-gray-100 font-semibold mb-2">Generated Design</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-slate-800/50 rounded-lg p-2">
                  <p className="text-gray-400 text-xs">Design Type</p>
                  <p className="text-gray-100 text-sm font-semibold">{generatedDesign.design_type}</p>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-2">
                  <p className="text-gray-400 text-xs">Total Runs</p>
                  <p className="text-gray-100 text-sm font-semibold">{generatedDesign.n_runs}</p>
                </div>
                {generatedDesign.n_treatments && (
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <p className="text-gray-400 text-xs">Treatments</p>
                    <p className="text-gray-100 text-sm font-semibold">{generatedDesign.n_treatments}</p>
                  </div>
                )}
                {generatedDesign.n_blocks && (
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <p className="text-gray-400 text-xs">Blocks</p>
                    <p className="text-gray-100 text-sm font-semibold">{generatedDesign.n_blocks}</p>
                  </div>
                )}
                {generatedDesign.size && (
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <p className="text-gray-400 text-xs">Square Size</p>
                    <p className="text-gray-100 text-sm font-semibold">{generatedDesign.size}×{generatedDesign.size}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Data Table */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-gray-100 font-medium">
                Experimental Data
              </label>
              <div className="flex space-x-2">
                <button
                  type="button"
                  onClick={handleGenerateDesign}
                  className="flex items-center space-x-1 px-3 py-1 bg-pink-600 text-white rounded-lg hover:bg-pink-700 transition-colors text-sm"
                >
                  <Shuffle className="w-4 h-4" />
                  <span>Regenerate</span>
                </button>
                <button
                  type="button"
                  onClick={addRow}
                  className="flex items-center space-x-1 px-3 py-1 bg-pink-600 text-white rounded-lg hover:bg-pink-700 transition-colors text-sm"
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
                    <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 w-14 sticky left-0 bg-slate-700/70">
                      #
                    </th>
                    {getHeaders().map((header, idx) => (
                      <th
                        key={idx}
                        className={`px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-r border-slate-600 min-w-[100px] ${
                          idx === getHeaders().length - 1 ? 'bg-pink-900/20' : ''
                        }`}
                      >
                        {header}
                      </th>
                    ))}
                    <th className="px-3 py-2 text-center text-gray-100 font-semibold text-sm border-b-2 border-slate-600 w-16">

                    </th>
                  </tr>
                </thead>
                <tbody>
                  {tableData.map((row, rowIndex) => (
                    <tr
                      key={rowIndex}
                      className="border-b border-slate-700/30 hover:bg-slate-600/10"
                    >
                      <td className="px-3 py-2 text-center text-gray-300 text-sm font-medium bg-slate-700/30 border-r border-slate-600 sticky left-0">
                        {rowIndex + 1}
                      </td>
                      {row.map((cell, colIndex) => (
                        <td key={colIndex} className="px-1 py-1 border-r border-slate-700/20">
                          <input
                            id={`cell-${rowIndex}-${colIndex}`}
                            type="text"
                            value={cell}
                            onChange={(e) => handleCellChange(rowIndex, colIndex, e.target.value)}
                            onKeyDown={(e) => handleKeyDown(e, rowIndex, colIndex)}
                            onClick={handleCellClick}
                            className="w-full px-2 py-1.5 bg-slate-800/50 text-gray-100 border border-slate-600/50 focus:border-pink-500 focus:bg-slate-700/50 hover:border-slate-500 rounded-sm focus:outline-none focus:ring-2 focus:ring-pink-500/50 text-sm transition-all"
                            autoComplete="off"
                          />
                        </td>
                      ))}
                      <td className="px-2 py-2 text-center">
                        <button
                          type="button"
                          onClick={() => removeRow(rowIndex)}
                          disabled={tableData.length === 1}
                          className="p-1 text-red-400 hover:text-red-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                          title="Remove row"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="space-y-1 mt-2">
              <p className="text-gray-400 text-xs">
                {designType === 'rcbd'
                  ? `RCBD with ${nTreatments} treatments × ${nBlocks} blocks = ${nTreatments * nBlocks} runs. Data auto-generated with random responses.`
                  : designType === 'latin'
                  ? `Latin Square ${squareSize}×${squareSize} = ${squareSize * squareSize} runs. Each treatment appears once per row and column.`
                  : `Graeco-Latin Square ${squareSize}×${squareSize} = ${squareSize * squareSize} runs. Two orthogonal Latin squares superimposed.`
                }
              </p>
              <p className="text-pink-400 text-xs">
                <strong>Excel-like navigation:</strong> Use Arrow keys, Enter, Tab. Click to select all.
              </p>
            </div>
          </div>

          {/* Significance Level */}
          <div>
            <label className="block text-gray-100 font-medium mb-2">
              Significance Level (α)
            </label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-pink-500"
            />
          </div>

          {/* Analyze Button */}
          <button
            type="submit"
            disabled={loading || tableData.length === 0}
            className="w-full bg-pink-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-pink-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Analyzing...' : 'Analyze Block Design'}
          </button>
        </form>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/30 backdrop-blur-lg rounded-xl p-4 border border-red-700/50">
          <p className="text-red-200 font-medium">Error: {error}</p>
        </div>
      )}

      {/* Results Display */}
      {result && <ResultCard result={result} />}
    </div>
  )
}

export default BlockDesigns
