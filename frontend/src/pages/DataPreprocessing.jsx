import { useState } from 'react'
import DataTransformationPanel from '../components/DataTransformationPanel'
import OutlierDetection from '../components/OutlierDetection'
import { Wand2, AlertTriangle, Download, Upload, FileSpreadsheet } from 'lucide-react'
import { parseTableData } from '../utils/clipboardParser'

const DataPreprocessing = () => {
  const [rawData, setRawData] = useState([])
  const [processedData, setProcessedData] = useState([])
  const [activeMode, setActiveMode] = useState(null) // 'transform' or 'outlier'
  const [columnName, setColumnName] = useState('Response')
  const [dataInput, setDataInput] = useState('')

  // Load data from textarea
  const loadData = () => {
    const lines = dataInput.trim().split('\n')
    const values = lines
      .map(line => {
        const cleaned = line.trim().replace(/,/g, '')
        return parseFloat(cleaned)
      })
      .filter(v => !isNaN(v))

    if (values.length > 0) {
      setRawData(values)
      setProcessedData(values)
      setActiveMode(null)
    }
  }

  // Paste data from clipboard
  const handlePaste = async (e) => {
    e.preventDefault()

    try {
      const text = e.clipboardData?.getData('text') ||
                   await navigator.clipboard.readText()

      if (!text) return

      // Try to parse as table data first
      const result = parseTableData(text, {
        expectHeaders: false,
        expectNumeric: true
      })

      if (result.success && result.data.length > 0) {
        // Flatten to single column if multiple columns
        const values = result.data.flat().filter(v => !isNaN(parseFloat(v))).map(v => parseFloat(v))
        if (values.length > 0) {
          setRawData(values)
          setProcessedData(values)
          setDataInput(values.join('\n'))
          setActiveMode(null)
        }
      }
    } catch (error) {
      console.error('Paste error:', error)
    }
  }

  // Handle transformation apply
  const handleTransformApply = (transformedValues, transformInfo) => {
    setProcessedData(transformedValues)
    setActiveMode(null)
  }

  // Handle transformation reset
  const handleTransformReset = () => {
    setProcessedData(rawData)
    setActiveMode(null)
  }

  // Handle outlier removal
  const handleOutlierApply = (cleanedValues, outlierInfo) => {
    setProcessedData(cleanedValues)
    setRawData(cleanedValues) // Update raw data to cleaned version
    setActiveMode(null)
  }

  // Export data
  const exportData = () => {
    const csv = processedData.join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `preprocessed_data_${Date.now()}.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  // Copy to clipboard
  const copyToClipboard = () => {
    const text = processedData.join('\n')
    navigator.clipboard.writeText(text)
      .then(() => alert('Data copied to clipboard!'))
      .catch(err => console.error('Copy failed:', err))
  }

  // Load example data
  const loadExampleData = () => {
    const example = [
      23, 28, 25, 30, 27, 24, 29, 26, 31, 25,
      28, 24, 27, 29, 26, 30, 25, 28, 27, 24,
      150, 26, 29, 25, 28, 30, 27, 24, 26, 29  // 150 is an outlier
    ]
    setRawData(example)
    setProcessedData(example)
    setDataInput(example.join('\n'))
    setActiveMode(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-gray-100 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-600 text-transparent bg-clip-text">
            Data Preprocessing
          </h1>
          <p className="text-gray-300">
            Transform, clean, and prepare your data before analysis
          </p>
        </div>

        {/* Data Input Section */}
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <FileSpreadsheet className="w-5 h-5 text-indigo-400" />
              <h2 className="text-xl font-bold text-gray-100">Data Input</h2>
            </div>
            <div className="flex gap-2">
              <button
                onClick={loadExampleData}
                className="px-3 py-1 text-sm bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition"
              >
                Load Example
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Area */}
            <div>
              <label className="block text-gray-200 font-medium mb-2">
                Enter Data (one value per line)
              </label>
              <textarea
                value={dataInput}
                onChange={(e) => setDataInput(e.target.value)}
                onPaste={handlePaste}
                placeholder="23&#10;28&#10;25&#10;30&#10;27&#10;...&#10;&#10;Or paste from Excel (Ctrl+V / Cmd+V)"
                className="w-full h-64 px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono text-sm"
              />
              <div className="flex gap-2 mt-2">
                <button
                  onClick={loadData}
                  disabled={!dataInput.trim()}
                  className="flex-1 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg transition"
                >
                  <Upload className="w-4 h-4 inline mr-2" />
                  Load Data
                </button>
                <input
                  type="text"
                  value={columnName}
                  onChange={(e) => setColumnName(e.target.value)}
                  placeholder="Column name"
                  className="w-32 px-3 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm"
                />
              </div>
            </div>

            {/* Data Summary */}
            <div>
              <label className="block text-gray-200 font-medium mb-2">
                Current Data Summary
              </label>
              <div className="bg-slate-700/30 rounded-lg p-4 h-64 overflow-y-auto">
                {processedData.length > 0 ? (
                  <>
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      <div className="bg-slate-600/30 rounded p-2">
                        <div className="text-xs text-gray-400">Count</div>
                        <div className="text-lg font-bold">{processedData.length}</div>
                      </div>
                      <div className="bg-slate-600/30 rounded p-2">
                        <div className="text-xs text-gray-400">Mean</div>
                        <div className="text-lg font-bold">
                          {(processedData.reduce((a, b) => a + b, 0) / processedData.length).toFixed(2)}
                        </div>
                      </div>
                      <div className="bg-slate-600/30 rounded p-2">
                        <div className="text-xs text-gray-400">Min</div>
                        <div className="text-lg font-bold">{Math.min(...processedData).toFixed(2)}</div>
                      </div>
                      <div className="bg-slate-600/30 rounded p-2">
                        <div className="text-xs text-gray-400">Max</div>
                        <div className="text-lg font-bold">{Math.max(...processedData).toFixed(2)}</div>
                      </div>
                    </div>
                    <div className="text-xs text-gray-400 mb-2">Preview (first 10 values):</div>
                    <div className="font-mono text-sm text-gray-300">
                      {processedData.slice(0, 10).map((v, i) => (
                        <div key={i}>{v}</div>
                      ))}
                      {processedData.length > 10 && <div className="text-gray-500">... and {processedData.length - 10} more</div>}
                    </div>
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-500">
                    No data loaded
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        {rawData.length > 0 && !activeMode && (
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50 mb-6">
            <h2 className="text-xl font-bold text-gray-100 mb-4">Preprocessing Options</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <button
                onClick={() => setActiveMode('transform')}
                className="flex flex-col items-center justify-center p-6 bg-gradient-to-br from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 rounded-lg transition"
              >
                <Wand2 className="w-8 h-8 mb-2" />
                <span className="font-semibold">Transform Data</span>
                <span className="text-xs text-gray-200 mt-1">Log, Box-Cox, Z-Score, etc.</span>
              </button>

              <button
                onClick={() => setActiveMode('outlier')}
                className="flex flex-col items-center justify-center p-6 bg-gradient-to-br from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 rounded-lg transition"
              >
                <AlertTriangle className="w-8 h-8 mb-2" />
                <span className="font-semibold">Detect Outliers</span>
                <span className="text-xs text-gray-200 mt-1">Z-Score, IQR, ML methods</span>
              </button>

              <button
                onClick={exportData}
                disabled={processedData.length === 0}
                className="flex flex-col items-center justify-center p-6 bg-gradient-to-br from-green-600 to-teal-600 hover:from-green-700 hover:to-teal-700 disabled:from-slate-600 disabled:to-slate-700 disabled:cursor-not-allowed rounded-lg transition"
              >
                <Download className="w-8 h-8 mb-2" />
                <span className="font-semibold">Export Data</span>
                <span className="text-xs text-gray-200 mt-1">Download as CSV</span>
              </button>
            </div>

            <div className="mt-4 flex justify-center">
              <button
                onClick={copyToClipboard}
                disabled={processedData.length === 0}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:cursor-not-allowed text-gray-100 rounded-lg transition text-sm"
              >
                Copy to Clipboard
              </button>
            </div>
          </div>
        )}

        {/* Transformation Panel */}
        {activeMode === 'transform' && rawData.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-100">Data Transformation</h2>
              <button
                onClick={() => setActiveMode(null)}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-gray-100 rounded-lg transition"
              >
                Back to Options
              </button>
            </div>
            <DataTransformationPanel
              data={processedData}
              columnName={columnName}
              onApply={handleTransformApply}
              onReset={handleTransformReset}
            />
          </div>
        )}

        {/* Outlier Detection Panel */}
        {activeMode === 'outlier' && rawData.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-100">Outlier Detection</h2>
              <button
                onClick={() => setActiveMode(null)}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-gray-100 rounded-lg transition"
              >
                Back to Options
              </button>
            </div>
            <OutlierDetection
              data={processedData}
              columnName={columnName}
              onApply={handleOutlierApply}
              onCancel={() => setActiveMode(null)}
            />
          </div>
        )}
      </div>
    </div>
  )
}

export default DataPreprocessing
