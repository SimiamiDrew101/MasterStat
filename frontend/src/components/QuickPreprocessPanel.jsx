import { useState } from 'react'
import { Wand2, AlertTriangle, ChevronDown, ChevronUp, Copy } from 'lucide-react'
import DataTransformationPanel from './DataTransformationPanel'
import OutlierDetection from './OutlierDetection'

const QuickPreprocessPanel = ({ className = '' }) => {
  const [isExpanded, setIsExpanded] = useState(false)
  const [activeMode, setActiveMode] = useState(null) // 'transform' or 'outlier'
  const [rawData, setRawData] = useState([])
  const [processedData, setProcessedData] = useState([])
  const [columnName, setColumnName] = useState('Column')
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

  // Copy to clipboard
  const copyToClipboard = () => {
    const text = processedData.join('\n')
    navigator.clipboard.writeText(text)
      .then(() => alert('Processed data copied to clipboard! Paste it back into your table.'))
      .catch(err => console.error('Copy failed:', err))
  }

  return (
    <div className={`bg-slate-800/30 backdrop-blur-lg rounded-2xl border border-slate-700/50 ${className}`}>
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-slate-700/30 transition rounded-t-2xl"
      >
        <div className="flex items-center gap-2">
          <Wand2 className="w-5 h-5 text-purple-400" />
          <h3 className="text-lg font-bold text-gray-100">Quick Data Preprocessing</h3>
          <span className="text-xs text-gray-400">
            (Transform or clean data before analysis)
          </span>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        )}
      </button>

      {/* Content */}
      {isExpanded && (
        <div className="p-4 pt-0 space-y-4">
          {/* Instructions */}
          <div className="bg-indigo-900/20 border border-indigo-700/50 rounded-lg p-3">
            <p className="text-sm text-indigo-200">
              <strong>How to use:</strong> Copy a column from your data table, paste it below, apply transformations or outlier detection, then copy the result back to your table.
            </p>
          </div>

          {/* Data Input */}
          {!activeMode && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Input Area */}
              <div>
                <label className="block text-gray-200 font-medium mb-2">
                  Paste Column Data (one value per line)
                </label>
                <textarea
                  value={dataInput}
                  onChange={(e) => setDataInput(e.target.value)}
                  placeholder="23&#10;28&#10;25&#10;30&#10;27&#10;...&#10;&#10;Paste column data here (Ctrl+V / Cmd+V)"
                  className="w-full h-48 px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500 font-mono text-sm"
                />
                <div className="flex gap-2 mt-2">
                  <input
                    type="text"
                    value={columnName}
                    onChange={(e) => setColumnName(e.target.value)}
                    placeholder="Column name"
                    className="flex-1 px-3 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
                  />
                  <button
                    onClick={loadData}
                    disabled={!dataInput.trim()}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg transition"
                  >
                    Load Data
                  </button>
                </div>
              </div>

              {/* Data Summary */}
              <div>
                <label className="block text-gray-200 font-medium mb-2">
                  Processed Data ({processedData.length} values)
                </label>
                <div className="bg-slate-700/30 rounded-lg p-4 h-48 overflow-y-auto">
                  {processedData.length > 0 ? (
                    <div className="font-mono text-sm text-gray-300">
                      {processedData.slice(0, 20).map((v, i) => (
                        <div key={i}>{typeof v === 'number' ? v.toFixed(4) : v}</div>
                      ))}
                      {processedData.length > 20 && (
                        <div className="text-gray-500 mt-2">... and {processedData.length - 20} more</div>
                      )}
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-500">
                      No data loaded
                    </div>
                  )}
                </div>
                <button
                  onClick={copyToClipboard}
                  disabled={processedData.length === 0}
                  className="w-full mt-2 flex items-center justify-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg transition"
                >
                  <Copy className="w-4 h-4" />
                  Copy Processed Data
                </button>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          {!activeMode && rawData.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <button
                onClick={() => setActiveMode('transform')}
                className="flex flex-col items-center justify-center p-4 bg-gradient-to-br from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 rounded-lg transition"
              >
                <Wand2 className="w-6 h-6 mb-2" />
                <span className="font-semibold">Transform Data</span>
                <span className="text-xs text-gray-200 mt-1">Log, Box-Cox, Z-Score, etc.</span>
              </button>

              <button
                onClick={() => setActiveMode('outlier')}
                className="flex flex-col items-center justify-center p-4 bg-gradient-to-br from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 rounded-lg transition"
              >
                <AlertTriangle className="w-6 h-6 mb-2" />
                <span className="font-semibold">Detect Outliers</span>
                <span className="text-xs text-gray-200 mt-1">Z-Score, IQR, ML methods</span>
              </button>
            </div>
          )}

          {/* Transformation Panel */}
          {activeMode === 'transform' && rawData.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-lg font-semibold text-gray-100">Transform: {columnName}</h4>
                <button
                  onClick={() => setActiveMode(null)}
                  className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-100 rounded-lg transition text-sm"
                >
                  Back
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
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-lg font-semibold text-gray-100">Outlier Detection: {columnName}</h4>
                <button
                  onClick={() => setActiveMode(null)}
                  className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-gray-100 rounded-lg transition text-sm"
                >
                  Back
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
      )}
    </div>
  )
}

export default QuickPreprocessPanel
