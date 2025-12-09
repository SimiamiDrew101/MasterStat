import { useState, useEffect } from 'react'
import { Wand2, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react'
import DataTransformationPanel from './DataTransformationPanel'
import OutlierDetection from './OutlierDetection'

const QuickPreprocessPanel = ({
  tableData = [],
  columnNames = [],
  onDataUpdate,
  className = ''
}) => {
  const [isExpanded, setIsExpanded] = useState(false)
  const [activeMode, setActiveMode] = useState(null) // 'transform' or 'outlier'
  const [selectedColumn, setSelectedColumn] = useState(0)
  const [rawData, setRawData] = useState([])
  const [processedData, setProcessedData] = useState([])

  // Extract column data when selection changes
  useEffect(() => {
    if (tableData.length > 0 && selectedColumn >= 0 && selectedColumn < columnNames.length) {
      const columnData = tableData
        .map(row => row[selectedColumn])
        .filter(val => val !== null && val !== undefined && val !== '' && !isNaN(parseFloat(val)))
        .map(val => parseFloat(val))

      setRawData(columnData)
      setProcessedData(columnData)
      setActiveMode(null)
    }
  }, [tableData, selectedColumn, columnNames])

  // Handle transformation apply
  const handleTransformApply = (transformedValues, transformInfo) => {
    setProcessedData(transformedValues)

    // Update the table data with transformed values
    if (onDataUpdate) {
      onDataUpdate(selectedColumn, transformedValues, transformInfo)
    }

    setActiveMode(null)
  }

  // Handle transformation reset
  const handleTransformReset = () => {
    setProcessedData(rawData)

    // Reset the table data to original
    if (onDataUpdate) {
      onDataUpdate(selectedColumn, rawData, { transform: 'none' })
    }

    setActiveMode(null)
  }

  // Handle outlier removal
  const handleOutlierApply = (cleanedValues, outlierInfo) => {
    setProcessedData(cleanedValues)
    setRawData(cleanedValues) // Update raw data to cleaned version

    // Update the table data with cleaned values
    if (onDataUpdate) {
      onDataUpdate(selectedColumn, cleanedValues, outlierInfo)
    }

    setActiveMode(null)
  }

  const hasData = rawData.length > 0
  const columnName = columnNames[selectedColumn] || 'Column'

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
              <strong>How to use:</strong> Select a column from your data table below, then apply transformations or outlier detection. Changes will be applied directly to the table.
            </p>
          </div>

          {/* Column Selection */}
          {!activeMode && (
            <div className="space-y-4">
              <div>
                <label className="block text-gray-200 font-medium mb-2">
                  Select Column to Preprocess
                </label>
                <select
                  value={selectedColumn}
                  onChange={(e) => setSelectedColumn(parseInt(e.target.value))}
                  className="w-full px-4 py-2 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  disabled={columnNames.length === 0}
                >
                  {columnNames.length > 0 ? (
                    columnNames.map((name, idx) => (
                      <option key={idx} value={idx}>
                        {name} ({tableData.filter(row => row[idx] !== null && row[idx] !== undefined && row[idx] !== '' && !isNaN(parseFloat(row[idx]))).length} values)
                      </option>
                    ))
                  ) : (
                    <option value="">No columns available</option>
                  )}
                </select>
              </div>

              {/* Data Preview */}
              {hasData && (
                <div className="bg-slate-700/30 rounded-lg p-4">
                  <div className="flex justify-between items-center mb-2">
                    <label className="text-gray-200 font-medium">
                      Current Data Preview ({rawData.length} values)
                    </label>
                  </div>
                  <div className="max-h-32 overflow-y-auto">
                    <div className="font-mono text-sm text-gray-300">
                      {rawData.slice(0, 10).map((v, i) => (
                        <div key={i}>{typeof v === 'number' ? v.toFixed(4) : v}</div>
                      ))}
                      {rawData.length > 10 && (
                        <div className="text-gray-500 mt-2">... and {rawData.length - 10} more</div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {!hasData && (
                <div className="bg-amber-900/20 border border-amber-700/50 rounded-lg p-4">
                  <p className="text-amber-200 text-sm">
                    No numeric data found in the selected column. Please enter data in your table first.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Action Buttons */}
          {!activeMode && hasData && (
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
          {activeMode === 'transform' && hasData && (
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
          {activeMode === 'outlier' && hasData && (
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
