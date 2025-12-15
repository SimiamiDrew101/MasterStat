import { useState, useEffect } from 'react'
import DataTransformationPanel from '../components/DataTransformationPanel'
import OutlierDetection from '../components/OutlierDetection'
import ImputationSelector from '../components/ImputationSelector'
import ImputationComparison from '../components/ImputationComparison'
import ExcelTable from '../components/ExcelTable'
import { Wand2, AlertTriangle, Download, Upload, FileSpreadsheet, Database, GitCompare, Plus, Minus } from 'lucide-react'
import { parseTableData } from '../utils/clipboardParser'

const DataPreprocessing = () => {
  const [rawData, setRawData] = useState([])
  const [processedData, setProcessedData] = useState([])
  const [activeMode, setActiveMode] = useState(null) // 'transform', 'outlier', 'imputation', or 'comparison'
  const [columns, setColumns] = useState([{ name: 'Column1', label: 'Column 1' }])
  const [tableData, setTableData] = useState([])
  const [selectedColumn, setSelectedColumn] = useState(0)

  // Extract data from selected column
  const extractColumnData = (colIndex) => {
    // Find the first completely empty row
    const firstEmptyRowIndex = tableData.findIndex(row => row.every(cell => !cell || cell === ''))

    // If no empty row found, use all rows; otherwise, use rows up to first empty row
    const endIndex = firstEmptyRowIndex === -1 ? tableData.length : firstEmptyRowIndex

    return tableData
      .slice(0, endIndex)
      .map(row => {
        const value = row[colIndex]
        if (!value || value === '' || value.toLowerCase() === 'na' || value.toLowerCase() === 'null') {
          return null
        }
        const num = parseFloat(value)
        return isNaN(num) ? null : num
      })
  }

  // Load data from table
  const loadData = () => {
    const columnData = extractColumnData(selectedColumn)
    if (columnData.length > 0) {
      setRawData(columnData)
      setProcessedData(columnData)
      setActiveMode(null)
    }
  }

  // Update when table data or selected column changes
  useEffect(() => {
    const columnData = extractColumnData(selectedColumn)
    if (columnData.length > 0) {
      setRawData(columnData)
      setProcessedData(columnData)
    }
  }, [selectedColumn, tableData])

  // Handle table data changes
  const handleTableDataChange = (newTableData) => {
    setTableData(newTableData)
  }

  // Add column
  const addColumn = () => {
    const newColIndex = columns.length
    setColumns([...columns, {
      name: `Column${newColIndex + 1}`,
      label: `Column ${newColIndex + 1}`
    }])
    // Add column to existing rows
    setTableData(tableData.map(row => [...row, '']))
  }

  // Remove column
  const removeColumn = (colIndex) => {
    if (columns.length <= 1) return // Keep at least one column

    const newColumns = columns.filter((_, idx) => idx !== colIndex)
    setColumns(newColumns)
    setTableData(tableData.map(row => row.filter((_, idx) => idx !== colIndex)))

    // Adjust selected column if needed
    if (selectedColumn >= newColumns.length) {
      setSelectedColumn(Math.max(0, newColumns.length - 1))
    }
  }

  // Rename column
  const renameColumn = (colIndex, newName) => {
    const newColumns = [...columns]
    newColumns[colIndex] = {
      ...newColumns[colIndex],
      label: newName,
      name: newName.replace(/\s+/g, '')
    }
    setColumns(newColumns)
  }

  // Handle transformation apply
  const handleTransformApply = (transformedValues, transformInfo) => {
    setProcessedData(transformedValues)
    // Update table with transformed values
    updateTableColumn(selectedColumn, transformedValues)
    setActiveMode(null)
  }

  // Update table column with new values
  const updateTableColumn = (colIndex, values) => {
    const newTableData = [...tableData]
    values.forEach((value, rowIndex) => {
      if (rowIndex < newTableData.length) {
        newTableData[rowIndex][colIndex] = value?.toString() || ''
      } else {
        // Add new row if needed
        const newRow = Array(columns.length).fill('')
        newRow[colIndex] = value?.toString() || ''
        newTableData.push(newRow)
      }
    })
    setTableData(newTableData)
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
    updateTableColumn(selectedColumn, cleanedValues)
    setActiveMode(null)
  }

  // Handle imputation apply
  const handleImputationApply = (imputedValues, imputationInfo) => {
    setProcessedData(imputedValues)
    updateTableColumn(selectedColumn, imputedValues)
    setActiveMode(null)
  }

  // Handle comparison method selection
  const handleComparisonSelectMethod = (method) => {
    setActiveMode('imputation')
  }

  // Export data
  const exportData = () => {
    // Export all non-empty rows
    const nonEmptyRows = tableData.filter(row =>
      row.some(cell => cell !== '' && cell !== null && cell !== undefined)
    )

    const csv = [
      columns.map(col => col.label).join(','),
      ...nonEmptyRows.map(row => row.join(','))
    ].join('\n')

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
    const nonEmptyRows = tableData.filter(row =>
      row.some(cell => cell !== '' && cell !== null && cell !== undefined)
    )

    const text = [
      columns.map(col => col.label).join('\t'),
      ...nonEmptyRows.map(row => row.join('\t'))
    ].join('\n')

    navigator.clipboard.writeText(text)
      .then(() => alert('Data copied to clipboard!'))
      .catch(err => console.error('Copy failed:', err))
  }

  // Load example data
  const loadExampleData = () => {
    const exampleValues = [
      23, 28, 25, 30, 27, 24, 29, 26, 31, 25,
      28, 24, 27, 29, 26, 30, 25, 28, 27, 24,
      150, 26, 29, 25, 28, 30, 27, 24, 26, 29  // 150 is an outlier
    ]

    // Create table data with proper number of columns
    const example = exampleValues.map(val => {
      const row = Array(columns.length).fill('')
      row[0] = val.toString()
      return row
    })

    setTableData(example)
    setActiveMode(null)
  }

  // Load example data with missing values
  const loadExampleMissingData = () => {
    const exampleValues = [
      23, 28, 'NA', 30, 27, 24, 29, 26, 'NA', 25,
      28, 24, 27, 'NA', 26, 30, 25, 28, 27, 24,
      26, 29, 25, 28, 30, 'NA', 24, 26, 29, 27
    ]

    // Create table data with proper number of columns
    const example = exampleValues.map(val => {
      const row = Array(columns.length).fill('')
      row[0] = val.toString()
      return row
    })

    setTableData(example)
    setActiveMode(null)
  }

  // Check if data has missing values
  const hasMissingData = processedData.some(v => v === null || v === undefined || isNaN(v))

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
              <button
                onClick={loadExampleMissingData}
                className="px-3 py-1 text-sm bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition"
              >
                Example with Missing Data
              </button>
            </div>
          </div>

          {/* Column Controls */}
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <label className="text-gray-200 font-medium">Working Column:</label>
              <select
                value={selectedColumn}
                onChange={(e) => setSelectedColumn(parseInt(e.target.value))}
                className="px-3 py-1.5 rounded-lg bg-slate-700/50 text-gray-100 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                {columns.map((col, idx) => (
                  <option key={idx} value={idx}>{col.label}</option>
                ))}
              </select>
              <button
                onClick={addColumn}
                className="flex items-center gap-1 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition text-sm"
              >
                <Plus className="w-3.5 h-3.5" />
                Add Column
              </button>
              {columns.length > 1 && (
                <button
                  onClick={() => removeColumn(selectedColumn)}
                  className="flex items-center gap-1 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded-lg transition text-sm"
                >
                  <Minus className="w-3.5 h-3.5" />
                  Remove Column
                </button>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Excel Table */}
            <div className="lg:col-span-2">
              <ExcelTable
                data={tableData}
                columns={columns}
                onChange={handleTableDataChange}
                minRows={15}
                maxRows={500}
                allowAddRows={true}
                allowDeleteRows={true}
              />
            </div>

            {/* Data Summary */}
            <div>
              <label className="block text-gray-200 font-medium mb-2">
                Column Summary: {columns[selectedColumn]?.label}
              </label>
              <div className="bg-slate-700/30 rounded-lg p-4 overflow-y-auto">
                {rawData.length > 0 ? (
                  <>
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      <div className="bg-slate-600/30 rounded p-2">
                        <div className="text-xs text-gray-400">Total Rows</div>
                        <div className="text-lg font-bold">{tableData.filter(row => row.some(cell => cell !== '')).length}</div>
                      </div>
                      <div className="bg-slate-600/30 rounded p-2">
                        <div className="text-xs text-gray-400">Valid Values</div>
                        <div className="text-lg font-bold">{rawData.filter(v => v !== null && !isNaN(v)).length}</div>
                      </div>
                      <div className="bg-slate-600/30 rounded p-2">
                        <div className="text-xs text-gray-400">Mean</div>
                        <div className="text-lg font-bold">
                          {rawData.filter(v => v !== null && !isNaN(v)).length > 0
                            ? (rawData.filter(v => v !== null && !isNaN(v)).reduce((a, b) => a + b, 0) / rawData.filter(v => v !== null && !isNaN(v)).length).toFixed(2)
                            : 'N/A'}
                        </div>
                      </div>
                      <div className="bg-slate-600/30 rounded p-2">
                        <div className="text-xs text-gray-400">Missing</div>
                        <div className="text-lg font-bold text-orange-400">{rawData.filter(v => v === null || isNaN(v)).length}</div>
                      </div>
                    </div>
                    <div className="text-xs text-gray-400 mb-2">Preview (first 10 values):</div>
                    <div className="font-mono text-sm text-gray-300 space-y-1">
                      {rawData.slice(0, 10).map((v, i) => (
                        <div key={i} className={v === null || v === undefined || isNaN(v) ? 'text-orange-400' : ''}>
                          {v === null || v === undefined || isNaN(v) ? 'NA' : v}
                        </div>
                      ))}
                      {rawData.length > 10 && <div className="text-gray-500">... and {rawData.length - 10} more</div>}
                    </div>
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-500 py-12">
                    Enter data in the table to see summary
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        {tableData.filter(row => row.some(cell => cell !== '')).length > 0 && !activeMode && (
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50 mb-6">
            <h2 className="text-xl font-bold text-gray-100 mb-4">Preprocessing Options</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
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
                onClick={() => setActiveMode('imputation')}
                disabled={!hasMissingData}
                className="flex flex-col items-center justify-center p-6 bg-gradient-to-br from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 disabled:from-slate-600 disabled:to-slate-700 disabled:cursor-not-allowed rounded-lg transition"
              >
                <Database className="w-8 h-8 mb-2" />
                <span className="font-semibold">Impute Missing</span>
                <span className="text-xs text-gray-200 mt-1">Fill missing values</span>
              </button>

              <button
                onClick={() => setActiveMode('comparison')}
                disabled={!hasMissingData}
                className="flex flex-col items-center justify-center p-6 bg-gradient-to-br from-pink-600 to-purple-600 hover:from-pink-700 hover:to-purple-700 disabled:from-slate-600 disabled:to-slate-700 disabled:cursor-not-allowed rounded-lg transition"
              >
                <GitCompare className="w-8 h-8 mb-2" />
                <span className="font-semibold">Compare Methods</span>
                <span className="text-xs text-gray-200 mt-1">Compare imputation</span>
              </button>

              <button
                onClick={exportData}
                disabled={tableData.filter(row => row.some(cell => cell !== '')).length === 0}
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
                disabled={tableData.filter(row => row.some(cell => cell !== '')).length === 0}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:cursor-not-allowed text-gray-100 rounded-lg transition text-sm"
              >
                Copy to Clipboard
              </button>
            </div>
          </div>
        )}

        {/* Transformation Panel */}
        {activeMode === 'transform' && rawData.filter(v => v !== null && !isNaN(v)).length > 0 && (
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
              data={rawData.filter(v => v !== null && !isNaN(v))}
              columnName={columns[selectedColumn]?.label || 'Column'}
              onApply={handleTransformApply}
              onReset={handleTransformReset}
            />
          </div>
        )}

        {/* Outlier Detection Panel */}
        {activeMode === 'outlier' && rawData.filter(v => v !== null && !isNaN(v)).length > 0 && (
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
              data={rawData.filter(v => v !== null && !isNaN(v))}
              columnName={columns[selectedColumn]?.label || 'Column'}
              onApply={handleOutlierApply}
              onCancel={() => setActiveMode(null)}
            />
          </div>
        )}

        {/* Imputation Panel */}
        {activeMode === 'imputation' && rawData.length > 0 && hasMissingData && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-100">Missing Data Imputation</h2>
              <button
                onClick={() => setActiveMode(null)}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-gray-100 rounded-lg transition"
              >
                Back to Options
              </button>
            </div>
            <ImputationSelector
              data={rawData}
              columnName={columns[selectedColumn]?.label || 'Column'}
              onApply={handleImputationApply}
              onCancel={() => setActiveMode(null)}
            />
          </div>
        )}

        {/* Imputation Comparison Panel */}
        {activeMode === 'comparison' && rawData.length > 0 && hasMissingData && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-100">Compare Imputation Methods</h2>
              <button
                onClick={() => setActiveMode(null)}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-gray-100 rounded-lg transition"
              >
                Back to Options
              </button>
            </div>
            <ImputationComparison
              data={rawData}
              columnName={columns[selectedColumn]?.label || 'Column'}
              onSelectMethod={handleComparisonSelectMethod}
            />
          </div>
        )}
      </div>
    </div>
  )
}

export default DataPreprocessing
