import { useState, useEffect } from 'react'
import { Plus, Trash2, Copy } from 'lucide-react'
import { parseTableData } from '../utils/clipboardParser'

/**
 * ExcelTable - Excel-like spreadsheet component with keyboard navigation
 * Supports arrow keys, Tab, Enter, and paste from Excel/clipboard
 */
const ExcelTable = ({
  data,
  columns,
  onChange,
  minRows = 10,
  maxRows = 1000,
  allowAddRows = true,
  allowDeleteRows = true,
  className = ''
}) => {
  const [tableData, setTableData] = useState(data || [])
  const [selectedCell, setSelectedCell] = useState(null)

  // Initialize table data
  useEffect(() => {
    if (data && data.length > 0) {
      setTableData(data)
    } else {
      // Initialize with empty rows
      const initialData = Array(minRows).fill(null).map(() =>
        Array(columns.length).fill('')
      )
      setTableData(initialData)
    }
  }, [data, columns.length, minRows])

  // Notify parent of changes
  useEffect(() => {
    if (onChange) {
      onChange(tableData)
    }
  }, [tableData])

  // Handle cell change
  const handleCellChange = (rowIndex, colIndex, value) => {
    const newData = [...tableData]
    newData[rowIndex][colIndex] = value
    setTableData(newData)

    // Auto-add row if typing in last row and allowAddRows is true
    if (allowAddRows && rowIndex === tableData.length - 1 && value.trim() !== '' && tableData.length < maxRows) {
      setTableData([...newData, Array(columns.length).fill('')])
    }
  }

  // Handle keyboard navigation
  const handleKeyDown = (e, rowIndex, colIndex) => {
    const numCols = columns.length
    const numRows = tableData.length

    let newRow = rowIndex
    let newCol = colIndex

    switch (e.key) {
      case 'ArrowUp':
        e.preventDefault()
        newRow = Math.max(0, rowIndex - 1)
        break
      case 'ArrowDown':
        e.preventDefault()
        newRow = Math.min(numRows - 1, rowIndex + 1)
        break
      case 'ArrowLeft':
        e.preventDefault()
        newCol = Math.max(0, colIndex - 1)
        break
      case 'ArrowRight':
        e.preventDefault()
        newCol = Math.min(numCols - 1, colIndex + 1)
        break
      case 'Enter':
        e.preventDefault()
        newRow = Math.min(numRows - 1, rowIndex + 1)
        break
      case 'Tab':
        e.preventDefault()
        if (e.shiftKey) {
          // Shift+Tab: move left or previous row
          if (colIndex > 0) {
            newCol = colIndex - 1
          } else if (rowIndex > 0) {
            newRow = rowIndex - 1
            newCol = numCols - 1
          }
        } else {
          // Tab: move right or next row
          if (colIndex < numCols - 1) {
            newCol = colIndex + 1
          } else if (rowIndex < numRows - 1) {
            newRow = rowIndex + 1
            newCol = 0
          }
        }
        break
      default:
        return
    }

    // Focus the new cell
    if (newRow !== rowIndex || newCol !== colIndex) {
      const cellId = `cell-${newRow}-${newCol}`
      const input = document.getElementById(cellId)
      if (input) {
        input.focus()
        setSelectedCell({ row: newRow, col: newCol })
      }
    }
  }

  // Handle paste from clipboard
  const handlePaste = async (e, rowIndex, colIndex) => {
    e.preventDefault()

    try {
      const text = e.clipboardData?.getData('text') || await navigator.clipboard.readText()
      if (!text) return

      // Parse the pasted data
      const result = parseTableData(text, {
        expectHeaders: false,
        expectNumeric: false
      })

      if (result.success && result.data.length > 0) {
        const newData = [...tableData]

        // Paste data starting from current cell
        result.data.forEach((pasteRow, pasteRowIdx) => {
          const targetRow = rowIndex + pasteRowIdx

          // Add more rows if needed
          while (targetRow >= newData.length && newData.length < maxRows) {
            newData.push(Array(columns.length).fill(''))
          }

          if (targetRow < newData.length) {
            pasteRow.forEach((value, pasteColIdx) => {
              const targetCol = colIndex + pasteColIdx
              if (targetCol < columns.length) {
                newData[targetRow][targetCol] = value?.toString() || ''
              }
            })
          }
        })

        setTableData(newData)
      } else {
        // Single value paste
        handleCellChange(rowIndex, colIndex, text)
      }
    } catch (error) {
      console.error('Paste error:', error)
    }
  }

  // Add new row
  const addRow = () => {
    if (tableData.length < maxRows) {
      setTableData([...tableData, Array(columns.length).fill('')])
    }
  }

  // Delete empty rows from the end
  const deleteEmptyRows = () => {
    const newData = [...tableData]
    // Keep at least minRows, delete empty rows from end
    while (newData.length > minRows) {
      const lastRow = newData[newData.length - 1]
      if (lastRow.every(cell => cell === '' || cell === null || cell === undefined)) {
        newData.pop()
      } else {
        break
      }
    }
    setTableData(newData)
  }

  // Copy all data to clipboard
  const copyToClipboard = () => {
    // Filter out completely empty rows
    const nonEmptyData = tableData.filter(row =>
      row.some(cell => cell !== '' && cell !== null && cell !== undefined)
    )

    const text = nonEmptyData.map(row => row.join('\t')).join('\n')
    navigator.clipboard.writeText(text)
      .then(() => {
        // Visual feedback could be added here
        console.log('Data copied to clipboard')
      })
      .catch(err => console.error('Copy failed:', err))
  }

  return (
    <div className={`space-y-3 ${className}`}>
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-400">
          Use arrow keys, Tab, or Enter to navigate. Paste from Excel (Ctrl+V / Cmd+V).
        </div>
        <div className="flex gap-2">
          {allowDeleteRows && (
            <button
              onClick={deleteEmptyRows}
              className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-gray-200 rounded-lg transition text-sm"
              title="Delete empty rows from end"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clean
            </button>
          )}
          <button
            onClick={copyToClipboard}
            className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-gray-200 rounded-lg transition text-sm"
            title="Copy all data to clipboard"
          >
            <Copy className="w-3.5 h-3.5" />
            Copy
          </button>
          {allowAddRows && (
            <button
              onClick={addRow}
              disabled={tableData.length >= maxRows}
              className="flex items-center gap-1 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-lg transition text-sm"
              title="Add new row"
            >
              <Plus className="w-3.5 h-3.5" />
              Add Row
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-lg border border-slate-600">
        <table className="w-full border-collapse">
          <thead>
            <tr className="bg-slate-700/50">
              {columns.map((col, colIndex) => (
                <th
                  key={colIndex}
                  className="border border-slate-600 py-2 px-3 text-gray-100 font-semibold min-w-[120px] text-left"
                >
                  {col.label || col.name || `Column ${colIndex + 1}`}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tableData.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className="hover:bg-slate-700/20"
              >
                {row.map((cell, colIndex) => (
                  <td key={colIndex} className="border border-slate-600 p-0">
                    <input
                      id={`cell-${rowIndex}-${colIndex}`}
                      type="text"
                      value={cell || ''}
                      onChange={(e) => handleCellChange(rowIndex, colIndex, e.target.value)}
                      onKeyDown={(e) => handleKeyDown(e, rowIndex, colIndex)}
                      onPaste={(e) => handlePaste(e, rowIndex, colIndex)}
                      onFocus={() => setSelectedCell({ row: rowIndex, col: colIndex })}
                      className={`w-full px-3 py-2 bg-transparent text-gray-100 focus:bg-slate-700/30 focus:outline-none ${
                        selectedCell?.row === rowIndex && selectedCell?.col === colIndex
                          ? 'ring-2 ring-indigo-500'
                          : ''
                      }`}
                      placeholder={columns[colIndex]?.placeholder || 'Value'}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Row count */}
      <div className="text-xs text-gray-500 text-right">
        {tableData.filter(row => row.some(cell => cell !== '' && cell !== null)).length} / {tableData.length} rows with data
      </div>
    </div>
  )
}

export default ExcelTable
