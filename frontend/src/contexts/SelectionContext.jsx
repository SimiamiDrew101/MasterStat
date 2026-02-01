import { createContext, useContext, useState, useCallback, useMemo } from 'react'

/**
 * SelectionContext - Global selection state for linked data exploration
 *
 * Features:
 * - Brushing: Select multiple points by range
 * - Point selection: Click individual points
 * - Row selection: Select rows in data table
 * - Cross-component synchronization
 */

const SelectionContext = createContext(null)

export const useSelection = () => {
  const context = useContext(SelectionContext)
  if (!context) {
    throw new Error('useSelection must be used within a SelectionProvider')
  }
  return context
}

export const SelectionProvider = ({ children }) => {
  // Selected indices (array of row indices)
  const [selectedIndices, setSelectedIndices] = useState([])

  // Highlighted index (for hover)
  const [highlightedIndex, setHighlightedIndex] = useState(null)

  // Brush selection (for range selection)
  const [brushSelection, setBrushSelection] = useState(null)

  // Active dataset (shared data across linked views)
  const [linkedData, setLinkedData] = useState([])
  const [linkedColumns, setLinkedColumns] = useState([])

  // Selection mode: 'point', 'brush', 'lasso'
  const [selectionMode, setSelectionMode] = useState('point')

  // Selection color
  const [selectionColor, setSelectionColor] = useState('#f59e0b')

  // Clear all selections
  const clearSelection = useCallback(() => {
    setSelectedIndices([])
    setBrushSelection(null)
  }, [])

  // Toggle selection of a single index
  const toggleIndex = useCallback((index) => {
    setSelectedIndices(prev => {
      if (prev.includes(index)) {
        return prev.filter(i => i !== index)
      } else {
        return [...prev, index]
      }
    })
  }, [])

  // Set selection to specific indices
  const selectIndices = useCallback((indices) => {
    setSelectedIndices(indices)
  }, [])

  // Add indices to selection
  const addToSelection = useCallback((indices) => {
    setSelectedIndices(prev => {
      const newSet = new Set([...prev, ...indices])
      return Array.from(newSet)
    })
  }, [])

  // Remove indices from selection
  const removeFromSelection = useCallback((indices) => {
    const toRemove = new Set(indices)
    setSelectedIndices(prev => prev.filter(i => !toRemove.has(i)))
  }, [])

  // Select by brush (x/y range)
  const selectByBrush = useCallback((xRange, yRange, xColumn, yColumn, data) => {
    if (!data || !xColumn || !yColumn) return

    const [xMin, xMax] = xRange
    const [yMin, yMax] = yRange

    const indices = []
    data.forEach((row, i) => {
      const x = row[xColumn]
      const y = row[yColumn]
      if (x >= xMin && x <= xMax && y >= yMin && y <= yMax) {
        indices.push(i)
      }
    })

    setBrushSelection({ xRange, yRange, xColumn, yColumn })
    setSelectedIndices(indices)
  }, [])

  // Select by value filter
  const selectByFilter = useCallback((column, filterFn, data) => {
    if (!data || !column) return

    const indices = []
    data.forEach((row, i) => {
      if (filterFn(row[column])) {
        indices.push(i)
      }
    })

    setSelectedIndices(indices)
  }, [])

  // Check if an index is selected
  const isSelected = useCallback((index) => {
    return selectedIndices.includes(index)
  }, [selectedIndices])

  // Get selected data rows
  const getSelectedData = useCallback(() => {
    if (!linkedData.length) return []
    return selectedIndices.map(i => linkedData[i]).filter(Boolean)
  }, [linkedData, selectedIndices])

  // Get unselected data rows
  const getUnselectedData = useCallback(() => {
    if (!linkedData.length) return linkedData
    const selectedSet = new Set(selectedIndices)
    return linkedData.filter((_, i) => !selectedSet.has(i))
  }, [linkedData, selectedIndices])

  // Compute selection statistics
  const selectionStats = useMemo(() => {
    if (!linkedData.length || !selectedIndices.length) {
      return null
    }

    const selectedData = selectedIndices.map(i => linkedData[i]).filter(Boolean)
    if (!selectedData.length) return null

    const stats = {}

    // Compute stats for each numeric column
    linkedColumns.forEach(col => {
      const values = selectedData
        .map(row => row[col])
        .filter(v => typeof v === 'number' && !isNaN(v))

      if (values.length > 0) {
        const sum = values.reduce((a, b) => a + b, 0)
        const mean = sum / values.length
        const sorted = [...values].sort((a, b) => a - b)
        const median = sorted.length % 2 === 0
          ? (sorted[sorted.length/2 - 1] + sorted[sorted.length/2]) / 2
          : sorted[Math.floor(sorted.length/2)]
        const min = Math.min(...values)
        const max = Math.max(...values)
        const variance = values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / values.length
        const std = Math.sqrt(variance)

        stats[col] = { count: values.length, sum, mean, median, min, max, std }
      }
    })

    return {
      count: selectedData.length,
      totalCount: linkedData.length,
      percentage: ((selectedData.length / linkedData.length) * 100).toFixed(1),
      columnStats: stats
    }
  }, [linkedData, linkedColumns, selectedIndices])

  const value = {
    // Selection state
    selectedIndices,
    highlightedIndex,
    brushSelection,
    selectionMode,
    selectionColor,

    // Linked data
    linkedData,
    linkedColumns,

    // Actions
    setSelectedIndices: selectIndices,
    setHighlightedIndex,
    setBrushSelection,
    setSelectionMode,
    setSelectionColor,
    setLinkedData,
    setLinkedColumns,

    // Selection methods
    clearSelection,
    toggleIndex,
    addToSelection,
    removeFromSelection,
    selectByBrush,
    selectByFilter,
    isSelected,
    getSelectedData,
    getUnselectedData,

    // Computed
    selectionStats,
    hasSelection: selectedIndices.length > 0,
    selectionCount: selectedIndices.length,
  }

  return (
    <SelectionContext.Provider value={value}>
      {children}
    </SelectionContext.Provider>
  )
}

export default SelectionContext
