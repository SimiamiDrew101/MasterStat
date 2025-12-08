/**
 * Clipboard Parser Utility
 * Handles parsing clipboard data from Excel, Google Sheets, and other spreadsheet applications
 * Supports both tab-separated (TSV) and comma-separated (CSV) formats
 */

/**
 * Parse clipboard text into a 2D array
 * @param {string} text - Raw clipboard text
 * @returns {Array<Array<string>>} - Parsed 2D array
 */
export const parseClipboardData = (text) => {
  if (!text || text.trim() === '') {
    return []
  }

  // Split by newlines (handle both \n and \r\n)
  const lines = text.split(/\r?\n/).filter(line => line.trim() !== '')

  if (lines.length === 0) {
    return []
  }

  // Detect delimiter (tab or comma)
  const delimiter = detectDelimiter(lines[0])

  // Parse each line
  const parsed = lines.map(line => {
    return line.split(delimiter).map(cell => cell.trim())
  })

  return parsed
}

/**
 * Detect delimiter in a line (tab or comma)
 * @param {string} line - First line of data
 * @returns {string} - Detected delimiter
 */
const detectDelimiter = (line) => {
  const tabCount = (line.match(/\t/g) || []).length
  const commaCount = (line.match(/,/g) || []).length

  // Prefer tab (Excel default) over comma
  return tabCount > 0 ? '\t' : ','
}

/**
 * Detect if first row contains headers
 * @param {Array<Array<string>>} data - Parsed data
 * @returns {boolean} - True if first row is likely headers
 */
export const hasHeaders = (data) => {
  if (data.length < 2) {
    return false
  }

  const firstRow = data[0]
  const secondRow = data[1]

  // If first row has non-numeric values and second row has numeric values, likely headers
  const firstRowNumeric = firstRow.every(cell => !isNaN(parseFloat(cell)))
  const secondRowNumeric = secondRow.every(cell => !isNaN(parseFloat(cell)))

  return !firstRowNumeric && secondRowNumeric
}

/**
 * Validate data types in a column
 * @param {Array<string>} column - Column data
 * @returns {Object} - Validation result with type and isValid
 */
export const validateColumnType = (column) => {
  const numericCount = column.filter(cell => {
    const num = parseFloat(cell)
    return !isNaN(num) && cell !== ''
  }).length

  const totalNonEmpty = column.filter(cell => cell !== '').length

  if (totalNonEmpty === 0) {
    return { type: 'empty', isValid: false }
  }

  const numericRatio = numericCount / totalNonEmpty

  if (numericRatio > 0.8) {
    return { type: 'numeric', isValid: true }
  } else if (numericRatio > 0) {
    return { type: 'mixed', isValid: false }
  } else {
    return { type: 'categorical', isValid: true }
  }
}

/**
 * Parse clipboard data into table format with validation
 * @param {string} text - Raw clipboard text
 * @param {Object} options - Parsing options
 * @param {boolean} options.expectHeaders - Whether to expect headers
 * @param {boolean} options.expectNumeric - Whether to expect all numeric data
 * @returns {Object} - Parsed data with validation info
 */
export const parseTableData = (text, options = {}) => {
  const {
    expectHeaders = true,
    expectNumeric = true
  } = options

  const rawData = parseClipboardData(text)

  if (rawData.length === 0) {
    return {
      success: false,
      error: 'No data found in clipboard'
    }
  }

  // Check for consistent column count
  const columnCounts = rawData.map(row => row.length)
  const maxColumns = Math.max(...columnCounts)
  const minColumns = Math.min(...columnCounts)

  if (maxColumns !== minColumns) {
    return {
      success: false,
      error: `Inconsistent column count: rows have between ${minColumns} and ${maxColumns} columns`
    }
  }

  // Detect headers
  const hasHeaderRow = expectHeaders && hasHeaders(rawData)
  const headers = hasHeaderRow ? rawData[0] : rawData[0].map((_, i) => `Column ${i + 1}`)
  const dataRows = hasHeaderRow ? rawData.slice(1) : rawData

  if (dataRows.length === 0) {
    return {
      success: false,
      error: 'No data rows found (only headers)'
    }
  }

  // Validate columns
  const columns = []
  for (let i = 0; i < maxColumns; i++) {
    const columnData = dataRows.map(row => row[i] || '')
    const validation = validateColumnType(columnData)

    columns.push({
      name: headers[i],
      data: columnData,
      type: validation.type,
      isValid: validation.isValid
    })
  }

  // Check if all columns are valid
  const invalidColumns = columns.filter(col => !col.isValid)

  if (expectNumeric && invalidColumns.length > 0) {
    return {
      success: false,
      error: `Invalid data in columns: ${invalidColumns.map(col => col.name).join(', ')}`,
      columns
    }
  }

  return {
    success: true,
    headers,
    data: dataRows,
    columns,
    rowCount: dataRows.length,
    columnCount: maxColumns
  }
}

/**
 * Convert parsed data to table format expected by components
 * @param {Array<Array<string>>} data - Parsed data
 * @param {Array<string>} headers - Column headers
 * @returns {Array<Object>} - Array of row objects
 */
export const convertToTableFormat = (data, headers) => {
  return data.map((row, index) => {
    const rowObj = { id: index + 1 }
    headers.forEach((header, i) => {
      const value = row[i]
      // Try to convert to number if possible
      const num = parseFloat(value)
      rowObj[header] = !isNaN(num) ? num : value
    })
    return rowObj
  })
}

/**
 * Handle paste event and parse clipboard data
 * @param {ClipboardEvent} event - Paste event
 * @param {Object} options - Parsing options
 * @returns {Promise<Object>} - Parsed data result
 */
export const handlePasteEvent = async (event, options = {}) => {
  event.preventDefault()

  try {
    // Get clipboard text
    const text = event.clipboardData?.getData('text') ||
                 await navigator.clipboard.readText()

    if (!text) {
      return {
        success: false,
        error: 'No text data found in clipboard'
      }
    }

    // Parse the data
    const result = parseTableData(text, options)

    return result
  } catch (error) {
    return {
      success: false,
      error: `Failed to read clipboard: ${error.message}`
    }
  }
}

/**
 * Create a paste handler function for table components
 * @param {Function} onDataParsed - Callback when data is successfully parsed
 * @param {Object} options - Parsing options
 * @returns {Function} - Paste event handler
 */
export const createPasteHandler = (onDataParsed, options = {}) => {
  return async (event) => {
    const result = await handlePasteEvent(event, options)

    if (result.success) {
      onDataParsed(result)
    } else {
      console.error('Clipboard paste error:', result.error)
      // You could also trigger a toast notification here
      if (options.onError) {
        options.onError(result.error)
      }
    }
  }
}

export default {
  parseClipboardData,
  parseTableData,
  hasHeaders,
  validateColumnType,
  convertToTableFormat,
  handlePasteEvent,
  createPasteHandler
}
