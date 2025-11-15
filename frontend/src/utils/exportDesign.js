/**
 * Export factorial design data to CSV format
 * @param {Array} tableData - The design matrix data
 * @param {Array} factors - Array of factor names
 * @param {String} responseName - Name of the response variable
 * @param {String} designType - Type of design (2k, 3k, fractional, etc.)
 */
export const exportToCSV = (tableData, factors, responseName, designType = 'factorial') => {
  if (!tableData || tableData.length === 0) {
    alert('No data to export')
    return
  }

  // Create CSV header
  const headers = [...factors, responseName]
  let csvContent = headers.join(',') + '\n'

  // Add data rows
  tableData.forEach(row => {
    const rowData = row.map(cell => {
      // Handle empty cells
      if (cell === null || cell === undefined || cell === '') {
        return ''
      }
      // Handle cells that might contain commas or quotes
      const cellStr = String(cell)
      if (cellStr.includes(',') || cellStr.includes('"') || cellStr.includes('\n')) {
        return `"${cellStr.replace(/"/g, '""')}"`
      }
      return cellStr
    })
    csvContent += rowData.join(',') + '\n'
  })

  // Create blob and download
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const link = document.createElement('a')
  const url = URL.createObjectURL(blob)

  link.setAttribute('href', url)
  link.setAttribute('download', `${designType}-design-${new Date().toISOString().split('T')[0]}.csv`)
  link.style.visibility = 'hidden'

  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

/**
 * Export factorial design with metadata to CSV
 * Includes design information as comments
 */
export const exportToCSVWithMetadata = (tableData, factors, responseName, metadata = {}) => {
  if (!tableData || tableData.length === 0) {
    alert('No data to export')
    return
  }

  let csvContent = ''

  // Add metadata as comments
  csvContent += `# Factorial Design Export\n`
  csvContent += `# Generated: ${new Date().toISOString()}\n`
  if (metadata.designType) csvContent += `# Design Type: ${metadata.designType}\n`
  if (metadata.numFactors) csvContent += `# Number of Factors: ${metadata.numFactors}\n`
  if (metadata.numRuns) csvContent += `# Number of Runs: ${metadata.numRuns}\n`
  if (metadata.fraction) csvContent += `# Fraction: ${metadata.fraction}\n`
  if (metadata.generators && metadata.generators.length > 0) {
    csvContent += `# Generators: ${metadata.generators.join(', ')}\n`
  }
  if (metadata.resolution) csvContent += `# Resolution: ${metadata.resolution}\n`
  csvContent += `#\n`

  // Create CSV header
  const headers = [...factors, responseName]
  csvContent += headers.join(',') + '\n'

  // Add data rows
  tableData.forEach(row => {
    const rowData = row.map(cell => {
      if (cell === null || cell === undefined || cell === '') {
        return ''
      }
      const cellStr = String(cell)
      if (cellStr.includes(',') || cellStr.includes('"') || cellStr.includes('\n')) {
        return `"${cellStr.replace(/"/g, '""')}"`
      }
      return cellStr
    })
    csvContent += rowData.join(',') + '\n'
  })

  // Create blob and download
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const link = document.createElement('a')
  const url = URL.createObjectURL(blob)

  const filename = metadata.designType
    ? `${metadata.designType.replace(/[^a-z0-9]/gi, '-')}-design-${new Date().toISOString().split('T')[0]}.csv`
    : `factorial-design-${new Date().toISOString().split('T')[0]}.csv`

  link.setAttribute('href', url)
  link.setAttribute('download', filename)
  link.style.visibility = 'hidden'

  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

/**
 * Copy table data to clipboard in tab-separated format (Excel-compatible)
 */
export const copyToClipboard = async (tableData, factors, responseName) => {
  if (!tableData || tableData.length === 0) {
    alert('No data to copy')
    return false
  }

  // Create TSV content (tab-separated for Excel)
  const headers = [...factors, responseName]
  let tsvContent = headers.join('\t') + '\n'

  tableData.forEach(row => {
    const rowData = row.map(cell => {
      if (cell === null || cell === undefined || cell === '') {
        return ''
      }
      return String(cell)
    })
    tsvContent += rowData.join('\t') + '\n'
  })

  try {
    await navigator.clipboard.writeText(tsvContent)
    return true
  } catch (err) {
    console.error('Failed to copy to clipboard:', err)
    return false
  }
}

/**
 * Export results to JSON format
 */
export const exportResultsToJSON = (results, filename = 'factorial-results') => {
  if (!results) {
    alert('No results to export')
    return
  }

  const jsonContent = JSON.stringify(results, null, 2)
  const blob = new Blob([jsonContent], { type: 'application/json' })
  const link = document.createElement('a')
  const url = URL.createObjectURL(blob)

  link.setAttribute('href', url)
  link.setAttribute('download', `${filename}-${new Date().toISOString().split('T')[0]}.json`)
  link.style.visibility = 'hidden'

  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}
