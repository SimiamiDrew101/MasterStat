import jsPDF from 'jspdf'
import 'jspdf-autotable'
import * as XLSX from 'xlsx'

// ==================== PDF EXPORT ====================

export const generatePDFReport = (design, wizardData) => {
  const doc = new jsPDF()
  const factorNames = design.factorNames || []
  const designMatrix = design.design_matrix || []
  const factorLevels = design.factorLevels || []
  const useActualValues = design.useActualValues || false

  let yPosition = 20

  // ===== COVER PAGE =====
  doc.setFontSize(24)
  doc.setFont('helvetica', 'bold')
  doc.text('Experimental Design Report', 105, yPosition, { align: 'center' })

  yPosition += 15
  doc.setFontSize(12)
  doc.setFont('helvetica', 'normal')
  doc.text(`Generated on ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}`, 105, yPosition, { align: 'center' })

  yPosition += 20
  doc.setFontSize(10)
  doc.setTextColor(100)
  doc.text('Created with MasterStat Experiment Wizard', 105, yPosition, { align: 'center' })

  // ===== EXPERIMENT SUMMARY =====
  yPosition += 25
  doc.setFontSize(16)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(0)
  doc.text('Experiment Summary', 20, yPosition)

  yPosition += 10
  doc.setFontSize(11)
  doc.setFont('helvetica', 'normal')

  const summaryData = [
    ['Objective', capitalize(wizardData.goal || 'Not specified')],
    ['Design Type', wizardData.selectedDesign?.type || 'Not specified'],
    ['Number of Factors', factorNames.length.toString()],
    ['Total Runs', designMatrix.length.toString()],
    ['Value Type', useActualValues ? 'Actual Values' : 'Coded (-1, 0, +1)'],
  ]

  if (wizardData.budget) {
    summaryData.push(['Budget Constraint', `${wizardData.budget} runs`])
  }

  if (wizardData.powerAnalysis?.minimumRuns) {
    summaryData.push([
      'Power Analysis',
      `${wizardData.powerAnalysis.minimumRuns} runs recommended (${capitalize(wizardData.powerAnalysis.effectSize)} effect, ${(wizardData.powerAnalysis.desiredPower * 100).toFixed(0)}% power)`
    ])
  }

  doc.autoTable({
    startY: yPosition,
    head: [['Parameter', 'Value']],
    body: summaryData,
    theme: 'striped',
    headStyles: { fillColor: [59, 130, 246], fontSize: 11, fontStyle: 'bold' },
    styles: { fontSize: 10 },
    margin: { left: 20, right: 20 }
  })

  yPosition = doc.lastAutoTable.finalY + 15

  // ===== FACTOR INFORMATION =====
  if (yPosition > 240) {
    doc.addPage()
    yPosition = 20
  }

  doc.setFontSize(16)
  doc.setFont('helvetica', 'bold')
  doc.text('Factor Information', 20, yPosition)

  yPosition += 10
  const factorData = factorNames.map((name, idx) => {
    const level = factorLevels[idx]
    if (level && level.min && level.max) {
      const units = level.units ? ` ${level.units}` : ''
      return [
        (idx + 1).toString(),
        name,
        `${level.min}${units}`,
        `${level.max}${units}`,
        level.units || '-'
      ]
    }
    return [(idx + 1).toString(), name, 'Coded', 'Coded', '-']
  })

  doc.autoTable({
    startY: yPosition,
    head: [['#', 'Factor Name', 'Low Level (-1)', 'High Level (+1)', 'Units']],
    body: factorData,
    theme: 'grid',
    headStyles: { fillColor: [59, 130, 246], fontSize: 10, fontStyle: 'bold' },
    styles: { fontSize: 9 },
    margin: { left: 20, right: 20 }
  })

  yPosition = doc.lastAutoTable.finalY + 15

  // ===== DESIGN RATIONALE =====
  if (yPosition > 240) {
    doc.addPage()
    yPosition = 20
  }

  doc.setFontSize(16)
  doc.setFont('helvetica', 'bold')
  doc.text('Design Rationale', 20, yPosition)

  yPosition += 10
  doc.setFontSize(10)
  doc.setFont('helvetica', 'normal')

  if (wizardData.selectedDesign) {
    const design = wizardData.selectedDesign
    const wrappedDescription = doc.splitTextToSize(design.description || '', 170)
    doc.text(wrappedDescription, 20, yPosition)
    yPosition += wrappedDescription.length * 5 + 10

    if (design.pros && design.pros.length > 0) {
      doc.setFont('helvetica', 'bold')
      doc.text('Advantages:', 20, yPosition)
      yPosition += 6
      doc.setFont('helvetica', 'normal')
      design.pros.forEach(pro => {
        const wrapped = doc.splitTextToSize(`• ${pro}`, 165)
        doc.text(wrapped, 25, yPosition)
        yPosition += wrapped.length * 5
      })
      yPosition += 5
    }

    if (design.cons && design.cons.length > 0) {
      if (yPosition > 250) {
        doc.addPage()
        yPosition = 20
      }
      doc.setFont('helvetica', 'bold')
      doc.text('Considerations:', 20, yPosition)
      yPosition += 6
      doc.setFont('helvetica', 'normal')
      design.cons.forEach(con => {
        const wrapped = doc.splitTextToSize(`• ${con}`, 165)
        doc.text(wrapped, 25, yPosition)
        yPosition += wrapped.length * 5
      })
    }
  }

  // ===== DESIGN MATRIX =====
  doc.addPage()
  yPosition = 20

  doc.setFontSize(16)
  doc.setFont('helvetica', 'bold')
  doc.text('Experimental Design Matrix', 20, yPosition)

  yPosition += 5
  doc.setFontSize(9)
  doc.setFont('helvetica', 'italic')
  doc.setTextColor(100)
  doc.text('Run these experiments in the order shown and record your response values', 20, yPosition)

  yPosition += 10
  doc.setTextColor(0)

  const matrixHeaders = ['Run', ...factorNames.map((name, idx) => {
    const level = factorLevels[idx]
    const units = level && level.units ? ` (${level.units})` : ''
    return `${name}${units}`
  }), 'Response']

  const matrixData = designMatrix.map((row, idx) => [
    (idx + 1).toString(),
    ...factorNames.map(name => {
      const value = row[name]
      return typeof value === 'number' ? value.toFixed(3) : value
    }),
    ''
  ])

  doc.autoTable({
    startY: yPosition,
    head: [matrixHeaders],
    body: matrixData,
    theme: 'grid',
    headStyles: { fillColor: [59, 130, 246], fontSize: 9, fontStyle: 'bold' },
    styles: { fontSize: 8, cellPadding: 2 },
    columnStyles: {
      0: { fontStyle: 'bold', halign: 'center' }
    },
    margin: { left: 20, right: 20 }
  })

  // ===== INSTRUCTIONS =====
  doc.addPage()
  yPosition = 20

  doc.setFontSize(16)
  doc.setFont('helvetica', 'bold')
  doc.text('Experimental Protocol', 20, yPosition)

  yPosition += 15
  doc.setFontSize(11)
  doc.setFont('helvetica', 'bold')
  doc.text('Setup Instructions:', 20, yPosition)

  yPosition += 8
  doc.setFontSize(10)
  doc.setFont('helvetica', 'normal')

  const instructions = [
    '1. Review all factor settings and ensure equipment/materials are available',
    '2. Calibrate all measurement instruments before beginning',
    '3. CRITICAL: Randomize the run order to prevent systematic bias',
    '   (Use the "Randomize Order" button in the wizard before exporting)',
    '4. Record the actual sequence of runs performed',
    '5. For each run:',
    '   a. Set factors to the specified levels',
    '   b. Allow system to stabilize (if applicable)',
    '   c. Conduct experiment and measure response',
    '   d. Record response value with appropriate precision',
    '   e. Document any anomalies or deviations',
    '6. Complete all runs before analysis to avoid bias'
  ]

  instructions.forEach(instruction => {
    const wrapped = doc.splitTextToSize(instruction, 165)
    doc.text(wrapped, 20, yPosition)
    yPosition += wrapped.length * 5 + 2
  })

  yPosition += 10
  doc.setFont('helvetica', 'bold')
  doc.text('Data Analysis:', 20, yPosition)

  yPosition += 8
  doc.setFont('helvetica', 'normal')

  const analysisSteps = [
    '1. Import completed design with responses to MasterStat',
    '2. Navigate to Response Surface Methodology (RSM) page',
    '3. Upload your CSV file with recorded responses',
    '4. Review model diagnostics and residual plots',
    '5. Optimize factor settings based on fitted model',
    '6. Validate optimal conditions with confirmation runs'
  ]

  analysisSteps.forEach(step => {
    const wrapped = doc.splitTextToSize(step, 165)
    doc.text(wrapped, 20, yPosition)
    yPosition += wrapped.length * 5 + 2
  })

  // ===== FOOTER ON ALL PAGES =====
  const pageCount = doc.internal.getNumberOfPages()
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i)
    doc.setFontSize(8)
    doc.setTextColor(150)
    doc.text(
      `Page ${i} of ${pageCount}`,
      doc.internal.pageSize.getWidth() / 2,
      doc.internal.pageSize.getHeight() - 10,
      { align: 'center' }
    )
    doc.text(
      'MasterStat Experiment Wizard',
      20,
      doc.internal.pageSize.getHeight() - 10
    )
  }

  return doc
}

// ==================== EXCEL EXPORT ====================

export const generateExcelWorkbook = (design, wizardData) => {
  const factorNames = design.factorNames || []
  const designMatrix = design.design_matrix || []
  const factorLevels = design.factorLevels || []
  const useActualValues = design.useActualValues || false

  const wb = XLSX.utils.book_new()

  // ===== SHEET 1: DESIGN MATRIX =====
  const matrixHeaders = ['Run', ...factorNames.map((name, idx) => {
    const level = factorLevels[idx]
    const units = level && level.units ? ` (${level.units})` : ''
    return `${name}${units}`
  }), 'Response', 'Notes']

  const matrixData = [
    matrixHeaders,
    ...designMatrix.map((row, idx) => [
      idx + 1,
      ...factorNames.map(name => {
        const value = row[name]
        return typeof value === 'number' ? Number(value.toFixed(3)) : value
      }),
      '', // Response column
      ''  // Notes column
    ])
  ]

  const ws1 = XLSX.utils.aoa_to_sheet(matrixData)

  // Set column widths
  ws1['!cols'] = [
    { wch: 6 },  // Run
    ...factorNames.map(() => ({ wch: 12 })),
    { wch: 12 }, // Response
    { wch: 30 }  // Notes
  ]

  // Style header row
  const headerRange = XLSX.utils.decode_range(ws1['!ref'])
  for (let C = headerRange.s.c; C <= headerRange.e.c; ++C) {
    const cellAddress = XLSX.utils.encode_cell({ r: 0, c: C })
    if (!ws1[cellAddress]) continue
    ws1[cellAddress].s = {
      font: { bold: true, color: { rgb: 'FFFFFF' } },
      fill: { fgColor: { rgb: '3B82F6' } },
      alignment: { horizontal: 'center', vertical: 'center' }
    }
  }

  XLSX.utils.book_append_sheet(wb, ws1, 'Design Matrix')

  // ===== SHEET 2: SUMMARY =====
  const summaryData = [
    ['Experiment Summary', ''],
    ['', ''],
    ['Parameter', 'Value'],
    ['Generated', new Date().toLocaleString()],
    ['Objective', capitalize(wizardData.goal || 'Not specified')],
    ['Design Type', wizardData.selectedDesign?.type || ''],
    ['Number of Factors', factorNames.length],
    ['Total Runs', designMatrix.length],
    ['Value Type', useActualValues ? 'Actual Values' : 'Coded'],
  ]

  if (wizardData.budget) {
    summaryData.push(['Budget Constraint', `${wizardData.budget} runs`])
  }

  if (wizardData.powerAnalysis?.minimumRuns) {
    summaryData.push(
      ['', ''],
      ['Power Analysis', ''],
      ['Effect Size', capitalize(wizardData.powerAnalysis.effectSize)],
      ['Statistical Power', `${(wizardData.powerAnalysis.desiredPower * 100).toFixed(0)}%`],
      ['Recommended Runs', wizardData.powerAnalysis.minimumRuns]
    )
  }

  summaryData.push(
    ['', ''],
    ['Factor Information', ''],
    ['Factor', 'Low Level', 'High Level', 'Units']
  )

  factorNames.forEach((name, idx) => {
    const level = factorLevels[idx]
    if (level && level.min && level.max) {
      summaryData.push([name, level.min, level.max, level.units || '-'])
    } else {
      summaryData.push([name, 'Coded (-1)', 'Coded (+1)', '-'])
    }
  })

  const ws2 = XLSX.utils.aoa_to_sheet(summaryData)
  ws2['!cols'] = [{ wch: 25 }, { wch: 20 }, { wch: 20 }, { wch: 15 }]

  XLSX.utils.book_append_sheet(wb, ws2, 'Summary')

  // ===== SHEET 3: INSTRUCTIONS =====
  const instructionsData = [
    ['Experimental Protocol'],
    [''],
    ['SETUP INSTRUCTIONS:'],
    ['1. Review all factor settings and ensure equipment/materials are available'],
    ['2. Calibrate all measurement instruments before beginning'],
    ['3. CRITICAL: Randomize the run order to prevent systematic bias'],
    ['4. Record the actual sequence of runs performed'],
    [''],
    ['FOR EACH RUN:'],
    ['a. Set factors to the specified levels'],
    ['b. Allow system to stabilize (if applicable)'],
    ['c. Conduct experiment and measure response'],
    ['d. Record response value with appropriate precision'],
    ['e. Document any anomalies or deviations in Notes column'],
    [''],
    ['6. Complete all runs before analysis to avoid bias'],
    [''],
    ['DATA ANALYSIS:'],
    ['1. Import completed design with responses to MasterStat'],
    ['2. Navigate to Response Surface Methodology (RSM) page'],
    ['3. Upload your CSV file with recorded responses'],
    ['4. Review model diagnostics and residual plots'],
    ['5. Optimize factor settings based on fitted model'],
    ['6. Validate optimal conditions with confirmation runs']
  ]

  const ws3 = XLSX.utils.aoa_to_sheet(instructionsData)
  ws3['!cols'] = [{ wch: 80 }]

  XLSX.utils.book_append_sheet(wb, ws3, 'Instructions')

  return wb
}

// ==================== JMP FORMAT EXPORT ====================

export const generateJMPFormat = (design, wizardData) => {
  const factorNames = design.factorNames || []
  const designMatrix = design.design_matrix || []
  const factorLevels = design.factorLevels || []

  // JMP uses tab-delimited format with specific headers
  const headers = ['Pattern', 'Run', ...factorNames, 'Y']
  let jmpContent = headers.join('\t') + '\n'

  designMatrix.forEach((row, idx) => {
    const pattern = idx + 1
    const runNumber = idx + 1
    const values = factorNames.map(name => {
      const value = row[name]
      return typeof value === 'number' ? value.toFixed(6) : value
    })
    jmpContent += [pattern, runNumber, ...values, ''].join('\t') + '\n'
  })

  return jmpContent
}

// ==================== MINITAB FORMAT EXPORT ====================

export const generateMinitabFormat = (design, wizardData) => {
  const factorNames = design.factorNames || []
  const designMatrix = design.design_matrix || []

  // Minitab uses CSV with StdOrder and RunOrder columns
  const headers = ['StdOrder', 'RunOrder', ...factorNames, 'Response']
  let minitabContent = headers.join(',') + '\n'

  designMatrix.forEach((row, idx) => {
    const stdOrder = idx + 1
    const runOrder = idx + 1
    const values = factorNames.map(name => {
      const value = row[name]
      return typeof value === 'number' ? value.toFixed(6) : value
    })
    minitabContent += [stdOrder, runOrder, ...values, ''].join(',') + '\n'
  })

  return minitabContent
}

// ==================== ENHANCED CSV EXPORT ====================

export const generateEnhancedCSV = (design, wizardData, includeMetadata = true) => {
  const factorNames = design.factorNames || []
  const designMatrix = design.design_matrix || []
  const factorLevels = design.factorLevels || []

  let csv = ''

  if (includeMetadata) {
    // Add metadata as comments
    csv += `# Experimental Design Generated by MasterStat\n`
    csv += `# Date: ${new Date().toLocaleString()}\n`
    csv += `# Design Type: ${wizardData.selectedDesign?.type || 'Not specified'}\n`
    csv += `# Objective: ${capitalize(wizardData.goal || 'Not specified')}\n`
    csv += `# Factors: ${factorNames.length}\n`
    csv += `# Runs: ${designMatrix.length}\n`
    if (wizardData.powerAnalysis?.minimumRuns) {
      csv += `# Power Analysis: ${wizardData.powerAnalysis.minimumRuns} runs recommended (${capitalize(wizardData.powerAnalysis.effectSize)} effect, ${(wizardData.powerAnalysis.desiredPower * 100).toFixed(0)}% power)\n`
    }
    csv += `#\n`
  }

  // Headers
  const headers = ['Run', ...factorNames.map((name, idx) => {
    const level = factorLevels[idx]
    const units = level && level.units ? ` (${level.units})` : ''
    return `${name}${units}`
  }), 'Response', 'Notes']

  csv += headers.join(',') + '\n'

  // Data
  designMatrix.forEach((row, idx) => {
    const values = factorNames.map(name => {
      const value = row[name]
      return typeof value === 'number' ? value.toFixed(3) : value
    })
    csv += `${idx + 1},${values.join(',')}, ,\n`
  })

  return csv
}

// ==================== HELPER FUNCTIONS ====================

function capitalize(str) {
  if (!str) return ''
  return str.charAt(0).toUpperCase() + str.slice(1)
}

// ==================== DOWNLOAD FUNCTIONS ====================

export const downloadFile = (content, filename, mimeType) => {
  const blob = new Blob([content], { type: mimeType })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  window.URL.revokeObjectURL(url)
}

export const downloadPDF = (design, wizardData) => {
  const doc = generatePDFReport(design, wizardData)
  doc.save('experiment-design-report.pdf')
}

export const downloadExcel = (design, wizardData) => {
  const wb = generateExcelWorkbook(design, wizardData)
  XLSX.writeFile(wb, 'experiment-design.xlsx')
}

export const downloadJMP = (design, wizardData) => {
  const content = generateJMPFormat(design, wizardData)
  downloadFile(content, 'experiment-design.jmp.txt', 'text/plain')
}

export const downloadMinitab = (design, wizardData) => {
  const content = generateMinitabFormat(design, wizardData)
  downloadFile(content, 'experiment-design-minitab.csv', 'text/csv')
}

export const downloadCSV = (design, wizardData, enhanced = true) => {
  const content = generateEnhancedCSV(design, wizardData, enhanced)
  downloadFile(content, 'experiment-design.csv', 'text/csv')
}
