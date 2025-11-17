import { Grid3X3, Info } from 'lucide-react'

/**
 * BlockStructureVisualization component displays the blocking structure visually
 * Shows color-coded treatment assignments for RCBD, Latin Square, and Graeco-Latin Square
 */
const BlockStructureVisualization = ({ designType, designTable, nBlocks, nTreatments, squareSize }) => {
  if (!designTable || designTable.length === 0) return null

  // Generate color palette for treatments
  const getColor = (index, total) => {
    const hue = (index / total) * 360
    return `hsl(${hue}, 70%, 60%)`
  }

  // Render RCBD structure
  const renderRCBD = () => {
    // Group by blocks
    const blockMap = {}
    designTable.forEach(row => {
      if (!blockMap[row.block]) {
        blockMap[row.block] = []
      }
      blockMap[row.block].push(row.treatment)
    })

    const blocks = Object.keys(blockMap).sort()
    const uniqueTreatments = [...new Set(designTable.map(r => r.treatment))].sort()

    return (
      <div>
        <div className="mb-4">
          <h5 className="text-sm font-semibold text-gray-300 mb-2">Treatment Assignment by Block</h5>
          <p className="text-xs text-gray-400">Each block contains all treatments in randomized order</p>
        </div>

        <div className="space-y-3">
          {blocks.map((block, blockIdx) => (
            <div key={block} className="bg-slate-700/30 rounded-lg p-4">
              <div className="text-sm font-semibold text-gray-200 mb-2">Block {block}</div>
              <div className="flex flex-wrap gap-2">
                {blockMap[block].map((treatment, treatmentIdx) => {
                  const treatmentIndex = uniqueTreatments.indexOf(treatment)
                  const color = getColor(treatmentIndex, uniqueTreatments.length)
                  return (
                    <div
                      key={`${block}-${treatmentIdx}`}
                      className="px-4 py-2 rounded-lg font-semibold text-white shadow-lg"
                      style={{ backgroundColor: color }}
                    >
                      {treatment}
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Treatment Legend */}
        <div className="mt-4 bg-slate-800/30 rounded-lg p-3">
          <div className="text-xs font-semibold text-gray-300 mb-2">Treatment Key</div>
          <div className="flex flex-wrap gap-2">
            {uniqueTreatments.map((treatment, idx) => (
              <div key={treatment} className="flex items-center gap-2">
                <div
                  className="w-4 h-4 rounded"
                  style={{ backgroundColor: getColor(idx, uniqueTreatments.length) }}
                ></div>
                <span className="text-xs text-gray-300">{treatment}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  // Render Latin Square structure
  const renderLatinSquare = () => {
    // Create grid structure
    const grid = Array(squareSize).fill(null).map(() => Array(squareSize).fill(null))

    designTable.forEach(row => {
      grid[row.row - 1][row.column - 1] = row.treatment
    })

    const uniqueTreatments = [...new Set(designTable.map(r => r.treatment))].sort()

    return (
      <div>
        <div className="mb-4">
          <h5 className="text-sm font-semibold text-gray-300 mb-2">
            Latin Square {squareSize}×{squareSize} Layout
          </h5>
          <p className="text-xs text-gray-400">
            Each treatment appears exactly once in each row and column
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="bg-slate-700/50 border border-slate-600 px-3 py-2 text-xs text-gray-400">
                  Row / Col
                </th>
                {Array(squareSize).fill(null).map((_, colIdx) => (
                  <th
                    key={colIdx}
                    className="bg-slate-700/50 border border-slate-600 px-3 py-2 text-xs font-semibold text-gray-300"
                  >
                    Col {colIdx + 1}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {grid.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  <td className="bg-slate-700/50 border border-slate-600 px-3 py-2 text-xs font-semibold text-gray-300">
                    Row {rowIdx + 1}
                  </td>
                  {row.map((treatment, colIdx) => {
                    const treatmentIndex = uniqueTreatments.indexOf(treatment)
                    const color = getColor(treatmentIndex, uniqueTreatments.length)
                    return (
                      <td
                        key={colIdx}
                        className="border border-slate-600 px-3 py-3 text-center text-white font-semibold"
                        style={{ backgroundColor: color }}
                      >
                        {treatment}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Treatment Legend */}
        <div className="mt-4 bg-slate-800/30 rounded-lg p-3">
          <div className="text-xs font-semibold text-gray-300 mb-2">Treatment Key</div>
          <div className="flex flex-wrap gap-2">
            {uniqueTreatments.map((treatment, idx) => (
              <div key={treatment} className="flex items-center gap-2">
                <div
                  className="w-4 h-4 rounded"
                  style={{ backgroundColor: getColor(idx, uniqueTreatments.length) }}
                ></div>
                <span className="text-xs text-gray-300">{treatment}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  // Render Crossover structure
  const renderCrossover = () => {
    // Group by sequence and subject
    const sequenceMap = {}
    designTable.forEach(row => {
      if (!sequenceMap[row.sequence]) {
        sequenceMap[row.sequence] = {}
      }
      if (!sequenceMap[row.sequence][row.subject]) {
        sequenceMap[row.sequence][row.subject] = []
      }
      sequenceMap[row.sequence][row.subject].push({
        period: row.period,
        treatment: row.treatment
      })
    })

    const uniqueTreatments = [...new Set(designTable.map(r => r.treatment))].sort()

    return (
      <div>
        <div className="mb-4">
          <h5 className="text-sm font-semibold text-gray-300 mb-2">Crossover Design Layout</h5>
          <p className="text-xs text-gray-400">Subjects receive treatments in different sequences across time periods</p>
        </div>

        {Object.keys(sequenceMap).sort().map(sequence => (
          <div key={sequence} className="mb-4 bg-slate-700/30 rounded-lg p-4">
            <div className="text-sm font-semibold text-gray-200 mb-3">Sequence: {sequence}</div>
            <div className="space-y-2">
              {Object.keys(sequenceMap[sequence]).sort().map(subject => {
                const periods = sequenceMap[sequence][subject].sort((a, b) =>
                  parseInt(a.period) - parseInt(b.period)
                )
                return (
                  <div key={subject} className="flex items-center gap-2">
                    <div className="text-xs text-gray-400 w-16">{subject}</div>
                    <div className="flex gap-2 flex-1">
                      {periods.map((p, idx) => {
                        const treatmentIndex = uniqueTreatments.indexOf(p.treatment)
                        const color = getColor(treatmentIndex, uniqueTreatments.length)
                        return (
                          <div key={idx} className="flex-1">
                            <div
                              className="px-3 py-2 rounded-lg font-semibold text-white text-center shadow-lg text-sm"
                              style={{ backgroundColor: color }}
                            >
                              <div className="text-xs opacity-75">Period {p.period}</div>
                              <div>{p.treatment}</div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        ))}

        {/* Treatment Legend */}
        <div className="mt-4 bg-slate-800/30 rounded-lg p-3">
          <div className="text-xs font-semibold text-gray-300 mb-2">Treatment Key</div>
          <div className="flex flex-wrap gap-2">
            {uniqueTreatments.map((treatment, idx) => (
              <div key={treatment} className="flex items-center gap-2">
                <div
                  className="w-4 h-4 rounded"
                  style={{ backgroundColor: getColor(idx, uniqueTreatments.length) }}
                ></div>
                <span className="text-xs text-gray-300">{treatment}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  // Render Incomplete Block structure
  const renderIncompleteBlock = () => {
    // Check if it's a Youden square (has row and column) or BIB (just block)
    const isYouden = designTable.some(row => row.row && row.column)

    if (isYouden) {
      // Youden Square rendering
      const rows = [...new Set(designTable.map(r => r.row))].sort()
      const cols = [...new Set(designTable.map(r => r.column))].sort()

      const grid = {}
      designTable.forEach(row => {
        if (!grid[row.row]) grid[row.row] = {}
        grid[row.row][row.column] = row.treatment
      })

      const uniqueTreatments = [...new Set(designTable.map(r => r.treatment))].sort()

      return (
        <div>
          <div className="mb-4">
            <h5 className="text-sm font-semibold text-gray-300 mb-2">Youden Square Layout</h5>
            <p className="text-xs text-gray-400">
              Incomplete Latin square - each treatment appears once per column, but not all treatments in each row
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className="bg-slate-700/50 border border-slate-600 px-3 py-2 text-xs text-gray-400">
                    Row / Col
                  </th>
                  {cols.map(col => (
                    <th
                      key={col}
                      className="bg-slate-700/50 border border-slate-600 px-3 py-2 text-xs font-semibold text-gray-300"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map(row => (
                  <tr key={row}>
                    <td className="bg-slate-700/50 border border-slate-600 px-3 py-2 text-xs font-semibold text-gray-300">
                      {row}
                    </td>
                    {cols.map(col => {
                      const treatment = grid[row]?.[col]
                      if (treatment) {
                        const treatmentIndex = uniqueTreatments.indexOf(treatment)
                        const color = getColor(treatmentIndex, uniqueTreatments.length)
                        return (
                          <td
                            key={col}
                            className="border border-slate-600 px-3 py-3 text-center text-white font-semibold"
                            style={{ backgroundColor: color }}
                          >
                            {treatment}
                          </td>
                        )
                      } else {
                        return (
                          <td
                            key={col}
                            className="border border-slate-600 px-3 py-3 text-center bg-slate-800/50"
                          >
                            <span className="text-gray-500 text-xs">-</span>
                          </td>
                        )
                      }
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Treatment Legend */}
          <div className="mt-4 bg-slate-800/30 rounded-lg p-3">
            <div className="text-xs font-semibold text-gray-300 mb-2">Treatment Key</div>
            <div className="flex flex-wrap gap-2">
              {uniqueTreatments.map((treatment, idx) => (
                <div key={treatment} className="flex items-center gap-2">
                  <div
                    className="w-4 h-4 rounded"
                    style={{ backgroundColor: getColor(idx, uniqueTreatments.length) }}
                  ></div>
                  <span className="text-xs text-gray-300">{treatment}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )
    } else {
      // BIB rendering
      const blockMap = {}
      designTable.forEach(row => {
        if (!blockMap[row.block]) {
          blockMap[row.block] = []
        }
        blockMap[row.block].push(row.treatment)
      })

      const blocks = Object.keys(blockMap).sort()
      const uniqueTreatments = [...new Set(designTable.map(r => r.treatment))].sort()

      return (
        <div>
          <div className="mb-4">
            <h5 className="text-sm font-semibold text-gray-300 mb-2">BIB Design Layout</h5>
            <p className="text-xs text-gray-400">
              Balanced Incomplete Block - not all treatments appear in each block, but design is balanced
            </p>
          </div>

          <div className="space-y-3">
            {blocks.map((block, blockIdx) => (
              <div key={block} className="bg-slate-700/30 rounded-lg p-4">
                <div className="text-sm font-semibold text-gray-200 mb-2">{block}</div>
                <div className="flex flex-wrap gap-2">
                  {blockMap[block].map((treatment, treatmentIdx) => {
                    const treatmentIndex = uniqueTreatments.indexOf(treatment)
                    const color = getColor(treatmentIndex, uniqueTreatments.length)
                    return (
                      <div
                        key={`${block}-${treatmentIdx}`}
                        className="px-4 py-2 rounded-lg font-semibold text-white shadow-lg"
                        style={{ backgroundColor: color }}
                      >
                        {treatment}
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>

          {/* Treatment Legend */}
          <div className="mt-4 bg-slate-800/30 rounded-lg p-3">
            <div className="text-xs font-semibold text-gray-300 mb-2">Treatment Key</div>
            <div className="flex flex-wrap gap-2">
              {uniqueTreatments.map((treatment, idx) => (
                <div key={treatment} className="flex items-center gap-2">
                  <div
                    className="w-4 h-4 rounded"
                    style={{ backgroundColor: getColor(idx, uniqueTreatments.length) }}
                  ></div>
                  <span className="text-xs text-gray-300">{treatment}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )
    }
  }

  // Render Graeco-Latin Square structure
  const renderGraecoLatinSquare = () => {
    // Create grid structure
    const grid = Array(squareSize).fill(null).map(() => Array(squareSize).fill(null))

    designTable.forEach(row => {
      grid[row.row - 1][row.column - 1] = {
        latin: row.latin_treatment,
        greek: row.greek_treatment
      }
    })

    const uniqueLatinTreatments = [...new Set(designTable.map(r => r.latin_treatment))].sort()
    const uniqueGreekTreatments = [...new Set(designTable.map(r => r.greek_treatment))].sort()

    return (
      <div>
        <div className="mb-4">
          <h5 className="text-sm font-semibold text-gray-300 mb-2">
            Graeco-Latin Square {squareSize}×{squareSize} Layout
          </h5>
          <p className="text-xs text-gray-400">
            Two orthogonal Latin squares superimposed. Each treatment combination appears once.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="bg-slate-700/50 border border-slate-600 px-3 py-2 text-xs text-gray-400">
                  Row / Col
                </th>
                {Array(squareSize).fill(null).map((_, colIdx) => (
                  <th
                    key={colIdx}
                    className="bg-slate-700/50 border border-slate-600 px-3 py-2 text-xs font-semibold text-gray-300"
                  >
                    Col {colIdx + 1}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {grid.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  <td className="bg-slate-700/50 border border-slate-600 px-3 py-2 text-xs font-semibold text-gray-300">
                    Row {rowIdx + 1}
                  </td>
                  {row.map((cell, colIdx) => {
                    const latinIndex = uniqueLatinTreatments.indexOf(cell.latin)
                    const greekIndex = uniqueGreekTreatments.indexOf(cell.greek)
                    const latinColor = getColor(latinIndex, uniqueLatinTreatments.length)
                    const greekColor = getColor(greekIndex, uniqueGreekTreatments.length)

                    return (
                      <td
                        key={colIdx}
                        className="border border-slate-600 px-2 py-2 text-center"
                        style={{
                          background: `linear-gradient(135deg, ${latinColor} 0%, ${latinColor} 50%, ${greekColor} 50%, ${greekColor} 100%)`
                        }}
                      >
                        <div className="flex flex-col text-white font-semibold text-xs">
                          <span className="drop-shadow-md">{cell.latin}</span>
                          <span className="drop-shadow-md">{cell.greek}</span>
                        </div>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Treatment Legends */}
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="bg-slate-800/30 rounded-lg p-3">
            <div className="text-xs font-semibold text-gray-300 mb-2">Latin Treatment Key</div>
            <div className="flex flex-wrap gap-2">
              {uniqueLatinTreatments.map((treatment, idx) => (
                <div key={treatment} className="flex items-center gap-2">
                  <div
                    className="w-4 h-4 rounded"
                    style={{ backgroundColor: getColor(idx, uniqueLatinTreatments.length) }}
                  ></div>
                  <span className="text-xs text-gray-300">{treatment}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-slate-800/30 rounded-lg p-3">
            <div className="text-xs font-semibold text-gray-300 mb-2">Greek Treatment Key</div>
            <div className="flex flex-wrap gap-2">
              {uniqueGreekTreatments.map((treatment, idx) => (
                <div key={treatment} className="flex items-center gap-2">
                  <div
                    className="w-4 h-4 rounded"
                    style={{ backgroundColor: getColor(idx, uniqueGreekTreatments.length) }}
                  ></div>
                  <span className="text-xs text-gray-300">{treatment}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center gap-2 mb-6">
        <Grid3X3 className="w-6 h-6 text-indigo-400" />
        <h3 className="text-xl font-bold text-gray-100">Block Structure Visualization</h3>
      </div>

      {designType === 'rcbd' && renderRCBD()}
      {designType === 'latin' && renderLatinSquare()}
      {designType === 'graeco' && renderGraecoLatinSquare()}
      {designType === 'crossover' && renderCrossover()}
      {designType === 'incomplete' && renderIncompleteBlock()}

      {/* Educational Note */}
      <div className="mt-6 bg-indigo-900/20 border border-indigo-700/30 rounded-lg p-4">
        <div className="flex items-start gap-2">
          <Info className="w-5 h-5 text-indigo-400 mt-0.5" />
          <div>
            <h5 className="font-semibold text-indigo-200 mb-1">About This Design</h5>
            <p className="text-sm text-indigo-100/80">
              {designType === 'rcbd' && (
                <>
                  In an RCBD, each block contains all treatments. Blocking controls nuisance variability
                  by grouping experimental units that are similar within blocks.
                </>
              )}
              {designType === 'latin' && (
                <>
                  Latin Square designs control two sources of nuisance variability (rows and columns).
                  Each treatment appears exactly once in each row and once in each column.
                </>
              )}
              {designType === 'graeco' && (
                <>
                  Graeco-Latin Squares control three sources of nuisance variability using two orthogonal
                  Latin squares. This design is highly efficient but requires specific square sizes.
                </>
              )}
              {designType === 'crossover' && (
                <>
                  Crossover designs are used when each subject receives multiple treatments over time.
                  Subjects are grouped by sequence, controlling for subject variability and period effects.
                  Carryover effects from previous treatments are also assessed.
                </>
              )}
              {designType === 'incomplete' && (
                <>
                  Incomplete Block designs are used when block sizes must be smaller than the number of
                  treatments. BIB designs balance treatment comparisons, while Youden Squares control
                  both row and column effects with incomplete blocks.
                </>
              )}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default BlockStructureVisualization
