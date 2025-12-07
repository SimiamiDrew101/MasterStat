import { useState, useMemo } from 'react'
import Plot from 'react-plotly.js'
import { Grid3x3 } from 'lucide-react'

const CorrelationHeatmap = ({
  correlationMatrix,
  pValues = null,
  variableNames = null,
  data = null,  // Raw data as {var1: [values], var2: [values], ...}
  method = 'pearson',  // Correlation method when using raw data
  title = 'Correlation Matrix',
  showValues = true,
  colorscale = 'RdBu'
}) => {
  const [displayValues, setDisplayValues] = useState(showValues)
  const [selectedColorscale, setSelectedColorscale] = useState(colorscale)

  // Calculate correlation matrix from raw data if provided
  const calculatedCorrelation = useMemo(() => {
    if (!data || Object.keys(data).length === 0) return null

    const varNames = Object.keys(data)
    const n = varNames.length
    const matrix = []

    // Calculate correlation for each pair
    for (let i = 0; i < n; i++) {
      const row = []
      for (let j = 0; j < n; j++) {
        if (i === j) {
          row.push(1.0)
        } else {
          const x = data[varNames[i]]
          const y = data[varNames[j]]

          // Calculate Pearson correlation
          const meanX = x.reduce((a, b) => a + b, 0) / x.length
          const meanY = y.reduce((a, b) => a + b, 0) / y.length

          let numerator = 0
          let denomX = 0
          let denomY = 0

          for (let k = 0; k < x.length; k++) {
            const dx = x[k] - meanX
            const dy = y[k] - meanY
            numerator += dx * dy
            denomX += dx * dx
            denomY += dy * dy
          }

          const corr = numerator / Math.sqrt(denomX * denomY)
          row.push(isNaN(corr) ? 0 : corr)
        }
      }
      matrix.push(row)
    }

    return { matrix, varNames }
  }, [data, method])

  // Use calculated or provided correlation matrix
  const finalCorrelationMatrix = correlationMatrix || calculatedCorrelation?.matrix
  const finalVariableNames = variableNames || calculatedCorrelation?.varNames

  // Process correlation matrix
  const { matrix, labels, annotations } = useMemo(() => {
    if (!finalCorrelationMatrix || finalCorrelationMatrix.length === 0) {
      return { matrix: [], labels: [], annotations: [] }
    }

    // Extract variable names (use provided names or default)
    const names = finalVariableNames || finalCorrelationMatrix.map((_, i) => `Var${i + 1}`)

    // Prepare annotations with correlation coefficients and p-values
    const annot = []
    for (let i = 0; i < finalCorrelationMatrix.length; i++) {
      for (let j = 0; j < finalCorrelationMatrix[i].length; j++) {
        const corr = finalCorrelationMatrix[i][j]
        let text = corr.toFixed(3)

        // Add significance stars if p-values are provided
        if (pValues && pValues[i] && pValues[i][j] !== undefined) {
          const pval = pValues[i][j]
          if (pval < 0.001) text += '***'
          else if (pval < 0.01) text += '**'
          else if (pval < 0.05) text += '*'
        }

        annot.push({
          x: names[j],
          y: names[i],
          text: displayValues ? text : '',
          font: {
            color: Math.abs(corr) > 0.5 ? '#ffffff' : '#1e293b',
            size: 10
          },
          showarrow: false
        })
      }
    }

    return {
      matrix: finalCorrelationMatrix,
      labels: names,
      annotations: annot
    }
  }, [finalCorrelationMatrix, finalVariableNames, pValues, displayValues])

  if (!finalCorrelationMatrix || finalCorrelationMatrix.length === 0) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center gap-3 mb-4">
          <Grid3x3 className="w-6 h-6 text-purple-400" />
          <h3 className="text-xl font-bold text-gray-100">Correlation Heatmap</h3>
        </div>
        <p className="text-gray-400">No correlation data available</p>
      </div>
    )
  }

  const colorScales = {
    RdBu: 'RdBu',
    Viridis: 'Viridis',
    Portland: 'Portland',
    Picnic: 'Picnic',
    Jet: 'Jet'
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Grid3x3 className="w-6 h-6 text-purple-400" />
          <h3 className="text-xl font-bold text-gray-100">{title}</h3>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4">
          {/* Color Scale Selector */}
          <div className="flex items-center gap-2">
            <label className="text-gray-300 text-sm">Color Scale:</label>
            <select
              value={selectedColorscale}
              onChange={(e) => setSelectedColorscale(e.target.value)}
              className="bg-slate-700 text-gray-100 px-3 py-1 rounded-lg text-sm border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              {Object.entries(colorScales).map(([key, value]) => (
                <option key={key} value={value}>{key}</option>
              ))}
            </select>
          </div>

          {/* Toggle Values */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={displayValues}
              onChange={(e) => setDisplayValues(e.target.checked)}
              className="w-4 h-4 text-purple-500 bg-slate-700 border-slate-600 rounded focus:ring-purple-500"
            />
            <span className="text-gray-300 text-sm">Show Values</span>
          </label>
        </div>
      </div>

      {/* Plot */}
      <Plot
        data={[
          {
            z: matrix,
            x: labels,
            y: labels,
            type: 'heatmap',
            colorscale: selectedColorscale,
            zmin: -1,
            zmax: 1,
            colorbar: {
              title: 'Correlation',
              titleside: 'right',
              tickmode: 'linear',
              tick0: -1,
              dtick: 0.5,
              len: 0.7,
              thickness: 15,
              outlinewidth: 0,
              tickfont: { color: '#e2e8f0' },
              titlefont: { color: '#e2e8f0' }
            },
            hovertemplate: '<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
          }
        ]}
        layout={{
          autosize: true,
          height: Math.max(400, labels.length * 40),
          plot_bgcolor: 'rgba(15, 23, 42, 0.5)',
          paper_bgcolor: 'rgba(15, 23, 42, 0)',
          font: { color: '#e2e8f0', family: 'Inter, system-ui, sans-serif' },
          xaxis: {
            tickangle: -45,
            side: 'bottom',
            gridcolor: 'rgba(148, 163, 184, 0.1)',
            tickfont: { size: 11 }
          },
          yaxis: {
            autorange: 'reversed',
            gridcolor: 'rgba(148, 163, 184, 0.1)',
            tickfont: { size: 11 }
          },
          margin: { l: 100, r: 80, t: 40, b: 100 },
          annotations: displayValues ? annotations : [],
          hovermode: 'closest'
        }}
        config={{
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          toImageButtonOptions: {
            format: 'png',
            filename: `correlation_heatmap_${new Date().toISOString().slice(0, 10)}`,
            height: 800,
            width: 800,
            scale: 2
          }
        }}
        style={{ width: '100%' }}
        useResizeHandler={true}
      />

      {/* Legend */}
      {pValues && (
        <div className="mt-4 bg-slate-700/30 rounded-lg p-3">
          <p className="text-gray-300 text-sm">
            <strong>Significance levels:</strong> * p {'<'} 0.05, ** p {'<'} 0.01, *** p {'<'} 0.001
          </p>
        </div>
      )}

      {/* Summary Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-4">
        <div className="bg-slate-700/30 rounded-lg p-3">
          <p className="text-gray-400 text-xs">Variables</p>
          <p className="text-gray-100 font-semibold">{labels.length}</p>
        </div>
        <div className="bg-slate-700/30 rounded-lg p-3">
          <p className="text-gray-400 text-xs">Strong Correlations (|r| &gt; 0.7)</p>
          <p className="text-gray-100 font-semibold">
            {matrix.reduce((count, row, i) =>
              count + row.reduce((rowCount, val, j) =>
                rowCount + (i !== j && Math.abs(val) > 0.7 ? 1 : 0), 0), 0) / 2}
          </p>
        </div>
        <div className="bg-slate-700/30 rounded-lg p-3">
          <p className="text-gray-400 text-xs">Weak Correlations (|r| &lt; 0.3)</p>
          <p className="text-gray-100 font-semibold">
            {matrix.reduce((count, row, i) =>
              count + row.reduce((rowCount, val, j) =>
                rowCount + (i !== j && Math.abs(val) < 0.3 ? 1 : 0), 0), 0) / 2}
          </p>
        </div>
      </div>
    </div>
  )
}

export default CorrelationHeatmap
