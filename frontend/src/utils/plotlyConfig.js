/**
 * Get standardized Plotly config with PNG and SVG export options
 * @param {string} filename - Base filename for exports (without extension)
 * @param {Object} additionalConfig - Additional Plotly config options to merge
 * @returns {Object} Plotly config object
 */
export const getPlotlyConfig = (filename = 'plot', additionalConfig = {}) => {
  const today = new Date().toISOString().split('T')[0]
  const baseFilename = `${filename}-${today}`

  return {
    displayModeBar: true,
    displaylogo: false,
    toImageButtonOptions: {
      format: 'png',
      filename: baseFilename,
      height: 800,
      width: 1200,
      scale: 2
    },
    modeBarButtonsToAdd: [
      {
        name: 'Download as SVG',
        icon: {
          width: 857.1,
          height: 1000,
          path: 'm214-7h429v214h-429v-214z m500 0h72v500q0 8-6 21t-11 20l-157 156q-5 6-19 12t-22 5v-232q0-22-15-38t-38-16h-322q-22 0-37 16t-16 38v232h-72v-714h72v232q0 22 16 38t37 16h465q22 0 38-16t15-38v-232z m-214 518v178q0 8-5 13t-13 5h-107q-7 0-13-5t-5-13v-178q0-8 5-13t13-5h107q7 0 13 5t5 13z m357-18v-518q0-22-15-38t-38-16h-750q-23 0-38 16t-16 38v750q0 22 16 38t38 16h517q23 0 50-12t42-26l156-157q16-15 27-42t11-49z',
          transform: 'matrix(1 0 0 -1 0 850)'
        },
        click: function(gd) {
          const filename = `${baseFilename}.svg`
          // Use Plotly's built-in downloadImage function with SVG format
          if (window.Plotly && window.Plotly.downloadImage) {
            window.Plotly.downloadImage(gd, {
              format: 'svg',
              filename: filename,
              height: 800,
              width: 1200
            })
          }
        }
      }
    ],
    responsive: true,
    ...additionalConfig
  }
}

/**
 * Get standardized Plotly layout with dark theme
 * @param {string} title - Plot title
 * @param {Object} additionalLayout - Additional layout options to merge
 * @returns {Object} Plotly layout object
 */
export const getPlotlyLayout = (title = '', additionalLayout = {}) => {
  return {
    title: {
      text: title,
      font: { size: 18, color: '#f1f5f9' }
    },
    paper_bgcolor: 'rgba(15, 23, 42, 0.5)',
    plot_bgcolor: 'rgba(30, 41, 59, 0.5)',
    font: { color: '#e2e8f0' },
    xaxis: {
      gridcolor: 'rgba(51, 65, 85, 0.5)',
      zerolinecolor: 'rgba(71, 85, 105, 0.7)'
    },
    yaxis: {
      gridcolor: 'rgba(51, 65, 85, 0.5)',
      zerolinecolor: 'rgba(71, 85, 105, 0.7)'
    },
    ...additionalLayout
  }
}
