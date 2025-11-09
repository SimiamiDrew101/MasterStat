/**
 * Export SVG element to PNG file
 * @param {SVGElement} svgElement - The SVG element to export
 * @param {string} filename - The filename for the downloaded PNG (without extension)
 * @param {Object} options - Optional configuration
 * @param {number} options.scale - Scale factor for higher resolution (default: 2)
 * @param {string} options.backgroundColor - Background color (default: '#0f172a' - slate-900)
 */
export const exportSvgToPng = (svgElement, filename = 'chart', options = {}) => {
  const {
    scale = 2,
    backgroundColor = '#0f172a'
  } = options

  if (!svgElement) {
    console.error('No SVG element provided for export')
    return
  }

  try {
    // Clone the SVG to avoid modifying the original
    const svgClone = svgElement.cloneNode(true)

    // Get SVG dimensions
    const svgRect = svgElement.getBoundingClientRect()
    const width = svgRect.width
    const height = svgRect.height

    // Set explicit dimensions on the clone
    svgClone.setAttribute('width', width)
    svgClone.setAttribute('height', height)

    // Serialize SVG to string
    const svgString = new XMLSerializer().serializeToString(svgClone)
    const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' })
    const svgUrl = URL.createObjectURL(svgBlob)

    // Create an image from the SVG
    const img = new Image()
    img.onload = () => {
      // Create canvas
      const canvas = document.createElement('canvas')
      canvas.width = width * scale
      canvas.height = height * scale
      const ctx = canvas.getContext('2d')

      // Set background color
      ctx.fillStyle = backgroundColor
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Scale for higher resolution
      ctx.scale(scale, scale)

      // Draw the image
      ctx.drawImage(img, 0, 0, width, height)

      // Convert canvas to blob and download
      canvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `${filename}.png`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)

        // Cleanup
        URL.revokeObjectURL(url)
        URL.revokeObjectURL(svgUrl)
      }, 'image/png')
    }

    img.onerror = (error) => {
      console.error('Error loading SVG image:', error)
      URL.revokeObjectURL(svgUrl)
    }

    img.src = svgUrl
  } catch (error) {
    console.error('Error exporting chart:', error)
  }
}

/**
 * Export multiple SVG elements to a single PNG
 * @param {Array<SVGElement>} svgElements - Array of SVG elements
 * @param {string} filename - The filename for the downloaded PNG
 * @param {Object} options - Optional configuration
 */
export const exportMultipleSvgsToPng = (svgElements, filename = 'charts', options = {}) => {
  const {
    scale = 2,
    backgroundColor = '#0f172a',
    spacing = 20
  } = options

  if (!svgElements || svgElements.length === 0) {
    console.error('No SVG elements provided for export')
    return
  }

  try {
    // Get dimensions of all SVGs
    const svgDimensions = svgElements.map(svg => svg.getBoundingClientRect())

    // Calculate total canvas dimensions
    const totalWidth = Math.max(...svgDimensions.map(d => d.width))
    const totalHeight = svgDimensions.reduce((sum, d) => sum + d.height, 0) + spacing * (svgElements.length - 1)

    // Create canvas
    const canvas = document.createElement('canvas')
    canvas.width = totalWidth * scale
    canvas.height = totalHeight * scale
    const ctx = canvas.getContext('2d')

    // Set background
    ctx.fillStyle = backgroundColor
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.scale(scale, scale)

    let currentY = 0
    let loadedCount = 0

    // Process each SVG
    svgElements.forEach((svgElement, index) => {
      const svgClone = svgElement.cloneNode(true)
      const width = svgDimensions[index].width
      const height = svgDimensions[index].height

      svgClone.setAttribute('width', width)
      svgClone.setAttribute('height', height)

      const svgString = new XMLSerializer().serializeToString(svgClone)
      const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' })
      const svgUrl = URL.createObjectURL(svgBlob)

      const img = new Image()
      img.onload = () => {
        ctx.drawImage(img, 0, currentY, width, height)
        currentY += height + spacing
        URL.revokeObjectURL(svgUrl)

        loadedCount++
        if (loadedCount === svgElements.length) {
          // All images loaded, download
          canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob)
            const link = document.createElement('a')
            link.href = url
            link.download = `${filename}.png`
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
            URL.revokeObjectURL(url)
          }, 'image/png')
        }
      }

      img.src = svgUrl
    })
  } catch (error) {
    console.error('Error exporting charts:', error)
  }
}
