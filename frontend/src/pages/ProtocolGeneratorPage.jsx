import { useEffect, useState } from 'react'
import ProtocolGenerator from '../components/ProtocolGenerator'

/**
 * Standalone Protocol Generator Page
 * Accessible directly via /protocol-generator route
 */
const ProtocolGeneratorPage = () => {
  const [savedProtocol, setSavedProtocol] = useState(null)

  // Load saved draft from localStorage on mount
  useEffect(() => {
    try {
      const draft = localStorage.getItem('protocol_draft')
      if (draft) {
        setSavedProtocol(JSON.parse(draft))
      }
    } catch (error) {
      console.error('Failed to load saved protocol:', error)
    }
  }, [])

  return (
    <ProtocolGenerator
      designData={savedProtocol}  // Load from localStorage if available
      standalone={true}            // Standalone mode (enables auto-save)
      onComplete={(protocol) => {
        console.log('Protocol completed:', protocol)
        // Could trigger additional actions here
      }}
    />
  )
}

export default ProtocolGeneratorPage
