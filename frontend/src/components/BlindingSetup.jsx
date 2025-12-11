import { useState, useEffect } from 'react'
import { Eye, EyeOff, Key, Download, AlertTriangle, CheckCircle, Info } from 'lucide-react'

const BlindingSetup = ({ protocol, setProtocol }) => {
  const [generatedCodes, setGeneratedCodes] = useState({})
  const [customCodeInput, setCustomCodeInput] = useState({})

  // Auto-generate codes when blinding type or code type changes
  useEffect(() => {
    if (protocol.blinding.type !== 'none') {
      generateBlindingCodes()
    }
  }, [protocol.blinding.type, protocol.blinding.codeType])

  /**
   * Generate blinding codes based on treatments
   */
  const generateBlindingCodes = () => {
    const treatments = extractTreatments(protocol)
    let codes = {}

    if (protocol.blinding.codeType === 'alphabetic') {
      treatments.forEach((treatment, idx) => {
        codes[treatment] = String.fromCharCode(65 + idx) // A, B, C, D...
      })
    } else if (protocol.blinding.codeType === 'numeric') {
      treatments.forEach((treatment, idx) => {
        codes[treatment] = `T${idx + 1}` // T1, T2, T3...
      })
    } else if (protocol.blinding.codeType === 'custom') {
      // Use existing custom codes or initialize empty
      codes = protocol.blinding.customCodes || {}
    }

    setGeneratedCodes(codes)
    setProtocol(prev => ({
      ...prev,
      blinding: {
        ...prev.blinding,
        generatedCodes: codes,
      },
    }))
  }

  /**
   * Extract unique treatment combinations from protocol
   */
  const extractTreatments = (protocol) => {
    const factors = protocol.materials?.factors || []

    if (factors.length === 0) {
      return ['Treatment 1', 'Treatment 2'] // Default
    }

    // Generate treatment labels from factors
    const treatments = []

    // For simplicity, create basic treatment labels
    // In real implementation, would generate from design matrix
    if (factors.length === 1) {
      treatments.push('Low', 'High')
    } else if (factors.length === 2) {
      treatments.push('Low-Low', 'Low-High', 'High-Low', 'High-High')
    } else {
      // For 3+ factors, use numeric codes
      const numTreatments = Math.min(Math.pow(2, factors.length), 26)
      for (let i = 0; i < numTreatments; i++) {
        treatments.push(`Treatment ${i + 1}`)
      }
    }

    return treatments
  }

  /**
   * Download blinding key as text file
   */
  const downloadBlindingKey = () => {
    if (Object.keys(generatedCodes).length === 0) {
      alert('No blinding codes generated yet')
      return
    }

    const keyContent = Object.entries(generatedCodes)
      .map(([treatment, code]) => `${code}\t${treatment}`)
      .join('\n')

    const fullContent = `BLINDING KEY - CONFIDENTIAL

Protocol: ${protocol.metadata?.title || 'Untitled'}
Investigator: ${protocol.metadata?.investigator || 'Unknown'}
Date: ${new Date().toISOString().split('T')[0]}
Blinding Type: ${protocol.blinding.type}

⚠️ DO NOT SHARE WITH BLINDED PARTIES ⚠️

Code\tActual Treatment
${keyContent}

This file contains the treatment assignment codes.
Keep secure and separate from the main protocol document.
`

    const blob = new Blob([fullContent], { type: 'text/plain' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `blinding_key_${protocol.metadata?.title?.replace(/\s+/g, '_') || 'protocol'}_${Date.now()}.txt`
    link.click()
    window.URL.revokeObjectURL(url)
  }

  /**
   * Handle custom code input
   */
  const handleCustomCodeChange = (treatment, code) => {
    const updated = {
      ...protocol.blinding.customCodes,
      [treatment]: code,
    }

    setProtocol(prev => ({
      ...prev,
      blinding: {
        ...prev.blinding,
        customCodes: updated,
        generatedCodes: updated,
      },
    }))
  }

  const treatments = extractTreatments(protocol)
  const hasBlindingCodes = Object.keys(generatedCodes).length > 0

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-100">Blinding & Masking</h2>

      <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4">
        <div className="flex items-start gap-2 text-blue-400">
          <Info className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div className="text-sm">
            <p className="font-semibold mb-1">What is blinding?</p>
            <p className="text-blue-300">
              Blinding (or masking) prevents bias by hiding treatment assignments from participants,
              experimenters, and/or evaluators. This ensures objective measurements and reduces placebo effects.
            </p>
          </div>
        </div>
      </div>

      {/* Blinding Type Selection */}
      <div>
        <label className="block text-gray-200 font-medium mb-3">Blinding Type *</label>
        <div className="grid grid-cols-3 gap-3">
          <button
            onClick={() => setProtocol(prev => ({
              ...prev,
              blinding: { ...prev.blinding, type: 'none', blindedParties: [] }
            }))}
            className={`px-4 py-4 rounded-lg border-2 transition ${
              protocol.blinding.type === 'none'
                ? 'border-slate-500 bg-slate-600/20 text-gray-200'
                : 'border-slate-600 bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            <EyeOff className="w-8 h-8 mx-auto mb-2" />
            <div className="font-semibold">None</div>
            <div className="text-xs opacity-75 mt-1">Open label</div>
          </button>

          <button
            onClick={() => setProtocol(prev => ({
              ...prev,
              blinding: { ...prev.blinding, type: 'single' }
            }))}
            className={`px-4 py-4 rounded-lg border-2 transition ${
              protocol.blinding.type === 'single'
                ? 'border-indigo-500 bg-indigo-600/20 text-indigo-300'
                : 'border-slate-600 bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            <Eye className="w-8 h-8 mx-auto mb-2" />
            <div className="font-semibold">Single-Blind</div>
            <div className="text-xs opacity-75 mt-1">One party blind</div>
          </button>

          <button
            onClick={() => setProtocol(prev => ({
              ...prev,
              blinding: { ...prev.blinding, type: 'double' }
            }))}
            className={`px-4 py-4 rounded-lg border-2 transition ${
              protocol.blinding.type === 'double'
                ? 'border-purple-500 bg-purple-600/20 text-purple-300'
                : 'border-slate-600 bg-slate-700/50 text-gray-300 hover:bg-slate-700'
            }`}
          >
            <Key className="w-8 h-8 mx-auto mb-2" />
            <div className="font-semibold">Double-Blind</div>
            <div className="text-xs opacity-75 mt-1">Two parties blind</div>
          </button>
        </div>
      </div>

      {/* Blinded Parties - only show if blinding is enabled */}
      {protocol.blinding.type !== 'none' && (
        <>
          <div>
            <label className="block text-gray-200 font-medium mb-3">Who is Blinded? *</label>
            <div className="space-y-2">
              {['experimenter', 'evaluator', 'subject'].map(party => (
                <label
                  key={party}
                  className="flex items-center gap-3 px-4 py-3 bg-slate-700/30 rounded-lg cursor-pointer hover:bg-slate-700/50 transition"
                >
                  <input
                    type="checkbox"
                    checked={protocol.blinding.blindedParties.includes(party)}
                    onChange={(e) => {
                      const updated = e.target.checked
                        ? [...protocol.blinding.blindedParties, party]
                        : protocol.blinding.blindedParties.filter(p => p !== party)
                      setProtocol(prev => ({
                        ...prev,
                        blinding: { ...prev.blinding, blindedParties: updated }
                      }))
                    }}
                    className="w-5 h-5 rounded bg-slate-600 border-slate-500"
                  />
                  <div className="flex-1">
                    <span className="text-gray-200 capitalize font-medium">{party}</span>
                    <div className="text-xs text-gray-400 mt-0.5">
                      {party === 'experimenter' && 'Person applying treatments'}
                      {party === 'evaluator' && 'Person measuring outcomes'}
                      {party === 'subject' && 'Experimental unit or participant'}
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Code Generation Type */}
          <div>
            <label className="block text-gray-200 font-medium mb-3">Code Generation Type</label>
            <div className="grid grid-cols-3 gap-3">
              <button
                onClick={() => setProtocol(prev => ({
                  ...prev,
                  blinding: { ...prev.blinding, codeType: 'alphabetic' }
                }))}
                className={`px-4 py-3 rounded-lg border transition ${
                  protocol.blinding.codeType === 'alphabetic'
                    ? 'border-indigo-500 bg-indigo-600/20 text-indigo-300'
                    : 'border-slate-600 bg-slate-700/50 text-gray-300 hover:bg-slate-700'
                }`}
              >
                <div className="font-mono font-bold mb-1">A, B, C...</div>
                <div className="text-xs opacity-75">Alphabetic</div>
              </button>

              <button
                onClick={() => setProtocol(prev => ({
                  ...prev,
                  blinding: { ...prev.blinding, codeType: 'numeric' }
                }))}
                className={`px-4 py-3 rounded-lg border transition ${
                  protocol.blinding.codeType === 'numeric'
                    ? 'border-indigo-500 bg-indigo-600/20 text-indigo-300'
                    : 'border-slate-600 bg-slate-700/50 text-gray-300 hover:bg-slate-700'
                }`}
              >
                <div className="font-mono font-bold mb-1">T1, T2, T3...</div>
                <div className="text-xs opacity-75">Numeric</div>
              </button>

              <button
                onClick={() => setProtocol(prev => ({
                  ...prev,
                  blinding: { ...prev.blinding, codeType: 'custom' }
                }))}
                className={`px-4 py-3 rounded-lg border transition ${
                  protocol.blinding.codeType === 'custom'
                    ? 'border-indigo-500 bg-indigo-600/20 text-indigo-300'
                    : 'border-slate-600 bg-slate-700/50 text-gray-300 hover:bg-slate-700'
                }`}
              >
                <div className="font-mono font-bold mb-1">Custom</div>
                <div className="text-xs opacity-75">Your labels</div>
              </button>
            </div>
          </div>

          {/* Custom Code Entry */}
          {protocol.blinding.codeType === 'custom' && (
            <div className="bg-orange-900/20 border border-orange-700 rounded-lg p-4">
              <div className="flex items-center gap-2 text-orange-400 mb-3">
                <AlertTriangle className="w-5 h-5" />
                <span className="font-semibold">Custom Code Entry</span>
              </div>
              <p className="text-gray-300 text-sm mb-4">
                Enter a unique code label for each treatment combination:
              </p>
              <div className="space-y-3">
                {treatments.map((treatment, idx) => (
                  <div key={idx} className="flex items-center gap-3">
                    <span className="text-gray-400 text-sm w-32 flex-shrink-0">{treatment}:</span>
                    <input
                      type="text"
                      value={protocol.blinding.customCodes?.[treatment] || ''}
                      onChange={(e) => handleCustomCodeChange(treatment, e.target.value)}
                      placeholder={`Code ${idx + 1}`}
                      className="flex-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded text-gray-100 focus:ring-2 focus:ring-orange-500"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Generated Codes Preview */}
          {hasBlindingCodes && protocol.blinding.codeType !== 'custom' && (
            <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-gray-100">Generated Blinding Codes</h4>
                <CheckCircle className="w-5 h-5 text-green-400" />
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-4">
                {Object.entries(generatedCodes).slice(0, 9).map(([treatment, code]) => (
                  <div key={code} className="flex justify-between items-center px-3 py-2 bg-slate-800 rounded">
                    <span className="text-indigo-400 font-mono font-bold">{code}</span>
                    <span className="text-gray-400 text-sm blur-sm hover:blur-none transition cursor-help" title={treatment}>
                      {treatment.substring(0, 15)}{treatment.length > 15 ? '...' : ''}
                    </span>
                  </div>
                ))}
              </div>

              {Object.keys(generatedCodes).length > 9 && (
                <p className="text-gray-400 text-xs text-center mb-4">
                  ... and {Object.keys(generatedCodes).length - 9} more codes
                </p>
              )}

              <button
                onClick={downloadBlindingKey}
                className="w-full px-4 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg flex items-center justify-center gap-2 font-semibold transition"
              >
                <Download className="w-5 h-5" />
                Download Blinding Key (Keep Secure!)
              </button>

              <p className="text-xs text-gray-400 mt-3 text-center">
                ⚠️ The blinding key file must be kept secure and separate from the protocol
              </p>
            </div>
          )}

          {/* Unblinding Criteria */}
          <div>
            <label className="block text-gray-200 font-medium mb-2">Unblinding Criteria</label>
            <textarea
              value={protocol.blinding.unblindingCriteria || ''}
              onChange={(e) => setProtocol(prev => ({
                ...prev,
                blinding: { ...prev.blinding, unblindingCriteria: e.target.value }
              }))}
              placeholder="Under what conditions will the blinding be broken? (e.g., serious adverse events, study completion)"
              className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
              rows={3}
            />
          </div>
        </>
      )}
    </div>
  )
}

export default BlindingSetup
