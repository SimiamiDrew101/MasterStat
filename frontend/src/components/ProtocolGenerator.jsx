import { useState, useEffect } from 'react'
import axios from 'axios'
import { FileText, Download, AlertCircle, CheckCircle, Info, Save, ChevronRight } from 'lucide-react'
import { createEmptyProtocol, applyTemplate } from '../utils/protocolTemplates'
import { validateProtocol, validateSection } from '../utils/protocolValidation'
import RandomizationPanel from './RandomizationPanel'
import BlindingSetup from './BlindingSetup'
import {
  ObjectiveSection,
  MaterialsSection,
  ProcedureSection,
  DataRecordingSection,
} from './ProtocolSections'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const ProtocolGenerator = ({
  designData = null,  // Pre-populated from wizard
  onSave = null,      // Optional save callback
  standalone = false,  // true = standalone page, false = wizard step
  onComplete = null,   // Callback when protocol is completed
}) => {
  const [protocol, setProtocol] = useState(() => {
    let empty = createEmptyProtocol(designData?.designType || 'factorial')

    // Auto-populate from design data if available
    if (designData) {
      empty = {
        ...empty,
        metadata: {
          ...empty.metadata,
          title: designData.title || `${(designData.designType || 'factorial').toUpperCase()} Experimental Protocol`,
          designType: designData.designType || 'factorial',
        },
        materials: {
          ...empty.materials,
          factors: designData.factors || [],
          sampleSize: designData.runs || designData.sampleSize || null,
        },
        randomization: {
          ...empty.randomization,
          randomizedDesign: designData.design || null,
        },
      }

      // Apply template for design type
      empty = applyTemplate(empty, designData.designType || 'factorial')
    }

    return empty
  })

  const [currentSection, setCurrentSection] = useState('objective')
  const [validation, setValidation] = useState({ errors: [], warnings: [], isValid: false })
  const [isSaving, setIsSaving] = useState(false)
  const [isExporting, setIsExporting] = useState(false)

  // Validation on protocol change
  useEffect(() => {
    const result = validateProtocol(protocol)
    setValidation(result)
  }, [protocol])

  // Auto-save to localStorage
  useEffect(() => {
    if (standalone) {
      try {
        localStorage.setItem('protocol_draft', JSON.stringify(protocol))
      } catch (error) {
        console.error('Failed to save to localStorage:', error)
      }
    }
  }, [protocol, standalone])

  /**
   * Update a field in a section
   */
  const updateField = (section, field, value) => {
    setProtocol(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value,
      },
    }))
  }

  /**
   * Update metadata field
   */
  const updateMetadata = (field, value) => {
    setProtocol(prev => ({
      ...prev,
      metadata: {
        ...prev.metadata,
        [field]: value,
      },
    }))
  }

  /**
   * Section configuration
   */
  const sections = [
    { id: 'objective', label: 'Objective & Hypothesis', icon: '🎯', required: true },
    { id: 'materials', label: 'Materials & Design', icon: '🔬', required: true },
    { id: 'procedure', label: 'Experimental Procedure', icon: '📋', required: true },
    { id: 'randomization', label: 'Randomization', icon: '🎲', required: true },
    { id: 'blinding', label: 'Blinding & Masking', icon: '👁️', required: false },
    { id: 'dataRecording', label: 'Data Recording', icon: '📊', required: true },
  ]

  /**
   * Render section content based on currentSection
   */
  const renderSectionContent = () => {
    switch (currentSection) {
      case 'objective':
        return <ObjectiveSection protocol={protocol} updateField={updateField} />
      case 'materials':
        return <MaterialsSection protocol={protocol} updateField={updateField} />
      case 'procedure':
        return <ProcedureSection protocol={protocol} updateField={updateField} />
      case 'randomization':
        return <RandomizationPanel protocol={protocol} setProtocol={setProtocol} />
      case 'blinding':
        return <BlindingSetup protocol={protocol} setProtocol={setProtocol} />
      case 'dataRecording':
        return <DataRecordingSection protocol={protocol} updateField={updateField} />
      default:
        return null
    }
  }

  /**
   * Export PDF protocol
   */
  const handleExportPDF = async () => {
    // Validate first
    if (validation.errors.length > 0) {
      alert('Please fix all errors before exporting:\n\n' + validation.errors.join('\n'))
      return
    }

    setIsExporting(true)
    try {
      // Call backend API
      const response = await axios.post(
        `${API_URL}/api/protocol/generate-pdf`,
        protocol,
        { responseType: 'blob' }
      )

      // Download main protocol PDF
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      const filename = `protocol_${protocol.metadata.title.replace(/\s+/g, '_')}_${Date.now()}.pdf`
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)

      // If blinded, also download blinding key
      if (protocol.blinding.type !== 'none') {
        await downloadBlindingKeyPDF()
      }

      alert('Protocol PDF generated successfully!')

      if (onComplete) {
        onComplete(protocol)
      }
    } catch (error) {
      console.error('PDF export failed:', error)
      alert(`Failed to generate PDF: ${error.response?.data?.detail || error.message}`)
    } finally {
      setIsExporting(false)
    }
  }

  /**
   * Download blinding key as separate PDF
   */
  const downloadBlindingKeyPDF = async () => {
    try {
      const response = await axios.post(
        `${API_URL}/api/protocol/generate-blinding-key`,
        protocol,
        { responseType: 'blob' }
      )

      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      const filename = `blinding_key_${protocol.metadata.title.replace(/\s+/g, '_')}_${Date.now()}.pdf`
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Blinding key export failed:', error)
      alert('Warning: Failed to generate blinding key PDF')
    }
  }

  /**
   * Save protocol (if in wizard mode)
   */
  const handleSave = () => {
    if (onSave) {
      onSave(protocol)
    }
    setIsSaving(true)
    setTimeout(() => setIsSaving(false), 1000)
  }

  /**
   * Get section validation status
   */
  const getSectionStatus = (sectionId) => {
    const sectionValidation = validateSection(protocol, sectionId)
    const hasErrors = sectionValidation.errors.length > 0
    const hasContent = protocol[sectionId] && Object.values(protocol[sectionId]).some(v =>
      Array.isArray(v) ? v.length > 0 : v !== '' && v !== null && v !== undefined
    )

    return {
      hasErrors,
      hasContent,
      isComplete: hasContent && !hasErrors,
    }
  }

  // Progress calculation
  const completedSections = sections.filter(s => getSectionStatus(s.id).isComplete).length
  const progress = (completedSections / sections.length) * 100

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-600 text-transparent bg-clip-text mb-2">
                Experimental Protocol Generator
              </h1>
              <p className="text-gray-300">
                Create a comprehensive, randomized, and optionally blinded experimental protocol
              </p>
            </div>
            {!standalone && onSave && (
              <button
                onClick={handleSave}
                disabled={isSaving}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-gray-100 rounded-lg flex items-center gap-2 transition"
              >
                <Save className="w-4 h-4" />
                {isSaving ? 'Saved!' : 'Save Draft'}
              </button>
            )}
          </div>

          {/* Progress Bar */}
          <div className="bg-slate-700/30 rounded-full h-2 overflow-hidden">
            <div
              className="bg-gradient-to-r from-blue-500 to-purple-500 h-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-sm text-gray-400 mt-1">
            {completedSections} of {sections.length} sections completed ({Math.round(progress)}%)
          </p>
        </div>

        {/* Metadata Section */}
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-6 border border-slate-700/50 mb-6">
          <h3 className="text-lg font-semibold text-gray-100 mb-4">Protocol Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-200 font-medium mb-2">Protocol Title *</label>
              <input
                type="text"
                value={protocol.metadata.title}
                onChange={(e) => updateMetadata('title', e.target.value)}
                placeholder="Descriptive protocol title"
                className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <div>
              <label className="block text-gray-200 font-medium mb-2">Principal Investigator *</label>
              <input
                type="text"
                value={protocol.metadata.investigator}
                onChange={(e) => updateMetadata('investigator', e.target.value)}
                placeholder="Your name"
                className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <div>
              <label className="block text-gray-200 font-medium mb-2">Institution</label>
              <input
                type="text"
                value={protocol.metadata.institution}
                onChange={(e) => updateMetadata('institution', e.target.value)}
                placeholder="University or organization"
                className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <div>
              <label className="block text-gray-200 font-medium mb-2">Date</label>
              <input
                type="date"
                value={protocol.metadata.date}
                onChange={(e) => updateMetadata('date', e.target.value)}
                className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-gray-100 focus:ring-2 focus:ring-indigo-500"
              />
            </div>
          </div>
        </div>

        {/* Validation Summary */}
        {validation.errors.length > 0 && (
          <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 mb-6">
            <div className="flex items-center gap-2 text-red-400 font-semibold mb-2">
              <AlertCircle className="w-5 h-5" />
              {validation.errors.length} Error(s) - Protocol Incomplete
            </div>
            <ul className="list-disc list-inside text-red-300 text-sm space-y-1">
              {validation.errors.slice(0, 5).map((err, i) => (
                <li key={i}>{err}</li>
              ))}
              {validation.errors.length > 5 && (
                <li className="text-red-400">... and {validation.errors.length - 5} more errors</li>
              )}
            </ul>
          </div>
        )}

        {validation.warnings.length > 0 && validation.errors.length === 0 && (
          <div className="bg-orange-900/30 border border-orange-700 rounded-lg p-4 mb-6">
            <div className="flex items-center gap-2 text-orange-400 font-semibold mb-2">
              <Info className="w-5 h-5" />
              {validation.warnings.length} Warning(s)
            </div>
            <ul className="list-disc list-inside text-orange-300 text-sm space-y-1">
              {validation.warnings.slice(0, 3).map((warn, i) => (
                <li key={i}>{warn}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Section Navigator */}
          <div className="lg:col-span-1">
            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50 sticky top-8">
              <h3 className="text-lg font-semibold text-gray-100 mb-4">Protocol Sections</h3>
              <div className="space-y-2">
                {sections.map(section => {
                  const status = getSectionStatus(section.id)

                  return (
                    <button
                      key={section.id}
                      onClick={() => setCurrentSection(section.id)}
                      className={`w-full text-left px-4 py-3 rounded-lg transition flex items-center gap-3 ${
                        currentSection === section.id
                          ? 'bg-indigo-600 text-white'
                          : 'bg-slate-700/50 text-gray-300 hover:bg-slate-700'
                      }`}
                    >
                      <span className="text-xl">{section.icon}</span>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium truncate">{section.label}</div>
                        {section.required && (
                          <div className="text-xs opacity-70">Required</div>
                        )}
                      </div>
                      {status.isComplete && <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0" />}
                      {status.hasErrors && <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />}
                    </button>
                  )
                })}
              </div>

              {/* Export Button */}
              <button
                onClick={handleExportPDF}
                disabled={validation.errors.length > 0 || isExporting}
                className="w-full mt-6 px-4 py-3 bg-gradient-to-r from-green-600 to-teal-600 hover:from-green-700 hover:to-teal-700 disabled:from-slate-600 disabled:to-slate-700 disabled:cursor-not-allowed text-white rounded-lg font-semibold flex items-center justify-center gap-2 transition"
              >
                <Download className="w-5 h-5" />
                {isExporting ? 'Generating...' : 'Export PDF Protocol'}
              </button>

              {protocol.blinding.type !== 'none' && (
                <p className="text-xs text-gray-400 mt-2 text-center">
                  Blinding key will be exported separately
                </p>
              )}
            </div>
          </div>

          {/* Section Content */}
          <div className="lg:col-span-3">
            <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
              {renderSectionContent()}

              {/* Navigation Buttons */}
              <div className="flex items-center justify-between mt-8 pt-6 border-t border-slate-700">
                <button
                  onClick={() => {
                    const currentIndex = sections.findIndex(s => s.id === currentSection)
                    if (currentIndex > 0) {
                      setCurrentSection(sections[currentIndex - 1].id)
                    }
                  }}
                  disabled={sections.findIndex(s => s.id === currentSection) === 0}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:cursor-not-allowed text-gray-100 rounded-lg transition"
                >
                  ← Previous
                </button>

                <button
                  onClick={() => {
                    const currentIndex = sections.findIndex(s => s.id === currentSection)
                    if (currentIndex < sections.length - 1) {
                      setCurrentSection(sections[currentIndex + 1].id)
                    }
                  }}
                  disabled={sections.findIndex(s => s.id === currentSection) === sections.length - 1}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-lg transition flex items-center gap-2"
                >
                  Next <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ProtocolGenerator
