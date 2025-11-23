import { useState } from 'react'
import { Plus, Trash2, Check, X, Layers } from 'lucide-react'

const MultiResponseManager = ({ responseNames, onUpdate, disabled = false }) => {
  const [isOpen, setIsOpen] = useState(false)
  const [newResponseName, setNewResponseName] = useState('')
  const [error, setError] = useState('')

  const handleAddResponse = () => {
    setError('')

    // Validation
    if (!newResponseName.trim()) {
      setError('Response name cannot be empty')
      return
    }

    if (responseNames.includes(newResponseName.trim())) {
      setError('Response name already exists')
      return
    }

    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(newResponseName.trim())) {
      setError('Response name must start with a letter and contain only letters, numbers, and underscores')
      return
    }

    if (responseNames.length >= 5) {
      setError('Maximum 5 responses allowed for visualization clarity')
      return
    }

    // Add new response
    onUpdate([...responseNames, newResponseName.trim()])
    setNewResponseName('')
  }

  const handleRemoveResponse = (index) => {
    if (responseNames.length <= 1) {
      setError('Must have at least one response')
      return
    }

    const newNames = responseNames.filter((_, i) => i !== index)
    onUpdate(newNames)
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleAddResponse()
    }
  }

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        disabled={disabled}
        className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <Layers className="w-4 h-4" />
        Manage Responses ({responseNames.length})
      </button>
    )
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-slate-800 rounded-2xl border border-slate-700 shadow-2xl w-full max-w-2xl mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <Layers className="w-6 h-6 text-indigo-400" />
            <h2 className="text-2xl font-bold text-gray-100">Manage Response Variables</h2>
          </div>
          <button
            onClick={() => {
              setIsOpen(false)
              setError('')
            }}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Current Responses */}
          <div>
            <h3 className="text-lg font-semibold text-gray-200 mb-3">Current Responses</h3>
            <div className="space-y-2">
              {responseNames.map((name, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg border border-slate-600"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white font-semibold">
                      {index + 1}
                    </div>
                    <span className="text-gray-100 font-mono font-medium">{name}</span>
                  </div>
                  <button
                    onClick={() => handleRemoveResponse(index)}
                    disabled={responseNames.length <= 1}
                    className="p-2 hover:bg-red-900/30 text-red-400 hover:text-red-300 rounded transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                    title={responseNames.length <= 1 ? "Must have at least one response" : "Remove response"}
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Add New Response */}
          {responseNames.length < 5 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-200 mb-3">Add New Response</h3>
              <div className="flex gap-2">
                <div className="flex-1">
                  <input
                    type="text"
                    value={newResponseName}
                    onChange={(e) => {
                      setNewResponseName(e.target.value)
                      setError('')
                    }}
                    onKeyPress={handleKeyPress}
                    placeholder="e.g., Y2, Yield, Cost"
                    className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-gray-100 placeholder-gray-500 focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 transition-all"
                  />
                  {error && (
                    <p className="mt-2 text-sm text-red-400">{error}</p>
                  )}
                </div>
                <button
                  onClick={handleAddResponse}
                  className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors flex items-center gap-2 font-medium"
                >
                  <Plus className="w-4 h-4" />
                  Add
                </button>
              </div>
            </div>
          )}

          {/* Info Box */}
          <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
            <h4 className="text-blue-300 font-semibold mb-2">Multi-Response Analysis</h4>
            <ul className="text-sm text-blue-100 space-y-1">
              <li>• Multiple responses allow comparison of different outcomes</li>
              <li>• Responses can be overlaid on the same contour plot</li>
              <li>• Useful for multi-objective optimization (e.g., maximize yield, minimize cost)</li>
              <li>• Maximum 5 responses recommended for visualization clarity</li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-6 border-t border-slate-700 bg-slate-800/50">
          <button
            onClick={() => {
              setIsOpen(false)
              setError('')
              setNewResponseName('')
            }}
            className="px-6 py-2 bg-slate-700 hover:bg-slate-600 text-gray-200 rounded-lg transition-colors flex items-center gap-2"
          >
            <Check className="w-4 h-4" />
            Done
          </button>
        </div>
      </div>
    </div>
  )
}

export default MultiResponseManager
