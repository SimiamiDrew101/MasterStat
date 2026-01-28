/**
 * Session History Component - Browse, manage, and restore saved sessions
 */

import { useState, useEffect, useRef } from 'react'
import {
  History,
  Search,
  Filter,
  Upload,
  Download,
  Trash2,
  Edit2,
  FolderOpen,
  X,
  Clock,
  Database,
  Check,
  AlertCircle,
  FileJson
} from 'lucide-react'
import { useSession } from '../contexts/SessionContext'

// Analysis type colors
const analysisTypeColors = {
  RSM: 'bg-purple-600',
  ANOVA: 'bg-blue-600',
  Factorial: 'bg-green-600',
  MixedModels: 'bg-orange-600',
  Nonlinear: 'bg-pink-600',
  BlockDesign: 'bg-cyan-600',
  QualityControl: 'bg-yellow-600',
  Unknown: 'bg-slate-600'
}

// Format relative time
const formatRelativeTime = (timestamp) => {
  const now = new Date()
  const date = new Date(timestamp)
  const diffMs = now - date
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins} min ago`
  if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`
  if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`
  return date.toLocaleDateString()
}

const SessionHistory = ({ isOpen, onClose, onLoadSession }) => {
  const {
    savedSessions,
    loading,
    error,
    removeSession,
    renameSession,
    exportSession,
    importSession,
    search,
    filterByType,
    clearError,
    refreshSessions
  } = useSession()

  // Local state
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState('all')
  const [filteredSessions, setFilteredSessions] = useState([])
  const [editingId, setEditingId] = useState(null)
  const [editName, setEditName] = useState('')
  const [confirmDelete, setConfirmDelete] = useState(null)
  const [importError, setImportError] = useState(null)

  const fileInputRef = useRef(null)

  // Update filtered sessions when search or filter changes
  useEffect(() => {
    const updateFiltered = async () => {
      let results = savedSessions

      if (filterType !== 'all') {
        results = await filterByType(filterType)
      }

      if (searchQuery.trim()) {
        results = await search(searchQuery)
        if (filterType !== 'all') {
          results = results.filter(s => s.analysis_type === filterType)
        }
      }

      setFilteredSessions(results)
    }

    updateFiltered()
  }, [savedSessions, searchQuery, filterType, search, filterByType])

  // Handle load session
  const handleLoad = async (session) => {
    if (onLoadSession) {
      onLoadSession(session)
    }
    onClose()
  }

  // Handle rename
  const handleStartRename = (session) => {
    setEditingId(session.id)
    setEditName(session.name)
  }

  const handleSaveRename = async () => {
    if (editingId && editName.trim()) {
      try {
        await renameSession(editingId, editName.trim())
        setEditingId(null)
        setEditName('')
      } catch (err) {
        console.error('Rename failed:', err)
      }
    }
  }

  const handleCancelRename = () => {
    setEditingId(null)
    setEditName('')
  }

  // Handle delete
  const handleDelete = async (sessionId) => {
    try {
      await removeSession(sessionId)
      setConfirmDelete(null)
    } catch (err) {
      console.error('Delete failed:', err)
    }
  }

  // Handle export
  const handleExport = async (sessionId) => {
    try {
      await exportSession(sessionId)
    } catch (err) {
      console.error('Export failed:', err)
    }
  }

  // Handle import
  const handleImport = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return

    try {
      setImportError(null)
      await importSession(file)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    } catch (err) {
      setImportError(err.message)
    }
  }

  // Get unique analysis types for filter
  const analysisTypes = [...new Set(savedSessions.map(s => s.analysis_type))]

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-slate-800 rounded-2xl border border-slate-700 shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <History className="w-6 h-6 text-indigo-400" />
            <h2 className="text-2xl font-bold text-gray-100">Session History</h2>
            <span className="px-2 py-1 bg-slate-700 rounded-full text-sm text-gray-300">
              {savedSessions.length} sessions
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Search and Filter Bar */}
        <div className="p-4 border-b border-slate-700 bg-slate-800/50">
          <div className="flex flex-wrap gap-4">
            {/* Search */}
            <div className="flex-1 min-w-[200px] relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search sessions..."
                className="w-full pl-10 pr-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-gray-100 placeholder-gray-500 focus:outline-none focus:border-indigo-500"
              />
            </div>

            {/* Filter by type */}
            <div className="relative">
              <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="pl-10 pr-8 py-2 bg-slate-700 border border-slate-600 rounded-lg text-gray-100 focus:outline-none focus:border-indigo-500 appearance-none cursor-pointer"
              >
                <option value="all">All Types</option>
                {analysisTypes.map(type => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
            </div>

            {/* Import button */}
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-gray-200 rounded-lg transition-colors"
            >
              <Upload className="w-4 h-4" />
              Import
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleImport}
              className="hidden"
            />
          </div>

          {/* Import error */}
          {importError && (
            <div className="mt-3 p-3 bg-red-900/30 border border-red-700 rounded-lg flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-400" />
              <span className="text-red-200 text-sm">{importError}</span>
              <button
                onClick={() => setImportError(null)}
                className="ml-auto p-1 hover:bg-red-800/30 rounded"
              >
                <X className="w-4 h-4 text-red-400" />
              </button>
            </div>
          )}
        </div>

        {/* Session List */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading && (
            <div className="text-center py-8 text-gray-400">
              Loading sessions...
            </div>
          )}

          {error && (
            <div className="p-4 bg-red-900/30 border border-red-700 rounded-lg mb-4">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-red-400" />
                <span className="text-red-200">{error}</span>
              </div>
              <button
                onClick={clearError}
                className="mt-2 text-sm text-red-300 hover:text-red-200"
              >
                Dismiss
              </button>
            </div>
          )}

          {!loading && filteredSessions.length === 0 && (
            <div className="text-center py-12">
              <Database className="w-12 h-12 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">
                {searchQuery || filterType !== 'all'
                  ? 'No sessions match your search'
                  : 'No saved sessions yet'}
              </p>
              <p className="text-gray-500 text-sm mt-2">
                Save your analysis to access it later
              </p>
            </div>
          )}

          <div className="space-y-3">
            {filteredSessions.map((session) => (
              <div
                key={session.id}
                className="bg-slate-700/50 rounded-xl p-4 border border-slate-600 hover:border-slate-500 transition-all"
              >
                <div className="flex items-start justify-between gap-4">
                  {/* Session info */}
                  <div className="flex-1 min-w-0">
                    {/* Name / Edit */}
                    {editingId === session.id ? (
                      <div className="flex items-center gap-2 mb-2">
                        <input
                          type="text"
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSaveRename()
                            if (e.key === 'Escape') handleCancelRename()
                          }}
                          className="flex-1 px-3 py-1 bg-slate-800 border border-indigo-500 rounded text-gray-100 focus:outline-none"
                          autoFocus
                        />
                        <button
                          onClick={handleSaveRename}
                          className="p-1 bg-green-600 hover:bg-green-700 rounded"
                        >
                          <Check className="w-4 h-4 text-white" />
                        </button>
                        <button
                          onClick={handleCancelRename}
                          className="p-1 bg-slate-600 hover:bg-slate-500 rounded"
                        >
                          <X className="w-4 h-4 text-white" />
                        </button>
                      </div>
                    ) : (
                      <h3 className="text-lg font-semibold text-gray-100 truncate mb-1">
                        {session.name}
                      </h3>
                    )}

                    {/* Metadata */}
                    <div className="flex flex-wrap items-center gap-3 text-sm">
                      <span className={`px-2 py-0.5 rounded text-white text-xs font-medium ${
                        analysisTypeColors[session.analysis_type] || analysisTypeColors.Unknown
                      }`}>
                        {session.analysis_type}
                      </span>
                      <span className="flex items-center gap-1 text-gray-400">
                        <Clock className="w-3 h-3" />
                        {formatRelativeTime(session.timestamp)}
                      </span>
                      {session.data?.factors && (
                        <span className="text-gray-500">
                          {session.data.factors.length} factors
                        </span>
                      )}
                      {session.data?.responses && (
                        <span className="text-gray-500">
                          {Array.isArray(session.data.responses)
                            ? session.data.responses.length
                            : 1} response(s)
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleLoad(session)}
                      className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors"
                    >
                      <FolderOpen className="w-4 h-4" />
                      Load
                    </button>
                    <button
                      onClick={() => handleStartRename(session)}
                      className="p-2 hover:bg-slate-600 text-gray-400 hover:text-gray-200 rounded-lg transition-colors"
                      title="Rename"
                    >
                      <Edit2 className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleExport(session.id)}
                      className="p-2 hover:bg-slate-600 text-gray-400 hover:text-gray-200 rounded-lg transition-colors"
                      title="Export"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                    {confirmDelete === session.id ? (
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => handleDelete(session.id)}
                          className="p-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                          title="Confirm delete"
                        >
                          <Check className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setConfirmDelete(null)}
                          className="p-2 bg-slate-600 hover:bg-slate-500 text-white rounded-lg transition-colors"
                          title="Cancel"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setConfirmDelete(session.id)}
                        className="p-2 hover:bg-red-900/30 text-gray-400 hover:text-red-400 rounded-lg transition-colors"
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-700 bg-slate-800/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <FileJson className="w-4 h-4" />
              <span>Sessions are stored locally in your browser</span>
            </div>
            <button
              onClick={refreshSessions}
              className="px-4 py-2 text-sm text-gray-300 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SessionHistory
