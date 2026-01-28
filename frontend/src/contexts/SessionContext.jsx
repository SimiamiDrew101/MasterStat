/**
 * Session Context - Global state management for session persistence
 */

import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import {
  saveSession,
  loadSession,
  getAllSessions,
  deleteSession,
  updateSessionName,
  downloadSession,
  importSessionFromJSON,
  searchSessions,
  filterByAnalysisType,
  getSessionCount
} from '../utils/sessionManager'

// Create context
const SessionContext = createContext(null)

/**
 * Session Provider Component
 */
export const SessionProvider = ({ children }) => {
  // State
  const [currentSession, setCurrentSession] = useState(null)
  const [savedSessions, setSavedSessions] = useState([])
  const [isSessionLoaded, setIsSessionLoaded] = useState(false)
  const [sessionCount, setSessionCount] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Load all sessions on mount
  useEffect(() => {
    loadAllSessions()
  }, [])

  /**
   * Load all sessions from IndexedDB
   */
  const loadAllSessions = useCallback(async () => {
    try {
      setLoading(true)
      const sessions = await getAllSessions()
      setSavedSessions(sessions)
      const count = await getSessionCount()
      setSessionCount(count)
      setError(null)
    } catch (err) {
      console.error('Failed to load sessions:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  /**
   * Save current analysis as a session
   * @param {string} name - Session name
   * @param {Object} sessionData - Session data including analysis_type, data, results
   * @returns {Promise<number>} - Session ID
   */
  const saveCurrentSession = useCallback(async (name, sessionData) => {
    try {
      setLoading(true)
      setError(null)

      const sessionId = await saveSession({
        name,
        timestamp: new Date().toISOString(),
        analysis_type: sessionData.analysis_type,
        data: sessionData.data,
        results: sessionData.results,
        metadata: sessionData.metadata
      })

      // Refresh session list
      await loadAllSessions()

      // Update current session reference
      const savedSession = await loadSession(sessionId)
      setCurrentSession(savedSession)
      setIsSessionLoaded(true)

      return sessionId
    } catch (err) {
      console.error('Failed to save session:', err)
      setError(err.message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [loadAllSessions])

  /**
   * Load a session by ID
   * @param {number} sessionId - Session ID to load
   * @returns {Promise<Object>} - Loaded session
   */
  const loadSessionById = useCallback(async (sessionId) => {
    try {
      setLoading(true)
      setError(null)

      const session = await loadSession(sessionId)
      if (session) {
        setCurrentSession(session)
        setIsSessionLoaded(true)
        return session
      } else {
        throw new Error('Session not found')
      }
    } catch (err) {
      console.error('Failed to load session:', err)
      setError(err.message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  /**
   * Delete a session
   * @param {number} sessionId - Session ID to delete
   */
  const removeSession = useCallback(async (sessionId) => {
    try {
      setLoading(true)
      setError(null)

      await deleteSession(sessionId)

      // If deleting current session, clear it
      if (currentSession && currentSession.id === sessionId) {
        setCurrentSession(null)
        setIsSessionLoaded(false)
      }

      // Refresh session list
      await loadAllSessions()
    } catch (err) {
      console.error('Failed to delete session:', err)
      setError(err.message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [currentSession, loadAllSessions])

  /**
   * Rename a session
   * @param {number} sessionId - Session ID
   * @param {string} newName - New name
   */
  const renameSession = useCallback(async (sessionId, newName) => {
    try {
      setLoading(true)
      setError(null)

      await updateSessionName(sessionId, newName)

      // Update current session if it's the one being renamed
      if (currentSession && currentSession.id === sessionId) {
        setCurrentSession(prev => ({ ...prev, name: newName }))
      }

      // Refresh session list
      await loadAllSessions()
    } catch (err) {
      console.error('Failed to rename session:', err)
      setError(err.message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [currentSession, loadAllSessions])

  /**
   * Export a session to JSON file
   * @param {number} sessionId - Session ID to export
   */
  const exportSession = useCallback(async (sessionId) => {
    try {
      setLoading(true)
      setError(null)
      await downloadSession(sessionId)
    } catch (err) {
      console.error('Failed to export session:', err)
      setError(err.message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  /**
   * Import a session from JSON file
   * @param {File} file - JSON file to import
   * @returns {Promise<number>} - New session ID
   */
  const importSession = useCallback(async (file) => {
    try {
      setLoading(true)
      setError(null)

      const text = await file.text()
      const sessionId = await importSessionFromJSON(text)

      // Refresh session list
      await loadAllSessions()

      return sessionId
    } catch (err) {
      console.error('Failed to import session:', err)
      setError(err.message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [loadAllSessions])

  /**
   * Search sessions
   * @param {string} query - Search query
   * @returns {Promise<Array>} - Matching sessions
   */
  const search = useCallback(async (query) => {
    try {
      if (!query || query.trim() === '') {
        return savedSessions
      }
      return await searchSessions(query)
    } catch (err) {
      console.error('Failed to search sessions:', err)
      return savedSessions
    }
  }, [savedSessions])

  /**
   * Filter sessions by analysis type
   * @param {string} analysisType - Analysis type
   * @returns {Promise<Array>} - Filtered sessions
   */
  const filterByType = useCallback(async (analysisType) => {
    try {
      return await filterByAnalysisType(analysisType)
    } catch (err) {
      console.error('Failed to filter sessions:', err)
      return savedSessions
    }
  }, [savedSessions])

  /**
   * Clear current session
   */
  const clearCurrentSession = useCallback(() => {
    setCurrentSession(null)
    setIsSessionLoaded(false)
  }, [])

  /**
   * Clear error
   */
  const clearError = useCallback(() => {
    setError(null)
  }, [])

  // Context value
  const value = {
    // State
    currentSession,
    savedSessions,
    isSessionLoaded,
    sessionCount,
    loading,
    error,

    // Actions
    saveCurrentSession,
    loadSessionById,
    removeSession,
    renameSession,
    exportSession,
    importSession,
    search,
    filterByType,
    clearCurrentSession,
    clearError,
    refreshSessions: loadAllSessions
  }

  return (
    <SessionContext.Provider value={value}>
      {children}
    </SessionContext.Provider>
  )
}

/**
 * Hook to use session context
 * @returns {Object} - Session context value
 */
export const useSession = () => {
  const context = useContext(SessionContext)
  if (!context) {
    throw new Error('useSession must be used within a SessionProvider')
  }
  return context
}

export default SessionContext
