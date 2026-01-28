/**
 * Session Manager - IndexedDB persistence using Dexie.js
 * Handles saving, loading, and managing analysis sessions
 */

import Dexie from 'dexie'

// Initialize IndexedDB database
const db = new Dexie('MasterStatDB')

// Define database schema
db.version(1).stores({
  sessions: '++id, name, timestamp, analysis_type, tags'
})

/**
 * Save a new session or update existing one
 * @param {Object} sessionData - Session data to save
 * @returns {Promise<number>} - Session ID
 */
export const saveSession = async (sessionData) => {
  try {
    const session = {
      name: sessionData.name || `Session ${new Date().toLocaleString()}`,
      timestamp: sessionData.timestamp || new Date().toISOString(),
      analysis_type: sessionData.analysis_type || 'Unknown',
      data: sessionData.data || {},
      results: sessionData.results || {},
      metadata: {
        version: '1.0',
        appVersion: '1.0.0',
        tags: sessionData.tags || [],
        ...sessionData.metadata
      }
    }

    // If updating existing session
    if (sessionData.id) {
      await db.sessions.update(sessionData.id, session)
      return sessionData.id
    }

    // Create new session
    const id = await db.sessions.add(session)
    return id
  } catch (error) {
    console.error('Failed to save session:', error)
    throw new Error(`Failed to save session: ${error.message}`)
  }
}

/**
 * Load a session by ID
 * @param {number} sessionId - Session ID to load
 * @returns {Promise<Object|null>} - Session data or null
 */
export const loadSession = async (sessionId) => {
  try {
    const session = await db.sessions.get(sessionId)
    return session || null
  } catch (error) {
    console.error('Failed to load session:', error)
    throw new Error(`Failed to load session: ${error.message}`)
  }
}

/**
 * Get all sessions sorted by timestamp (newest first)
 * @returns {Promise<Array>} - Array of sessions
 */
export const getAllSessions = async () => {
  try {
    const sessions = await db.sessions
      .orderBy('timestamp')
      .reverse()
      .toArray()
    return sessions
  } catch (error) {
    console.error('Failed to get sessions:', error)
    throw new Error(`Failed to get sessions: ${error.message}`)
  }
}

/**
 * Delete a session by ID
 * @param {number} sessionId - Session ID to delete
 * @returns {Promise<void>}
 */
export const deleteSession = async (sessionId) => {
  try {
    await db.sessions.delete(sessionId)
  } catch (error) {
    console.error('Failed to delete session:', error)
    throw new Error(`Failed to delete session: ${error.message}`)
  }
}

/**
 * Delete multiple sessions
 * @param {Array<number>} sessionIds - Array of session IDs to delete
 * @returns {Promise<void>}
 */
export const deleteSessions = async (sessionIds) => {
  try {
    await db.sessions.bulkDelete(sessionIds)
  } catch (error) {
    console.error('Failed to delete sessions:', error)
    throw new Error(`Failed to delete sessions: ${error.message}`)
  }
}

/**
 * Update session name
 * @param {number} sessionId - Session ID
 * @param {string} newName - New name
 * @returns {Promise<void>}
 */
export const updateSessionName = async (sessionId, newName) => {
  try {
    await db.sessions.update(sessionId, { name: newName })
  } catch (error) {
    console.error('Failed to update session name:', error)
    throw new Error(`Failed to update session name: ${error.message}`)
  }
}

/**
 * Export a session to JSON string
 * @param {number} sessionId - Session ID to export
 * @returns {Promise<string>} - JSON string
 */
export const exportSessionToJSON = async (sessionId) => {
  try {
    const session = await db.sessions.get(sessionId)
    if (!session) {
      throw new Error('Session not found')
    }

    // Remove internal ID for export
    const exportData = { ...session }
    delete exportData.id
    exportData.exportedAt = new Date().toISOString()

    return JSON.stringify(exportData, null, 2)
  } catch (error) {
    console.error('Failed to export session:', error)
    throw new Error(`Failed to export session: ${error.message}`)
  }
}

/**
 * Import a session from JSON string
 * @param {string} jsonString - JSON string to import
 * @returns {Promise<number>} - New session ID
 */
export const importSessionFromJSON = async (jsonString) => {
  try {
    const sessionData = JSON.parse(jsonString)

    // Validate required fields
    if (!sessionData.name || !sessionData.analysis_type) {
      throw new Error('Invalid session format: missing required fields')
    }

    // Update timestamp to import time
    sessionData.timestamp = new Date().toISOString()
    sessionData.metadata = {
      ...sessionData.metadata,
      importedAt: new Date().toISOString(),
      originalExportedAt: sessionData.exportedAt
    }
    delete sessionData.exportedAt
    delete sessionData.id // Remove any existing ID

    const id = await db.sessions.add(sessionData)
    return id
  } catch (error) {
    console.error('Failed to import session:', error)
    throw new Error(`Failed to import session: ${error.message}`)
  }
}

/**
 * Search sessions by name or analysis type
 * @param {string} query - Search query
 * @returns {Promise<Array>} - Matching sessions
 */
export const searchSessions = async (query) => {
  try {
    const lowerQuery = query.toLowerCase()
    const sessions = await db.sessions.toArray()

    return sessions.filter(session =>
      session.name.toLowerCase().includes(lowerQuery) ||
      session.analysis_type.toLowerCase().includes(lowerQuery) ||
      (session.metadata?.tags || []).some(tag =>
        tag.toLowerCase().includes(lowerQuery)
      )
    ).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
  } catch (error) {
    console.error('Failed to search sessions:', error)
    throw new Error(`Failed to search sessions: ${error.message}`)
  }
}

/**
 * Filter sessions by analysis type
 * @param {string} analysisType - Analysis type to filter by
 * @returns {Promise<Array>} - Filtered sessions
 */
export const filterByAnalysisType = async (analysisType) => {
  try {
    if (!analysisType || analysisType === 'all') {
      return getAllSessions()
    }

    const sessions = await db.sessions
      .where('analysis_type')
      .equals(analysisType)
      .reverse()
      .sortBy('timestamp')

    return sessions
  } catch (error) {
    console.error('Failed to filter sessions:', error)
    throw new Error(`Failed to filter sessions: ${error.message}`)
  }
}

/**
 * Get session count
 * @returns {Promise<number>} - Number of sessions
 */
export const getSessionCount = async () => {
  try {
    return await db.sessions.count()
  } catch (error) {
    console.error('Failed to get session count:', error)
    return 0
  }
}

/**
 * Clear all sessions (use with caution)
 * @returns {Promise<void>}
 */
export const clearAllSessions = async () => {
  try {
    await db.sessions.clear()
  } catch (error) {
    console.error('Failed to clear sessions:', error)
    throw new Error(`Failed to clear sessions: ${error.message}`)
  }
}

/**
 * Download session as JSON file
 * @param {number} sessionId - Session ID to download
 */
export const downloadSession = async (sessionId) => {
  try {
    const jsonString = await exportSessionToJSON(sessionId)
    const session = await db.sessions.get(sessionId)

    const blob = new Blob([jsonString], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${session.name.replace(/[^a-z0-9]/gi, '_')}_${session.analysis_type}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  } catch (error) {
    console.error('Failed to download session:', error)
    throw new Error(`Failed to download session: ${error.message}`)
  }
}

// Export database instance for advanced usage
export { db }

export default {
  saveSession,
  loadSession,
  getAllSessions,
  deleteSession,
  deleteSessions,
  updateSessionName,
  exportSessionToJSON,
  importSessionFromJSON,
  searchSessions,
  filterByAnalysisType,
  getSessionCount,
  clearAllSessions,
  downloadSession,
  db
}
