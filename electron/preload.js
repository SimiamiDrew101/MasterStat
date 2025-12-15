/**
 * MasterStat Electron Preload Script
 * Provides secure bridge between main process and renderer
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to renderer process
contextBridge.exposeInMainWorld('electron', {
  // App info
  platform: process.platform,
  version: process.env.npm_package_version || '1.0.0',

  // IPC communication
  onNewAnalysis: (callback) => ipcRenderer.on('new-analysis', callback),
  onOpenData: (callback) => ipcRenderer.on('open-data', callback),
  onExportResults: (callback) => ipcRenderer.on('export-results', callback),

  // Utility
  isDevelopment: process.env.NODE_ENV === 'development',
});

console.log('MasterStat preload script loaded');
