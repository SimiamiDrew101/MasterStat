/**
 * MasterStat Electron Main Process
 * Manages application lifecycle and Python backend
 */

const { app, BrowserWindow, Menu, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const waitOn = require('wait-on');

let mainWindow;
let backendProcess;
const BACKEND_PORT = 8000;
const FRONTEND_PORT = 5173;

/**
 * Start Python backend server
 */
function startBackend() {
  console.log('Starting Python backend...');

  // Determine Python executable path
  const pythonPath = process.platform === 'win32' ? 'python' : 'python3';
  const backendPath = path.join(__dirname, '..', 'backend');

  // Spawn Python backend
  backendProcess = spawn(pythonPath, [
    '-m', 'uvicorn',
    'app.main:app',
    '--host', '127.0.0.1',
    '--port', BACKEND_PORT.toString(),
  ], {
    cwd: backendPath,
    env: { ...process.env }
  });

  backendProcess.stdout.on('data', (data) => {
    console.log(`[Backend] ${data}`);
  });

  backendProcess.stderr.on('data', (data) => {
    console.error(`[Backend Error] ${data}`);
  });

  backendProcess.on('close', (code) => {
    console.log(`Backend process exited with code ${code}`);
    if (code !== 0 && !app.isQuitting) {
      // Backend crashed unexpectedly
      dialog.showErrorBox(
        'Backend Error',
        'The MasterStat backend has stopped unexpectedly. The application will now quit.'
      );
      app.quit();
    }
  });

  return new Promise((resolve) => {
    // Wait for backend to be ready
    waitOn({
      resources: [`http://127.0.0.1:${BACKEND_PORT}/health`],
      timeout: 30000,
      interval: 500,
    })
      .then(() => {
        console.log('Backend is ready!');
        resolve();
      })
      .catch((err) => {
        console.error('Backend failed to start:', err);
        dialog.showErrorBox(
          'Startup Error',
          'Failed to start MasterStat backend. Please check Python installation.'
        );
        app.quit();
      });
  });
}

/**
 * Create main application window
 */
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 768,
    title: 'MasterStat - Statistical Analysis & DOE',
    backgroundColor: '#1e293b', // Match app dark theme
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    show: false, // Don't show until ready
  });

  // Load the app from the backend (which serves the frontend)
  const isDev = process.env.NODE_ENV === 'development';

  if (isDev && process.env.ELECTRON_START_URL) {
    // Development with separate Vite server: load from Vite dev server
    mainWindow.loadURL(process.env.ELECTRON_START_URL);
  } else {
    // Production or development without separate server: load from backend
    // Backend serves the frontend at http://127.0.0.1:8000
    mainWindow.loadURL(`http://127.0.0.1:${BACKEND_PORT}`);
  }

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();

    // Open DevTools in development
    if (process.env.NODE_ENV === 'development') {
      mainWindow.webContents.openDevTools();
    }
  });

  // Handle window close
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Create menu
  createMenu();
}

/**
 * Create application menu
 */
function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'New Analysis',
          accelerator: 'CmdOrCtrl+N',
          click: () => {
            mainWindow.webContents.send('new-analysis');
          }
        },
        {
          label: 'Open Data...',
          accelerator: 'CmdOrCtrl+O',
          click: () => {
            mainWindow.webContents.send('open-data');
          }
        },
        { type: 'separator' },
        {
          label: 'Export Results...',
          accelerator: 'CmdOrCtrl+E',
          click: () => {
            mainWindow.webContents.send('export-results');
          }
        },
        { type: 'separator' },
        {
          label: 'Quit',
          accelerator: 'CmdOrCtrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectAll' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'Documentation',
          click: async () => {
            const { shell } = require('electron');
            await shell.openExternal('https://github.com/SimiamiDrew101/MasterStat');
          }
        },
        {
          label: 'Report Issue',
          click: async () => {
            const { shell } = require('electron');
            await shell.openExternal('https://github.com/SimiamiDrew101/MasterStat/issues');
          }
        },
        { type: 'separator' },
        {
          label: 'About MasterStat',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About MasterStat',
              message: 'MasterStat v1.0.0',
              detail: 'Professional-grade statistical analysis and Design of Experiments platform\n\nFree & Open Source\nLicensed under CC BY 4.0',
              buttons: ['OK']
            });
          }
        }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

/**
 * App lifecycle: Ready
 */
app.whenReady().then(async () => {
  console.log('Electron app ready, starting backend...');

  // Start backend first
  await startBackend();

  // Then create window
  createWindow();

  app.on('activate', () => {
    // On macOS, re-create window when dock icon is clicked
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

/**
 * App lifecycle: All windows closed
 */
app.on('window-all-closed', () => {
  // On macOS, apps usually stay active until Cmd+Q
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

/**
 * App lifecycle: Quit
 */
app.on('before-quit', () => {
  app.isQuitting = true;

  // Kill Python backend
  if (backendProcess) {
    console.log('Stopping Python backend...');
    backendProcess.kill('SIGTERM');
  }
});

/**
 * Handle uncaught exceptions
 */
process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
  dialog.showErrorBox('Application Error', error.message);
});
