import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import Home from './pages/Home'
import ExperimentWizardPage from './pages/ExperimentWizardPage'
import HypothesisTesting from './pages/HypothesisTesting'
import ANOVA from './pages/ANOVA'
import FactorialDesigns from './pages/FactorialDesigns'
import BlockDesigns from './pages/BlockDesigns'
import MixedModels from './pages/MixedModels'
import RSM from './pages/RSM'
import MixtureDesign from './pages/MixtureDesign'
import RobustDesign from './pages/RobustDesign'
import BayesianDOE from './pages/BayesianDOE'
import ExperimentPlanning from './pages/ExperimentPlanning'
import DataPreprocessing from './pages/DataPreprocessing'
import ProtocolGeneratorPage from './pages/ProtocolGeneratorPage'
import PredictionProfiler from './pages/PredictionProfiler'
import OptimalDesigns from './pages/OptimalDesigns'
import NonlinearRegression from './pages/NonlinearRegression'
import QualityControl from './pages/QualityControl'
import ReliabilityAnalysis from './pages/ReliabilityAnalysis'
import GLM from './pages/GLM'
import CustomDesign from './pages/CustomDesign'
import PredictiveModeling from './pages/PredictiveModeling'
import SessionHistory from './components/SessionHistory'
import { SessionProvider, useSession } from './contexts/SessionContext'
import { Menu, X, BarChart3, History } from 'lucide-react'

// Inner App component that uses session context
function AppContent() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [showSessionHistory, setShowSessionHistory] = useState(false)
  const { sessionCount } = useSession()

  const menuItems = [
    { path: '/', label: 'Home', icon: '🏠' },
    { path: '/experiment-wizard', label: 'Experiment Wizard', icon: '✨' },
    { path: '/planning', label: 'Experiment Planning', icon: '🎯' },
    { path: '/preprocessing', label: 'Data Preprocessing', icon: '🔧' },
    { path: '/protocol-generator', label: 'Protocol Generator', icon: '📋' },
    { path: '/hypothesis', label: 'Hypothesis Testing', icon: '📊' },
    { path: '/anova', label: 'ANOVA', icon: '📈' },
    { path: '/factorial', label: 'Factorial Designs', icon: '🔬' },
    { path: '/blocks', label: 'Block Designs', icon: '🧱' },
    { path: '/mixed', label: 'Mixed Models', icon: '🔀' },
    { path: '/rsm', label: 'Response Surface', icon: '🗻' },
    { path: '/prediction-profiler', label: 'Prediction Profiler', icon: '🎯' },
    { path: '/optimal-designs', label: 'Optimal Designs', icon: '✨' },
    { path: '/nonlinear-regression', label: 'Nonlinear Regression', icon: '📉' },
    { path: '/quality-control', label: 'Quality Control', icon: '🎯' },
    { path: '/reliability', label: 'Reliability Analysis', icon: '⏱️' },
    { path: '/glm', label: 'Generalized Linear Models', icon: '📊' },
    { path: '/mixture', label: 'Mixture Designs', icon: '💧' },
    { path: '/robust', label: 'Robust Design', icon: '🛡️' },
    { path: '/bayesian-doe', label: 'Bayesian DOE', icon: '🎲' },
    { path: '/custom-design', label: 'Custom Design', icon: '⚙️' },
    { path: '/predictive-modeling', label: 'Predictive Modeling', icon: '🤖' },
  ]

  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        {/* Header */}
        <header className="bg-slate-800/50 backdrop-blur-lg border-b border-slate-700/50 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <Link to="/" className="flex items-center space-x-3 hover:opacity-80 transition-opacity">
                <BarChart3 className="w-8 h-8 text-blue-400" />
                <h1 className="text-2xl font-bold text-gray-100">MasterStat</h1>
              </Link>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setShowSessionHistory(true)}
                  className="flex items-center gap-2 px-3 py-2 bg-slate-700/50 hover:bg-slate-600/50 text-gray-200 rounded-lg transition-colors"
                  title="Session History"
                >
                  <History className="w-5 h-5" />
                  <span className="hidden sm:inline">Sessions</span>
                  {sessionCount > 0 && (
                    <span className="px-1.5 py-0.5 text-xs bg-indigo-600 rounded-full">
                      {sessionCount}
                    </span>
                  )}
                </button>
                <button
                  onClick={() => setSidebarOpen(!sidebarOpen)}
                  className="lg:hidden text-gray-200 p-2 rounded-md hover:bg-slate-700/50"
                >
                  {sidebarOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                </button>
              </div>
            </div>
          </div>
        </header>

        <div className="flex">
          {/* Backdrop overlay for mobile sidebar */}
          {sidebarOpen && (
            <div
              className="fixed inset-0 bg-black/50 z-30 lg:hidden mt-16"
              onClick={() => setSidebarOpen(false)}
            />
          )}

          {/* Sidebar */}
          <aside
            className={`${
              sidebarOpen ? 'translate-x-0' : '-translate-x-full'
            } lg:translate-x-0 fixed lg:static inset-y-0 left-0 z-40 w-64 bg-slate-800/50 backdrop-blur-lg border-r border-slate-700/50 transition-transform duration-300 ease-in-out mt-16 lg:mt-0`}
          >
            <nav className="p-4 space-y-2">
              {menuItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={() => setSidebarOpen(false)}
                  className="flex items-center space-x-3 px-4 py-3 rounded-lg text-gray-200 hover:bg-slate-700/50 transition-all duration-200"
                >
                  <span className="text-2xl">{item.icon}</span>
                  <span className="font-medium">{item.label}</span>
                </Link>
              ))}
            </nav>
          </aside>

          {/* Main Content */}
          <main className="flex-1 p-6 lg:p-8">
            <div className="max-w-6xl mx-auto">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/experiment-wizard" element={<ExperimentWizardPage />} />
                <Route path="/planning" element={<ExperimentPlanning />} />
                <Route path="/preprocessing" element={<DataPreprocessing />} />
                <Route path="/protocol-generator" element={<ProtocolGeneratorPage />} />
                <Route path="/hypothesis" element={<HypothesisTesting />} />
                <Route path="/anova" element={<ANOVA />} />
                <Route path="/factorial" element={<FactorialDesigns />} />
                <Route path="/blocks" element={<BlockDesigns />} />
                <Route path="/mixed" element={<MixedModels />} />
                <Route path="/rsm" element={<RSM />} />
                <Route path="/prediction-profiler" element={<PredictionProfiler />} />
                <Route path="/optimal-designs" element={<OptimalDesigns />} />
                <Route path="/nonlinear-regression" element={<NonlinearRegression />} />
                <Route path="/quality-control" element={<QualityControl />} />
                <Route path="/reliability" element={<ReliabilityAnalysis />} />
                <Route path="/glm" element={<GLM />} />
                <Route path="/mixture" element={<MixtureDesign />} />
                <Route path="/robust" element={<RobustDesign />} />
                <Route path="/bayesian-doe" element={<BayesianDOE />} />
                <Route path="/custom-design" element={<CustomDesign />} />
                <Route path="/predictive-modeling" element={<PredictiveModeling />} />
              </Routes>
            </div>
          </main>
        </div>

        {/* Session History Modal */}
        <SessionHistory
          isOpen={showSessionHistory}
          onClose={() => setShowSessionHistory(false)}
          onLoadSession={(session) => {
            console.log('Load session:', session)
            // Navigation to appropriate page will be handled by the page component
          }}
        />
      </div>
    </Router>
  )
}

// Main App component wrapped with SessionProvider
function App() {
  return (
    <SessionProvider>
      <AppContent />
    </SessionProvider>
  )
}

export default App
