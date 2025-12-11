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
import { Menu, X, BarChart3 } from 'lucide-react'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

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
    { path: '/mixture', label: 'Mixture Designs', icon: '💧' },
    { path: '/robust', label: 'Robust Design', icon: '🛡️' },
    { path: '/bayesian-doe', label: 'Bayesian DOE', icon: '🎲' },
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
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden text-gray-200 p-2 rounded-md hover:bg-slate-700/50"
              >
                {sidebarOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>
          </div>
        </header>

        <div className="flex">
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
                <Route path="/mixture" element={<MixtureDesign />} />
                <Route path="/robust" element={<RobustDesign />} />
                <Route path="/bayesian-doe" element={<BayesianDOE />} />
              </Routes>
            </div>
          </main>
        </div>
      </div>
    </Router>
  )
}

export default App
