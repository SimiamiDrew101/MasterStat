import { CheckCircle, XCircle } from 'lucide-react'
import BoxPlot from './BoxPlot'
import MeansPlot from './MeansPlot'
import ResidualPlots from './ResidualPlots'
import FDistributionPlot from './FDistributionPlot'
import InteractionPlot from './InteractionPlot'
import ParetoChart from './ParetoChart'
import MainEffectsPlot from './MainEffectsPlot'
import CubePlot from './CubePlot'

const ResultCard = ({ result }) => {
  if (!result) return null

  const renderValue = (value, decimals = 4) => {
    if (value === null || value === undefined) return 'N/A'
    if (typeof value === 'number') return value.toFixed(decimals)
    return value
  }

  const renderTable = (tableData) => {
    if (!tableData) return null

    if (Array.isArray(tableData.source)) {
      // ANOVA-style table
      return (
        <div className="overflow-x-auto">
          <table className="w-full text-gray-100">
            <thead>
              <tr className="border-b border-slate-600">
                <th className="text-left py-2 px-4">Source</th>
                <th className="text-right py-2 px-4">SS</th>
                <th className="text-right py-2 px-4">df</th>
                <th className="text-right py-2 px-4">MS</th>
                <th className="text-right py-2 px-4">F</th>
                <th className="text-right py-2 px-4">p-value</th>
              </tr>
            </thead>
            <tbody>
              {tableData.source.map((source, idx) => (
                <tr key={idx} className="border-b border-slate-700/30">
                  <td className="py-2 px-4">{source}</td>
                  <td className="text-right py-2 px-4">{renderValue(tableData.ss[idx])}</td>
                  <td className="text-right py-2 px-4">{tableData.df[idx]}</td>
                  <td className="text-right py-2 px-4">{renderValue(tableData.ms[idx])}</td>
                  <td className="text-right py-2 px-4">{renderValue(tableData.f[idx])}</td>
                  <td className="text-right py-2 px-4">{renderValue(tableData.p[idx], 6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
    } else if (typeof tableData === 'object') {
      // Object-based table
      return (
        <div className="overflow-x-auto">
          <table className="w-full text-gray-100">
            <thead>
              <tr className="border-b border-slate-600">
                <th className="text-left py-2 px-4">Source</th>
                <th className="text-right py-2 px-4">SS</th>
                <th className="text-right py-2 px-4">df</th>
                {Object.values(tableData)[0]?.ms !== undefined && (
                  <th className="text-right py-2 px-4">MS</th>
                )}
                {Object.values(tableData)[0]?.F !== undefined && (
                  <th className="text-right py-2 px-4">F</th>
                )}
                {Object.values(tableData)[0]?.p_value !== undefined && (
                  <th className="text-right py-2 px-4">p-value</th>
                )}
                {Object.values(tableData)[0]?.significant !== undefined && (
                  <th className="text-right py-2 px-4">Significant</th>
                )}
              </tr>
            </thead>
            <tbody>
              {Object.entries(tableData).map(([source, values]) => (
                <tr key={source} className="border-b border-slate-700/30">
                  <td className="py-2 px-4">{source}</td>
                  <td className="text-right py-2 px-4">{renderValue(values.sum_sq)}</td>
                  <td className="text-right py-2 px-4">{values.df}</td>
                  {values.ms !== undefined && (
                    <td className="text-right py-2 px-4">{renderValue(values.ms)}</td>
                  )}
                  {values.F !== undefined && (
                    <td className="text-right py-2 px-4">{renderValue(values.F)}</td>
                  )}
                  {values.p_value !== undefined && (
                    <td className="text-right py-2 px-4">{renderValue(values.p_value, 6)}</td>
                  )}
                  {values.significant !== undefined && (
                    <td className="text-right py-2 px-4">
                      {values.significant ? '✓' : '✗'}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
    }
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50 space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-2xl font-bold text-gray-100">Analysis Results</h3>
        {result.reject_null !== undefined && (
          <div className={`flex items-center space-x-2 px-4 py-2 rounded-full ${
            result.reject_null ? 'bg-green-500/30' : 'bg-blue-500/30'
          }`}>
            {result.reject_null ? (
              <CheckCircle className="w-5 h-5 text-green-300" />
            ) : (
              <XCircle className="w-5 h-5 text-blue-300" />
            )}
            <span className="text-gray-100 font-medium">
              {result.reject_null ? 'Reject H₀' : 'Fail to Reject H₀'}
            </span>
          </div>
        )}
      </div>

      {/* Interpretation and Recommendations */}
      {result.interpretation && (
        <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-xl p-5 border border-purple-500/30">
          <h4 className="text-gray-100 font-bold text-lg mb-3 flex items-center">
            <span className="mr-2">📊</span>
            Interpretation & Recommendations
          </h4>

          {/* Summary */}
          <div className="mb-4 p-4 bg-slate-800/50 rounded-lg">
            <p className="text-gray-100 text-sm leading-relaxed">
              {result.interpretation.summary}
            </p>
          </div>

          {/* Recommendations */}
          {result.interpretation.recommendations && result.interpretation.recommendations.length > 0 && (
            <div className="space-y-2">
              <p className="text-gray-300 text-sm font-semibold mb-2">Recommendations:</p>
              {result.interpretation.recommendations.map((rec, idx) => (
                <div key={idx} className="flex items-start space-x-2 text-gray-100 text-sm">
                  <span className="text-purple-400 mt-0.5">•</span>
                  <span>{rec}</span>
                </div>
              ))}
            </div>
          )}

          {/* Model Fit Badge */}
          {result.interpretation.model_fit_quality && (
            <div className="mt-4 flex items-center space-x-2">
              <span className="text-gray-300 text-sm font-semibold">Model Fit:</span>
              <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                result.interpretation.model_fit_quality === 'excellent' ? 'bg-green-500/30 text-green-200' :
                result.interpretation.model_fit_quality === 'good' ? 'bg-blue-500/30 text-blue-200' :
                result.interpretation.model_fit_quality === 'moderate' ? 'bg-yellow-500/30 text-yellow-200' :
                'bg-red-500/30 text-red-200'
              }`}>
                {result.interpretation.model_fit_quality.toUpperCase()}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Test Type */}
      {result.test_type && (
        <div>
          <p className="text-gray-300 text-sm">Test Type</p>
          <p className="text-gray-100 text-lg font-semibold">{result.test_type}</p>
        </div>
      )}

      {/* Main Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {result.t_statistic !== undefined && (
          <div className="bg-slate-700/50 rounded-lg p-4">
            <p className="text-gray-300 text-sm">t-statistic</p>
            <p className="text-gray-100 text-2xl font-bold">{renderValue(result.t_statistic)}</p>
          </div>
        )}
        {result.f_statistic !== undefined && (
          <div className="bg-slate-700/50 rounded-lg p-4">
            <p className="text-gray-300 text-sm">F-statistic</p>
            <p className="text-gray-100 text-2xl font-bold">{renderValue(result.f_statistic)}</p>
          </div>
        )}
        {result.z_statistic !== undefined && (
          <div className="bg-slate-700/50 rounded-lg p-4">
            <p className="text-gray-300 text-sm">Z-statistic</p>
            <p className="text-gray-100 text-2xl font-bold">{renderValue(result.z_statistic)}</p>
          </div>
        )}
        {result.p_value !== undefined && (
          <div className="bg-slate-700/50 rounded-lg p-4">
            <p className="text-gray-300 text-sm">p-value</p>
            <p className="text-gray-100 text-2xl font-bold">{renderValue(result.p_value, 6)}</p>
          </div>
        )}
        {result.alpha !== undefined && (
          <div className="bg-slate-700/50 rounded-lg p-4">
            <p className="text-gray-300 text-sm">Significance Level (α)</p>
            <p className="text-gray-100 text-2xl font-bold">{renderValue(result.alpha, 3)}</p>
          </div>
        )}
        {result.degrees_of_freedom !== undefined && (
          <div className="bg-slate-700/50 rounded-lg p-4">
            <p className="text-gray-300 text-sm">Degrees of Freedom</p>
            <p className="text-gray-100 text-2xl font-bold">
              {typeof result.degrees_of_freedom === 'object'
                ? `${result.degrees_of_freedom.df1}, ${result.degrees_of_freedom.df2}`
                : result.degrees_of_freedom}
            </p>
          </div>
        )}
      </div>

      {/* Confidence Interval */}
      {result.confidence_interval && (
        <div className="bg-slate-700/50 rounded-lg p-4">
          <p className="text-gray-300 text-sm mb-2">
            {(result.confidence_interval.level * 100).toFixed(0)}% Confidence Interval
          </p>
          <p className="text-gray-100 text-lg font-semibold">
            [{renderValue(result.confidence_interval.lower)}, {renderValue(result.confidence_interval.upper)}]
          </p>
        </div>
      )}

      {/* Box Plot */}
      {result.boxplot_data && result.boxplot_data.length > 0 && (
        <BoxPlot data={result.boxplot_data} />
      )}

      {/* Means Plot with CI (for ANOVA) */}
      {result.means_ci && (
        <MeansPlot data={result.means_ci} />
      )}

      {/* Interaction Plot (for Two-Way ANOVA) */}
      {result.interaction_means && result.factor_a_name && result.factor_b_name && (
        <InteractionPlot
          data={result.interaction_means}
          factorAName={result.factor_a_name}
          factorBName={result.factor_b_name}
        />
      )}

      {/* Pareto Chart (for Factorial Designs) */}
      {result.effect_magnitudes && result.effect_magnitudes.length > 0 && (
        <ParetoChart data={result.effect_magnitudes} />
      )}

      {/* Main Effects Plot (for Factorial Designs) */}
      {result.main_effects_plot_data && (
        <MainEffectsPlot data={result.main_effects_plot_data} />
      )}

      {/* Cube Plot (for 2³ Factorial Designs) */}
      {result.cube_data && result.factors && (
        <CubePlot data={result.cube_data} factors={result.factors} />
      )}

      {/* F-Distribution Plot (for ANOVA) */}
      {result.f_statistic && result.f_critical && result.test_type?.includes('ANOVA') && (
        <FDistributionPlot
          fStatistic={result.f_statistic}
          fCritical={result.f_critical}
          alpha={result.alpha}
          df1={result.anova_table?.df?.[0]}
          df2={result.anova_table?.df?.[1]}
        />
      )}

      {/* ANOVA Table */}
      {result.anova_table && (
        <div>
          <h4 className="text-gray-100 font-semibold mb-3">ANOVA Table</h4>
          {renderTable(result.anova_table)}
        </div>
      )}

      {/* Residual Diagnostic Plots (for ANOVA) */}
      {result.residuals && result.fitted_values && result.standardized_residuals && (
        <ResidualPlots
          residuals={result.residuals}
          fittedValues={result.fitted_values}
          standardizedResiduals={result.standardized_residuals}
        />
      )}

      {/* Sample Statistics */}
      {result.sample_stats && (
        <div>
          <h4 className="text-gray-100 font-semibold mb-3">Sample Statistics</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(result.sample_stats).map(([key, value]) => (
              value !== null && (
                <div key={key} className="bg-slate-700/50 rounded-lg p-3">
                  <p className="text-gray-300 text-sm">{key.replace(/_/g, ' ')}</p>
                  <p className="text-gray-100 text-lg font-semibold">{renderValue(value)}</p>
                </div>
              )
            ))}
          </div>
        </div>
      )}

      {/* Group Statistics */}
      {result.group_statistics && (
        <div>
          <h4 className="text-gray-100 font-semibold mb-3">Group Statistics</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(result.group_statistics).map(([group, stats]) => (
              <div key={group} className="bg-slate-700/50 rounded-lg p-4">
                <p className="text-gray-100 font-medium mb-2">Group: {group}</p>
                <div className="space-y-1 text-sm">
                  <p className="text-gray-300">Mean: <span className="text-gray-100">{renderValue(stats.mean)}</span></p>
                  <p className="text-gray-300">Std: <span className="text-gray-100">{renderValue(stats.std)}</span></p>
                  <p className="text-gray-300">n: <span className="text-gray-100">{stats.n}</span></p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Additional Metrics */}
      {result.r_squared !== undefined && (
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-700/50 rounded-lg p-4">
            <p className="text-gray-300 text-sm">R²</p>
            <p className="text-gray-100 text-xl font-bold">{renderValue(result.r_squared)}</p>
          </div>
          {result.adj_r_squared !== undefined && (
            <div className="bg-slate-700/50 rounded-lg p-4">
              <p className="text-gray-300 text-sm">Adjusted R²</p>
              <p className="text-gray-100 text-xl font-bold">{renderValue(result.adj_r_squared)}</p>
            </div>
          )}
        </div>
      )}

      {/* Raw JSON for debugging */}
      <details className="bg-slate-800/50 rounded-lg p-4">
        <summary className="text-gray-100 cursor-pointer font-medium">View Raw Results (JSON)</summary>
        <pre className="text-gray-200 text-xs mt-4 overflow-x-auto">
          {JSON.stringify(result, null, 2)}
        </pre>
      </details>
    </div>
  )
}

export default ResultCard
