import { CheckCircle, XCircle, AlertCircle } from 'lucide-react'

const AssumptionTestsANOVA = ({ assumptions }) => {
  if (!assumptions) return null

  const { normality, homogeneity_of_variance } = assumptions

  // Determine overall status
  const allNormalityPassed = normality.shapiro_wilk.passed &&
                              normality.anderson_darling.passed &&
                              normality.kolmogorov_smirnov.passed

  const allVariancePassed = homogeneity_of_variance.levene.passed &&
                            homogeneity_of_variance.bartlett.passed

  const getStatusIcon = (passed) => {
    if (passed) {
      return <CheckCircle className="w-5 h-5 text-green-400" />
    }
    return <XCircle className="w-5 h-5 text-red-400" />
  }

  const getStatusColor = (passed) => {
    return passed ? 'border-green-500/50 bg-green-500/10' : 'border-red-500/50 bg-red-500/10'
  }

  const getOverallIcon = (allPassed) => {
    if (allPassed) {
      return <CheckCircle className="w-6 h-6 text-green-400" />
    }
    return <AlertCircle className="w-6 h-6 text-yellow-400" />
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-slate-700/50">
      <div className="flex items-center space-x-3 mb-4">
        <AlertCircle className="w-6 h-6 text-blue-400" />
        <h3 className="text-xl font-bold text-gray-100">Assumptions Testing</h3>
      </div>

      <p className="text-gray-400 text-sm mb-6">
        ANOVA requires that residuals are normally distributed and that group variances are equal (homoscedasticity).
        These tests help validate those assumptions.
      </p>

      {/* Normality Tests */}
      <div className="mb-6">
        <div className="flex items-center space-x-2 mb-3">
          {getOverallIcon(allNormalityPassed)}
          <h4 className="text-lg font-semibold text-gray-200">Normality of Residuals</h4>
        </div>

        <div className="space-y-3">
          {/* Shapiro-Wilk */}
          <div className={`rounded-lg border p-4 ${getStatusColor(normality.shapiro_wilk.passed)}`}>
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-2">
                  {getStatusIcon(normality.shapiro_wilk.passed)}
                  <h5 className="font-semibold text-gray-100">Shapiro-Wilk Test</h5>
                </div>
                <p className="text-sm text-gray-300 mb-2">{normality.shapiro_wilk.interpretation}</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-400">Statistic:</span>
                    <span className="ml-2 text-gray-200 font-mono">{normality.shapiro_wilk.statistic}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">p-value:</span>
                    <span className="ml-2 text-gray-200 font-mono">{normality.shapiro_wilk.p_value}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Anderson-Darling */}
          <div className={`rounded-lg border p-4 ${getStatusColor(normality.anderson_darling.passed)}`}>
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-2">
                  {getStatusIcon(normality.anderson_darling.passed)}
                  <h5 className="font-semibold text-gray-100">Anderson-Darling Test</h5>
                </div>
                <p className="text-sm text-gray-300 mb-2">{normality.anderson_darling.interpretation}</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-400">Statistic:</span>
                    <span className="ml-2 text-gray-200 font-mono">{normality.anderson_darling.statistic}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Critical Value (5%):</span>
                    <span className="ml-2 text-gray-200 font-mono">{normality.anderson_darling.critical_value}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Kolmogorov-Smirnov */}
          <div className={`rounded-lg border p-4 ${getStatusColor(normality.kolmogorov_smirnov.passed)}`}>
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-2">
                  {getStatusIcon(normality.kolmogorov_smirnov.passed)}
                  <h5 className="font-semibold text-gray-100">Kolmogorov-Smirnov Test</h5>
                </div>
                <p className="text-sm text-gray-300 mb-2">{normality.kolmogorov_smirnov.interpretation}</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-400">Statistic:</span>
                    <span className="ml-2 text-gray-200 font-mono">{normality.kolmogorov_smirnov.statistic}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">p-value:</span>
                    <span className="ml-2 text-gray-200 font-mono">{normality.kolmogorov_smirnov.p_value}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {!allNormalityPassed && (
          <div className="mt-3 bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-3">
            <p className="text-yellow-200 text-sm">
              <strong>Recommendation:</strong> Consider using non-parametric alternatives (Kruskal-Wallis H test)
              or transforming your data. ANOVA is robust to moderate violations with large, equal sample sizes.
            </p>
          </div>
        )}
      </div>

      {/* Homogeneity of Variance Tests */}
      <div>
        <div className="flex items-center space-x-2 mb-3">
          {getOverallIcon(allVariancePassed)}
          <h4 className="text-lg font-semibold text-gray-200">Homogeneity of Variance</h4>
        </div>

        <div className="space-y-3">
          {/* Levene's Test */}
          <div className={`rounded-lg border p-4 ${getStatusColor(homogeneity_of_variance.levene.passed)}`}>
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-2">
                  {getStatusIcon(homogeneity_of_variance.levene.passed)}
                  <h5 className="font-semibold text-gray-100">Levene's Test</h5>
                  <span className="text-xs text-gray-400">(Robust to non-normality)</span>
                </div>
                <p className="text-sm text-gray-300 mb-2">{homogeneity_of_variance.levene.interpretation}</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-400">Statistic:</span>
                    <span className="ml-2 text-gray-200 font-mono">{homogeneity_of_variance.levene.statistic}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">p-value:</span>
                    <span className="ml-2 text-gray-200 font-mono">{homogeneity_of_variance.levene.p_value}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Bartlett's Test */}
          <div className={`rounded-lg border p-4 ${getStatusColor(homogeneity_of_variance.bartlett.passed)}`}>
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-2">
                  {getStatusIcon(homogeneity_of_variance.bartlett.passed)}
                  <h5 className="font-semibold text-gray-100">Bartlett's Test</h5>
                  <span className="text-xs text-gray-400">(Sensitive to non-normality)</span>
                </div>
                <p className="text-sm text-gray-300 mb-2">{homogeneity_of_variance.bartlett.interpretation}</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-400">Statistic:</span>
                    <span className="ml-2 text-gray-200 font-mono">{homogeneity_of_variance.bartlett.statistic}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">p-value:</span>
                    <span className="ml-2 text-gray-200 font-mono">{homogeneity_of_variance.bartlett.p_value}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {!allVariancePassed && (
          <div className="mt-3 bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-3">
            <p className="text-yellow-200 text-sm">
              <strong>Recommendation:</strong> Consider Welch's ANOVA (which doesn't assume equal variances)
              or transforming your data. ANOVA is fairly robust to variance heterogeneity when sample sizes are equal.
            </p>
          </div>
        )}
      </div>

      {/* Overall Summary */}
      <div className="mt-6 bg-slate-700/30 rounded-lg p-4">
        <h5 className="font-semibold text-gray-200 mb-2">Summary</h5>
        <div className="space-y-1 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Normality assumption:</span>
            <span className={`font-semibold ${allNormalityPassed ? 'text-green-400' : 'text-yellow-400'}`}>
              {allNormalityPassed ? 'Satisfied' : 'Questionable'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Homogeneity assumption:</span>
            <span className={`font-semibold ${allVariancePassed ? 'text-green-400' : 'text-yellow-400'}`}>
              {allVariancePassed ? 'Satisfied' : 'Questionable'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Overall ANOVA validity:</span>
            <span className={`font-semibold ${(allNormalityPassed && allVariancePassed) ? 'text-green-400' : 'text-yellow-400'}`}>
              {(allNormalityPassed && allVariancePassed) ? 'High confidence' : 'Proceed with caution'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AssumptionTestsANOVA
