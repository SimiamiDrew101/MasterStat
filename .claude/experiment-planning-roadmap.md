# Experiment Planning - Feature Roadmap

## Currently Implemented ✓

### Sample Size Calculations
- **t-Tests**
  - One-sample t-test
  - Two-sample t-test (independent groups)
  - Paired t-test
  - Support for unequal sample sizes (ratio parameter)
  - Alternative hypotheses: two-sided, greater, less

- **ANOVA**
  - One-way ANOVA
  - Two-way ANOVA (with interaction effects)
  - Effect of interest selection (main A, main B, interaction)

- **Proportion Tests**
  - One-sample proportion test
  - Two-sample proportion test (A/B testing)
  - McNemar's test (paired proportions)
  - Support for unequal sample sizes

### Effect Size Tools
- **Effect Size Converter**
  - Convert between: Cohen's d, Cohen's f, η², correlation (r), odds ratio
  - Provides interpretations for all metrics
  - Shows all equivalent values

- **Effect Size from Pilot Data**
  - Independent samples: calculate Cohen's d with 95% CI
  - Paired samples: calculate Cohen's d
  - ANOVA: calculate Cohen's f and η²
  - Forest plot visualization with export

- **Minimum Detectable Effect Size**
  - Calculate smallest detectable effect given sample size constraints
  - Supports all t-test and ANOVA variants
  - Sensitivity curve visualization showing trade-offs
  - Export functionality

### Visualizations
- Power curves (power vs. sample size, power vs. effect size)
- Sensitivity curves (sample size vs. minimum detectable effect)
- Forest plots for pilot data effect sizes
- All charts exportable as high-resolution PNG

---

## Potential Future Additions

### 1. Correlation & Regression Analysis
**Priority: HIGH**
- Sample size for correlation coefficients
- Sample size for simple/multiple regression
- Power for testing R² and individual predictors
- **Use Cases**: Psychology studies, predictive modeling, relationship studies

### 2. Survival Analysis
**Priority: MEDIUM-HIGH**
- Log-rank test sample size
- Cox proportional hazards regression
- Kaplan-Meier curve comparisons
- **Use Cases**: Clinical trials, time-to-event studies, reliability engineering

### 3. Equivalence & Non-Inferiority Testing
**Priority: HIGH**
- TOST (Two One-Sided Tests) procedure
- Non-inferiority margin specification
- Equivalence bounds calculation
- **Use Cases**: Biosimilar studies, generic drug trials, quality control

### 4. Repeated Measures / Within-Subjects Designs
**Priority: HIGH**
- Repeated measures ANOVA
- Mixed designs (between + within factors)
- Accounting for correlation structure
- Greenhouse-Geisser correction planning
- **Use Cases**: Longitudinal studies, learning experiments, intervention effects over time

### 5. Cluster Randomized Trials
**Priority: MEDIUM**
- Intraclass correlation (ICC) incorporation
- Design effect calculation
- Optimal cluster size determination
- **Use Cases**: Educational interventions, community health programs, organizational studies

### 6. Sequential Analysis
**Priority: MEDIUM**
- Interim analysis planning
- Alpha spending functions (O'Brien-Fleming, Pocock)
- Group sequential designs
- Futility boundaries
- **Use Cases**: Clinical trials with interim looks, adaptive trials

### 7. Multiple Comparisons Correction
**Priority: MEDIUM**
- Bonferroni, Holm, FDR corrections
- Power loss with corrections
- Optimal study design accounting for multiple tests
- **Use Cases**: Genomics, neuroimaging, exploratory studies

### 8. Crossover Designs
**Priority: LOW-MEDIUM**
- 2×2 crossover
- Higher-order crossovers
- Washout period considerations
- Carryover effect planning
- **Use Cases**: Pharmacology, behavioral studies, within-subject comparisons

### 9. Non-Parametric Tests
**Priority: LOW-MEDIUM**
- Wilcoxon rank-sum test
- Wilcoxon signed-rank test
- Kruskal-Wallis test
- Efficiency relative to parametric tests
- **Use Cases**: Small samples, ordinal data, non-normal distributions

### 10. Reliability & Agreement Studies
**Priority: MEDIUM**
- Sample size for ICC estimation
- Kappa coefficient studies
- Bland-Altman analysis planning
- Cronbach's alpha precision
- **Use Cases**: Inter-rater reliability, measurement validation, test development

### 11. Multi-Level / Hierarchical Models
**Priority: LOW-MEDIUM**
- Nested data structures
- Random effects variance components
- Level-specific sample size requirements
- **Use Cases**: Educational data, organizational hierarchies, repeated measures

### 12. Bayesian Sample Size Determination
**Priority: LOW**
- Prior specification
- Credible interval precision
- Bayes factor thresholds
- Posterior probability of effect
- **Use Cases**: Small samples, informative priors, sequential updating

### 13. Optimal Allocation Strategies
**Priority: LOW**
- Minimize variance for given budget
- Optimal ratios for unequal costs
- Response-adaptive randomization
- **Use Cases**: Budget constraints, expensive interventions, rare populations

### 14. Adaptive Designs
**Priority: LOW**
- Sample size re-estimation
- Treatment arm dropping
- Seamless Phase II/III
- **Use Cases**: Oncology trials, early-phase research

### 15. Chi-Square & Contingency Tables
**Priority: MEDIUM-HIGH**
- Chi-square test of independence
- Fisher's exact test
- Goodness-of-fit tests
- Effect size (Cramér's V, phi coefficient)
- **Use Cases**: Categorical data analysis, survey research, epidemiology

---

## Implementation Priority Summary

### Phase 1 (Next Implementation)
1. Correlation & Regression sample size
2. Equivalence/Non-inferiority testing
3. Chi-square tests
4. Repeated measures designs

### Phase 2 (Medium-term)
1. Survival analysis
2. Cluster randomized trials
3. Multiple comparisons planning
4. Reliability studies

### Phase 3 (Long-term / As Needed)
1. Sequential analysis
2. Crossover designs
3. Non-parametric tests
4. Bayesian methods
5. Adaptive designs

---

## Notes
- All new features should include:
  - Clear parameter input UI with tooltips
  - Effect size estimation/conversion tools
  - Visualization where applicable
  - Export functionality
  - Interpretation guidance
  - Example use cases

- Consider user workflow:
  - Most users need: sample size → analyze → report
  - Power for existing data should be de-emphasized (post-hoc power)
  - Focus on prospective planning tools
