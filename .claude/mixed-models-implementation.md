# Mixed Models Implementation Status

## Currently Implemented ✓

### 1. Mixed Model ANOVA (`/api/mixed/mixed-model-anova`)

**Features:**
- ✓ Expected Mean Squares (EMS) calculation for each source
- ✓ Variance component estimation using ANOVA method
- ✓ Proper F-tests with correct error terms for mixed models
- ✓ Support for fixed and random factors
- ✓ Interaction effects with correct testing
- ✓ Variance percentages showing relative contributions
- ✓ Automatic interpretation of results
- ✓ Model fit statistics (R², AIC, BIC)

**Supported Designs:**
- One-way Fixed Effects ANOVA
- One-way Random Effects ANOVA
- Two-Factor Mixed Model (1 fixed + 1 random)
  - With interaction
  - Additive model (no interaction)

**Key Statistical Features:**
1. **Expected Mean Squares (EMS)**:
   - Shows what each MS estimates
   - Example: `σ² + 2σ²(Treatment×Subject) + 6σ²(Treatment)`
   - Essential for understanding proper error terms

2. **Corrected F-Tests**:
   - Fixed effects tested against interaction MS (not residual)
   - Random effects tested against interaction MS
   - Interaction tested against residual MS
   - Provides correct p-values for mixed models

3. **Variance Components**:
   - σ²_error: Within-cell variability
   - σ²_random: Between-level variability for random factors
   - σ²_interaction: Interaction variability
   - Shown as absolute values and percentages

**Example Output:**
```json
{
  "anova_table": {
    "Treatment": {
      "sum_sq": 93.5208,
      "df": 1,
      "mean_sq": 93.5208,
      "ems": "σ² + 2σ²(Treatment×Subject) + 6σ²(Treatment)",
      "F_corrected": 433.3012,
      "p_value_corrected": 0.0023,
      "error_term": "Treatment:Subject"
    }
  },
  "variance_components": {
    "σ²_error": 0.145833,
    "σ²_Treatment×Subject": 0.035,
    "σ²_Subject": 1.54125
  },
  "variance_percentages": {
    "σ²_error": 8.47,
    "σ²_Treatment×Subject": 2.03,
    "σ²_Subject": 89.5
  }
}
```

---

## To Be Implemented Next

### 2. Split-Plot Design (`/api/mixed/split-plot`)

**Status:** ✅ COMPLETED

**Includes:**
- Whole-plot factor (applied to large experimental units)
- Sub-plot factor (applied within whole plots)
- Block effects
- Two error terms:
  - Whole-plot error (for whole-plot factor)
  - Sub-plot error (for sub-plot factor and interaction)
- Proper F-tests using correct error terms
- EMS for each source
- Variance component estimation

**Use Cases:**
- Agricultural field trials (e.g., irrigation methods × crop varieties)
- Educational studies (classroom-level and student-level interventions)
- Industrial experiments (batch-level and within-batch factors)

**Example Output:**
```json
{
  "anova_table": {
    "Block": {
      "ems": "σ² + 3σ²(Irrigation×Block) + 6σ²(Block)",
      "F_corrected": 14.9378,
      "p_value_corrected": 0.062744,
      "error_term": "Irrigation:Block"
    },
    "Irrigation": {
      "ems": "σ² + 3σ²(Irrigation×Block) + 9σ²(Irrigation)",
      "F_corrected": 2142.2228,
      "p_value_corrected": 0.000466,
      "error_term": "Irrigation:Block"
    },
    "Irrigation:Block": {
      "ems": "σ² + 3σ²(Irrigation×Block)",
      "label": "Error (Whole-plot)"
    },
    "Variety": {
      "ems": "σ² + 6σ²(Variety)",
      "F_corrected": 51450.0,
      "p_value_corrected": 0.0,
      "error_term": "Residual"
    },
    "Irrigation:Variety": {
      "ems": "σ² + 3σ²(Irrigation×Variety)",
      "F_corrected": 338.0,
      "p_value_corrected": 0.0,
      "error_term": "Residual"
    },
    "Residual": {
      "ems": "σ²",
      "label": "Error (Sub-plot)"
    }
  },
  "variance_components": {
    "σ²_subplot": 0.001111,
    "σ²_wholeplot": 0.03537,
    "σ²_Irrigation×Variety": 0.374444,
    "σ²_Block": 0.249074,
    "σ²_Irrigation": 25.50963
  },
  "variance_percentages": {
    "σ²_Irrigation": 97.48,
    "σ²_Irrigation×Variety": 1.43,
    "σ²_Block": 0.95,
    "σ²_wholeplot": 0.14
  }
}
```

---

### 3. Nested Design (`/api/mixed/nested-design`)

**Status:** Endpoint exists but needs full implementation

**Will Include:**
- Hierarchical factor structure (B nested within A)
- Proper partitioning of variance
- EMS showing nesting structure
- Variance components for each level
- Tests for both between-group and within-group effects

**Use Cases:**
- Students nested within Schools
- Samples nested within Batches
- Measurements nested within Subjects
- Employees nested within Departments

---

### 4. Expected Mean Squares Table (EMS) - Visualization

**Status:** Planned for UI

**Features:**
- Clear table showing EMS for each source
- Highlights which MS to use as denominator for F-tests
- Shows variance components being estimated
- Educational annotations explaining EMS notation

---

### 5. Variance Component Visualization

**Status:** Planned for UI

**Features:**
- Pie chart showing relative variance contributions
- Bar chart with confidence intervals
- Comparison across different sources
- Interpretation guidance

---

## Implementation Sequence

1. ✅ **Mixed Model ANOVA** - COMPLETED
   - EMS calculations
   - Variance components
   - Proper F-tests

2. ✅ **Split-Plot Design** - COMPLETED
   - Whole-plot and sub-plot analysis
   - Two error terms (whole-plot error and sub-plot error)
   - Complex EMS structure with proper denominator MS
   - Support for RCBD and CRD at whole-plot level

3. **Next: Nested Design**
   - Hierarchical structure
   - Variance partitioning
   - Nested factor testing

4. **UI Development**
   - Data entry interface
   - Factor designation (fixed/random)
   - Results tables with EMS
   - Visualizations (variance components, EMS tables)
   - Export functionality

5. **Advanced Features**
   - Confidence intervals for variance components
   - Unbalanced designs
   - Three-factor mixed models
   - Repeated measures (special case of split-plot)
   - Cross-classified random effects

---

## Key Concepts Implemented

### Fixed vs. Random Effects
- **Fixed Effects**: Levels are specifically chosen (e.g., Treatment A vs. B)
  - Interest in these specific levels
  - Inferences apply only to these levels
  - Tested against interaction or residual

- **Random Effects**: Levels are randomly sampled (e.g., Subjects, Schools)
  - Interest in population of levels
  - Inferences apply to all possible levels
  - Contributes to variance components

### Why Proper Error Terms Matter
In mixed models, using MSE (residual) for all tests is **incorrect**:
- Fixed effects should be tested against the interaction term when random effects are present
- Using wrong error term leads to:
  - Inflated Type I error rates (false positives)
  - Incorrect p-values
  - Invalid conclusions

### Variance Components
Partition total variability into sources:
- Helps understand what contributes most to variation
- Guides where to focus improvement efforts
- Essential for sample size planning in future studies
- Used in intraclass correlation calculations

---

## Notes for Future Enhancements

1. **Unbalanced Designs**: Current implementation assumes balanced data
   - Need adjusted variance component formulas
   - More complex EMS calculations

2. **REML Estimation**: Currently using ANOVA method
   - Consider adding REML option for better estimates
   - More accurate with unbalanced data

3. **Confidence Intervals**: Add for variance components
   - Using Satterthwaite approximation
   - Bootstrap methods for complex designs

4. **Power Analysis**: Add sample size determination for mixed models
   - Based on variance component estimates
   - Account for clustering/nesting

5. **Diagnostics**: Add model checking
   - Residual plots
   - Q-Q plots
   - Variance homogeneity tests
