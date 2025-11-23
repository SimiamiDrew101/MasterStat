# COMPREHENSIVE TECHNICAL ANALYSIS: Multi-Response Overlay Feature for RSM

## Executive Summary

This document provides a detailed technical analysis of the current RSM implementation and outlines a comprehensive design for implementing a Multi-Response Overlay visualization feature. The analysis covers data structures, API endpoints, visualization components, and provides a phased implementation approach.

---

## 1. CURRENT RSM DATA STRUCTURE ANALYSIS

### 1.1 Table Data Structure (Frontend)

**Location**: `/Users/nj/Desktop/MasterStat/frontend/src/pages/RSM.jsx`

**Current Structure** (Lines 28-29, 192-196):
```javascript
const [tableData, setTableData] = useState([])

// Table format: Array of arrays
// Example: [[x1_val, x2_val, response_val], [x1_val, x2_val, response_val], ...]
const table = response.data.design_matrix.map(row => {
  const tableRow = factorNames.map(factor => row[factor] || 0)
  tableRow.push('')  // Single empty response column
  return tableRow
})
```

**Key Characteristics**:
- Each row is an array: `[factor1, factor2, ..., factorN, response]`
- Response is in the LAST column (index: `row.length - 1`)
- Single response variable stored in state: `responseName` (Line 24)
- Data conversion happens in `handleFitModel` (Lines 231-238):
  ```javascript
  const data = validRows.map(row => {
    const point = {}
    factorNames.forEach((factor, i) => {
      point[factor] = parseFloat(row[i])
    })
    point[responseName] = parseFloat(row[row.length - 1])  // Single response
    return point
  })
  ```

**Implication for Multi-Response**:
- Need to extend table to support multiple response columns
- Need to track multiple response names
- Data validation must handle multiple response columns

---

## 2. BACKEND /fit-model ENDPOINT ANALYSIS

### 2.1 Current Implementation

**Location**: `/Users/nj/Desktop/MasterStat/backend/app/api/rsm.py` (Lines 29-250)

**Request Model** (Lines 13-17):
```python
class RSMRequest(BaseModel):
    data: List[Dict[str, float]]  # Example: [{"X1": 1, "X2": 2, "Y": 10}, ...]
    factors: List[str]
    response: str  # SINGLE response variable name
    alpha: float = 0.05
```

**Current Behavior**:
- Fits a SINGLE second-order polynomial model
- Returns coefficients for ONE response
- Formula: `Y ~ X1 + X2 + I(X1**2) + I(X2**2) + X1:X2`
- Returns ONE set of diagnostics, ANOVA, R², etc.

**Key Code Section** (Lines 38-54):
```python
# Build second-order model formula
linear_terms = " + ".join(request.factors)
quadratic_terms = " + ".join([f"I({f}**2)" for f in request.factors])
interaction_terms = []
for i in range(len(request.factors)):
    for j in range(i+1, len(request.factors)):
        interaction_terms.append(f"{request.factors[i]}:{request.factors[j]}")

if interaction_terms:
    formula = f"{request.response} ~ {linear_terms} + {quadratic_terms} + {' + '.join(interaction_terms)}"
else:
    formula = f"{request.response} ~ {linear_terms} + {quadratic_terms}"

# Fit model
model = ols(formula, data=df).fit()
```

### 2.2 Changes Required for Multi-Response

**Option A: Batch Fitting (Recommended for MVP)**
```python
class MultiRSMRequest(BaseModel):
    data: List[Dict[str, float]]
    factors: List[str]
    responses: List[str]  # Multiple response names
    alpha: float = 0.05

@router.post("/fit-multi-model")
async def fit_multi_rsm_model(request: MultiRSMRequest):
    """Fit separate RSM models for multiple responses"""
    results = {}

    for response in request.responses:
        # Fit individual model (reuse existing logic)
        single_request = RSMRequest(
            data=request.data,
            factors=request.factors,
            response=response,
            alpha=request.alpha
        )
        results[response] = await fit_rsm_model(single_request)

    return {
        "models": results,
        "n_responses": len(request.responses),
        "factors": request.factors
    }
```

**Option B: Multivariate Response Surface (Advanced)**
- Would require MANOVA-based approach
- More complex, better for Phase 2

---

## 3. FRONTEND ContourPlot COMPONENT ANALYSIS

### 3.1 Current Implementation

**Location**: `/Users/nj/Desktop/MasterStat/frontend/src/components/ContourPlot.jsx` (Lines 1-326)

**Input Props**:
```javascript
const ContourPlot = ({
  surfaceData,        // Array of {x, y, z} points
  factor1,            // X-axis factor name
  factor2,            // Y-axis factor name
  responseName,       // Single response name (for labels)
  experimentalData,   // Optional overlay data
  optimizationResult,
  canonicalResult,
  steepestAscentResult,
  ridgeAnalysisResult
})
```

**Data Format Expected** (Lines 18-29):
```javascript
// surfaceData structure
// Input: [{x: -2, y: -2, z: 5.2}, {x: -2, y: -1.8, z: 5.5}, ...]
const xValues = [...new Set(surfaceData.map(d => d.x))].sort((a, b) => a - b)
const yValues = [...new Set(surfaceData.map(d => d.y))].sort((a, b) => a - b)

// Create 2D grid for Plotly contour
const zGrid = []
for (let i = 0; i < yValues.length; i++) {
  zGrid[i] = []
  for (let j = 0; j < xValues.length; j++) {
    const point = surfaceData.find(d => d.x === xValues[j] && d.y === yValues[i])
    zGrid[i][j] = point ? point.z : 0
  }
}
```

**Plotly Trace Structure** (Lines 35-64):
```javascript
traces.push({
  type: 'contour',
  x: xValues,
  y: yValues,
  z: zGrid,
  colorscale: [...],
  contours: {
    coloring: 'heatmap',
    showlabels: true
  },
  colorbar: {
    title: { text: responseName }
  }
})
```

### 3.2 Multi-Response Overlay Challenges

**Challenge 1: Z-Value Scale Differences**
- Response 1 might range from 10-50
- Response 2 might range from 0.01-0.05
- Direct overlay would be visually meaningless

**Challenge 2: Contour Interpretation**
- Multiple contour sets can overlap and become confusing
- Need visual distinction (colors, line styles, transparency)

**Challenge 3: Data Density**
- Each response generates 20x20 = 400 surface points
- Multiple responses = increased rendering complexity

---

## 4. EXISTING MULTI-RESPONSE SUPPORT

### 4.1 Desirability Function Implementation

**Location**: `/Users/nj/Desktop/MasterStat/backend/app/api/rsm.py` (Lines 934-1059)

**Current Multi-Response Handling**:
```python
class DesirabilitySpec(BaseModel):
    response_name: str
    coefficients: Dict[str, Any]  # Model coefficients for THIS response
    goal: str  # 'maximize', 'minimize', 'target'
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    target: Optional[float]
    weight: float = 1.0
    importance: float = 1.0

class DesirabilityRequest(BaseModel):
    responses: List[DesirabilitySpec]  # Multiple responses!
    factors: List[str]
```

**Key Insight**: The system ALREADY supports multiple response models, but only for optimization purposes, not visualization.

**Frontend Implementation** (Lines 51-53, 687-703):
```javascript
const [multiResponseModels, setMultiResponseModels] = useState([])  // Array of {name, coefficients}
const [desirabilitySpecs, setDesirabilitySpecs] = useState([])

// Helper: Save current model to multi-response models
const saveCurrentModelToMultiResponse = () => {
  const newModel = {
    name: responseName,
    coefficients: Object.fromEntries(
      Object.entries(modelResult.coefficients).map(([k, v]) => [k, v.estimate])
    )
  }
  setMultiResponseModels([...multiResponseModels, newModel])
}
```

**Workflow**:
1. User fits model for Response 1
2. Saves it to `multiResponseModels`
3. Changes `responseName` to Response 2
4. Fits another model
5. Saves that too
6. Uses desirability function to optimize across both

**Limitation**: No visualization overlay - only optimization

---

## 5. TECHNICAL CHALLENGES FOR MULTI-RESPONSE OVERLAY

### 5.1 Data Structure Changes

**Challenge**: Extend table to support multiple response columns

**Current**:
```
| X1  | X2  | Y   |
|-----|-----|-----|
| -1  | -1  | 10  |
| -1  |  1  | 12  |
```

**Proposed**:
```
| X1  | X2  | Y1  | Y2  | Y3  |
|-----|-----|-----|-----|-----|
| -1  | -1  | 10  | 0.5 | 85  |
| -1  |  1  | 12  | 0.3 | 90  |
```

**Implementation**:
```javascript
// New state
const [responseNames, setResponseNames] = useState(['Y1'])  // Array instead of single string
const [activeResponses, setActiveResponses] = useState(['Y1'])  // Which to visualize

// Modified table structure
const table = response.data.design_matrix.map(row => {
  const tableRow = factorNames.map(factor => row[factor] || 0)
  // Add empty columns for each response
  responseNames.forEach(() => tableRow.push(''))
  return tableRow
})
```

### 5.2 Visualization Complexity

**Challenge**: Display multiple overlapping response surfaces clearly

**Solutions**:

1. **Layered Contours with Transparency**
   - Each response gets its own contour trace
   - Use transparency (opacity: 0.5-0.7)
   - Different color schemes per response

2. **Different Line Styles**
   - Solid lines for Response 1
   - Dashed for Response 2
   - Dotted for Response 3

3. **Toggle Individual Responses**
   - Checkboxes to show/hide each response
   - Better for 3+ responses

### 5.3 Scale/Normalization Issues

**Problem**: Responses on different scales can't be meaningfully overlaid

**Solutions**:

1. **Standardization** (Recommended for overlay):
   ```javascript
   // Z-score normalization
   const normalizedZ = (z - mean) / stdDev
   ```

2. **Min-Max Scaling**:
   ```javascript
   const scaledZ = (z - min) / (max - min)  // Maps to [0, 1]
   ```

3. **Separate Colorbars**:
   - Each response gets its own colorbar
   - Side-by-side placement

4. **Unified Desirability Scale**:
   - Convert all responses to desirability [0, 1]
   - Overlay desirability surfaces instead of raw values

### 5.4 User Interaction Design

**Challenges**:
- How to add/remove response columns?
- How to select which responses to overlay?
- How to configure overlay settings?

**Proposed UI Flow**:
1. "Manage Responses" button in Design tab
2. Modal dialog to add/remove response columns
3. In Visualize tab: Multi-select checkboxes for which responses to show
4. Overlay settings panel (normalization method, transparency, colors)

---

## 6. RECOMMENDED IMPLEMENTATION ARCHITECTURE

### 6.1 Phased Approach

**PHASE 1: Multi-Response Data Infrastructure (Week 1)**
- Extend table to support N response columns
- Update data validation to handle multiple responses
- Create multi-response model fitting endpoint
- Store multiple fitted models in frontend state

**PHASE 2: Side-by-Side Visualization (Week 2)**
- Create component to display 2-3 contour plots side-by-side
- No overlay complexity yet
- Simple comparison view
- Test with real multi-response datasets

**PHASE 3: Basic Overlay (Week 3)**
- Implement 2-response overlay with standardization
- Toggle between responses
- Single colorbar approach
- Alpha transparency

**PHASE 4: Advanced Overlay (Week 4)**
- Support N responses (up to 5)
- Multiple normalization options
- Advanced color schemes
- Desirability overlay mode

### 6.2 Backend Changes Needed

**New Endpoints**:

1. **Multi-Model Fitting**:
   ```python
   POST /api/rsm/fit-multi-model
   Request: { data, factors, responses: ["Y1", "Y2", "Y3"], alpha }
   Response: { models: { Y1: {...}, Y2: {...}, Y3: {...} } }
   ```

2. **Surface Data Generation for Multiple Responses**:
   ```python
   POST /api/rsm/generate-multi-surface
   Request: { models, factors, grid_resolution: 20 }
   Response: {
     surfaces: {
       Y1: [{x, y, z}, ...],
       Y2: [{x, y, z}, ...],
       Y3: [{x, y, z}, ...]
     },
     normalization: { Y1: {mean, std}, Y2: {mean, std}, ... }
   }
   ```

3. **Multi-Response Optimization** (Enhancement to existing):
   ```python
   POST /api/rsm/multi-optimize
   # Extends existing desirability endpoint
   # Returns optimal point + visualization data for overlay
   ```

### 6.3 Frontend Components to Create/Modify

**New Components**:

1. **`MultiResponseManager.jsx`**
   - Modal for adding/removing response columns
   - Validation and column management
   - Props: `responseNames`, `onUpdate`

2. **`MultiResponseContourOverlay.jsx`**
   - Extends `ContourPlot.jsx`
   - Handles multiple surface datasets
   - Normalization and scaling logic
   - Props:
     ```javascript
     {
       surfacesData: { Y1: [...], Y2: [...], Y3: [...] },
       responseConfigs: { Y1: {color, visible, normalized}, ... },
       factor1, factor2,
       experimentalData,
       overlayMode: 'layered' | 'side-by-side' | 'animated'
     }
     ```

3. **`ResponseSelector.jsx`**
   - Checkbox list to select which responses to visualize
   - Color picker for each response
   - Transparency slider

**Modified Components**:

1. **`RSM.jsx`** (Main page):
   - Add `responseNames` state (array)
   - Add `fittedModels` state (object keyed by response name)
   - Update table rendering to handle multiple response columns
   - Add "Manage Responses" UI

2. **`ContourPlot.jsx`**:
   - Extract normalization logic to utility function
   - Make colorscale configurable via props
   - Support multiple traces

### 6.4 Data Flow Between Components

```
RSM.jsx (Parent)
  |
  ├─> MultiResponseManager ─────> Updates responseNames[]
  |
  ├─> handleFitMultiModel() ─────> POST /fit-multi-model
  |                                      |
  |                                      v
  |                                 fittedModels = {Y1: {...}, Y2: {...}}
  |
  ├─> generateMultiSurface() ────> Compute surface data for each response
  |                                      |
  |                                      v
  |                                 multiSurfaceData = {Y1: [...], Y2: [...]}
  |
  └─> MultiResponseContourOverlay
        Props: {
          surfacesData: multiSurfaceData,
          responseConfigs: {
            Y1: {visible: true, color: 'Viridis', opacity: 0.7},
            Y2: {visible: true, color: 'Plasma', opacity: 0.5}
          }
        }
```

---

## 7. VISUALIZATION STRATEGIES

### 7.1 Layered Contours with Transparency

**Best For**: 2-3 responses, similar scales

**Plotly Implementation**:
```javascript
const traces = []

// Response 1
traces.push({
  type: 'contour',
  x: xValues,
  y: yValues,
  z: normalizedZGrid1,
  colorscale: 'Viridis',
  opacity: 0.7,
  contours: {
    coloring: 'lines',  // Only contour lines, not filled
    showlabels: true
  },
  name: 'Y1',
  showlegend: true
})

// Response 2
traces.push({
  type: 'contour',
  x: xValues,
  y: yValues,
  z: normalizedZGrid2,
  colorscale: 'Plasma',
  opacity: 0.5,
  contours: {
    coloring: 'lines',
    showlabels: true
  },
  name: 'Y2',
  showlegend: true
})
```

**Advantages**:
- Direct visual comparison
- Can see where responses align/conflict

**Disadvantages**:
- Cluttered with 3+ responses
- Hard to interpret exact values

### 7.2 Side-by-Side Comparison

**Best For**: 2-4 responses, detailed analysis

**Implementation**:
```javascript
<div className="grid grid-cols-2 gap-4">
  {Object.entries(surfacesData).map(([responseName, surfaceData]) => (
    <ContourPlot
      key={responseName}
      surfaceData={surfaceData}
      responseName={responseName}
      factor1={factor1}
      factor2={factor2}
    />
  ))}
</div>
```

**Advantages**:
- Clear, unambiguous
- Each response fully visible
- Easy to implement

**Disadvantages**:
- Harder to see correlations
- More screen space required

### 7.3 Toggle Between Responses

**Best For**: Many responses (5+), quick switching

**Implementation**:
```javascript
const [activeResponseIndex, setActiveResponseIndex] = useState(0)
const activeResponse = responseNames[activeResponseIndex]

<div>
  <select onChange={(e) => setActiveResponseIndex(e.target.value)}>
    {responseNames.map((name, idx) => (
      <option key={idx} value={idx}>{name}</option>
    ))}
  </select>

  <ContourPlot
    surfaceData={surfacesData[activeResponse]}
    responseName={activeResponse}
    ...
  />
</div>
```

**Advantages**:
- Clean UI
- Scalable to many responses

**Disadvantages**:
- Can't compare responses simultaneously
- Requires switching back and forth

### 7.4 Color Coding Strategies

**Strategy A: Different Colormaps**
```javascript
const colorMaps = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis']
responseNames.forEach((name, idx) => {
  traces.push({
    ...
    colorscale: colorMaps[idx % colorMaps.length]
  })
})
```

**Strategy B: Unified Scale with Desirability**
```javascript
// Convert all responses to desirability [0, 1]
// Use single colormap: 0 (bad) = blue, 1 (good) = red
traces.push({
  colorscale: 'RdYlBu',  // Red-Yellow-Blue
  zmin: 0,
  zmax: 1
})
```

**Strategy C: Line Styles (Contour-only mode)**
```javascript
{
  contours: {
    coloring: 'none',  // No fill
    showlines: true,
    line: {
      color: 'red',
      width: 2,
      dash: 'solid'  // or 'dash', 'dot', 'dashdot'
    }
  }
}
```

---

## 8. MVP SCOPE DEFINITION

### 8.1 Minimal Viable Product Features

**Scope**: 2-Response Overlay First

**Included**:
1. Ability to add 2 response columns to data table
2. Fit models for both responses
3. Side-by-side contour plot comparison
4. Basic overlay with standardization
5. Toggle visibility of each response

**Excluded (for later phases)**:
- 3+ responses
- Advanced normalization options
- Animated transitions
- Desirability surface overlay
- 3D multi-surface plots

### 8.2 Success Criteria

1. User can input data with 2 response variables
2. System fits 2 separate RSM models
3. Both contour plots display correctly side-by-side
4. User can toggle overlay view
5. Overlay is visually interpretable
6. No performance degradation (<2s to render)

### 8.3 Validation Test Cases

1. **Different Scale Responses**:
   - Y1: Yield (%) [Range: 60-95]
   - Y2: Purity (%) [Range: 85-99]
   - Verify normalization works

2. **Opposing Optimization Goals**:
   - Y1: Maximize quality (higher = better)
   - Y2: Minimize defects (lower = better)
   - Verify overlay shows trade-offs

3. **Highly Correlated Responses**:
   - Y1 and Y2 have correlation > 0.9
   - Verify overlay isn't redundant

---

## 9. API DESIGN

### 9.1 Request/Response Format

**Endpoint 1: Fit Multi-Response Models**

```
POST /api/rsm/fit-multi-model

Request:
{
  "data": [
    {"X1": -1, "X2": -1, "Y1": 10, "Y2": 0.5},
    {"X1": -1, "X2": 1, "Y1": 12, "Y2": 0.3},
    ...
  ],
  "factors": ["X1", "X2"],
  "responses": ["Y1", "Y2"],
  "alpha": 0.05
}

Response:
{
  "models": {
    "Y1": {
      "coefficients": {...},
      "anova": {...},
      "r_squared": 0.95,
      "diagnostics": {...}
    },
    "Y2": {
      "coefficients": {...},
      "anova": {...},
      "r_squared": 0.88,
      "diagnostics": {...}
    }
  },
  "n_responses": 2,
  "factors": ["X1", "X2"],
  "summary": {
    "all_significant": true,
    "min_r_squared": 0.88,
    "avg_r_squared": 0.915
  }
}
```

**Endpoint 2: Generate Multi-Surface Data**

```
POST /api/rsm/generate-multi-surface

Request:
{
  "models": {
    "Y1": { "coefficients": {...} },
    "Y2": { "coefficients": {...} }
  },
  "factors": ["X1", "X2"],
  "grid_resolution": 20,
  "x_range": [-2, 2],
  "y_range": [-2, 2],
  "normalize": true  // Apply z-score normalization
}

Response:
{
  "surfaces": {
    "Y1": [
      {"x": -2, "y": -2, "z": 10.5, "z_normalized": -1.2},
      {"x": -2, "y": -1.8, "z": 10.8, "z_normalized": -1.1},
      ...
    ],
    "Y2": [
      {"x": -2, "y": -2, "z": 0.5, "z_normalized": 0.8},
      ...
    ]
  },
  "normalization_params": {
    "Y1": {"mean": 15.2, "std": 4.3, "min": 5.1, "max": 25.6},
    "Y2": {"mean": 0.35, "std": 0.12, "min": 0.05, "max": 0.65}
  }
}
```

### 9.2 Error Handling

**Validation Errors**:
```json
{
  "error": "validation_error",
  "message": "Response Y2 has missing values in rows: [3, 7, 12]",
  "details": {
    "response": "Y2",
    "missing_rows": [3, 7, 12]
  }
}
```

**Model Fitting Errors**:
```json
{
  "error": "fitting_failed",
  "message": "Model for Y2 failed to converge",
  "details": {
    "response": "Y2",
    "reason": "insufficient_data",
    "suggestion": "Need at least 10 observations for 2-factor second-order model"
  }
}
```

---

## 10. INTEGRATION POINTS

### 10.1 Cross-Validation Integration

**Current**: Operates on single response (Lines 3006-3176 in rsm.py)

**Multi-Response Extension**:
```python
@router.post("/cross-validate-multi")
async def cross_validate_multi_model(request: MultiCrossValidationRequest):
    """
    Perform K-fold CV for each response independently
    Returns: Per-response CV metrics + correlation analysis
    """
    results = {}
    for response in request.responses:
        # Run CV for each response
        cv_result = await cross_validate_model(
            CrossValidationRequest(
                data=request.data,
                factors=request.factors,
                response=response,
                k_folds=request.k_folds
            )
        )
        results[response] = cv_result

    # Additional: Cross-response correlation analysis
    correlation_matrix = compute_response_correlation(
        request.data, request.responses
    )

    return {
        "individual_cv": results,
        "correlation_matrix": correlation_matrix,
        "interpretation": generate_multi_response_interpretation(results)
    }
```

**Question**: Should CV work per response or globally?
**Answer**: Per response, with additional correlation analysis across responses

### 10.2 Optimization Integration

**Existing**: Desirability functions handle multi-objective optimization (Lines 934-1059)

**Enhancement**: Link overlay visualization to desirability
```javascript
// In MultiResponseContourOverlay component
const showDesirabilityOverlay = () => {
  // Convert each response surface to desirability
  // Overlay composite desirability surface
  // Highlight optimal region
}
```

**Integration Flow**:
1. User views multi-response overlay
2. Identifies trade-off regions
3. Clicks "Optimize with Desirability"
4. Sets goals for each response
5. System shows optimal point on overlay
6. User can adjust goals and see real-time updates

### 10.3 Experiment Wizard Integration

**Current**: ExperimentWizard helps design experiments (RSM.jsx, Lines 2144-2150)

**Enhancement**: Support multi-response planning
```javascript
// In ExperimentWizard
const [multiResponseMode, setMultiResponseMode] = useState(false)
const [plannedResponses, setPlannedResponses] = useState(['Y1'])

// Wizard should ask:
// "How many responses will you measure?"
// "What are their names and expected ranges?"
// "Are they correlated?"

// Recommendation engine considers:
// - More responses = need more runs for validation
// - Correlated responses = can use smaller design
```

---

## 11. TESTING STRATEGY

### 11.1 Unit Tests

**Backend Tests** (`test_rsm_multi_response.py`):

```python
def test_fit_multi_model_basic():
    """Test fitting 2 responses with same factors"""
    request = {
        "data": [
            {"X1": -1, "X2": -1, "Y1": 10, "Y2": 0.5},
            {"X1": 1, "X2": -1, "Y1": 15, "Y2": 0.3},
            # ... (20 data points)
        ],
        "factors": ["X1", "X2"],
        "responses": ["Y1", "Y2"],
        "alpha": 0.05
    }
    response = client.post("/api/rsm/fit-multi-model", json=request)
    assert response.status_code == 200
    assert "Y1" in response.json()["models"]
    assert "Y2" in response.json()["models"]
    assert response.json()["models"]["Y1"]["r_squared"] > 0.7

def test_multi_surface_generation():
    """Test surface data generation for multiple responses"""
    # ... test implementation

def test_normalization_consistency():
    """Ensure normalized values are correct"""
    # ... verify z-score formula

def test_missing_data_handling():
    """Test behavior with missing response values"""
    # ... should handle gracefully or error clearly
```

**Frontend Tests** (`MultiResponseContourOverlay.test.jsx`):

```javascript
describe('MultiResponseContourOverlay', () => {
  it('renders 2 response surfaces side-by-side', () => {
    const surfacesData = {
      Y1: generateMockSurface(),
      Y2: generateMockSurface()
    }
    render(<MultiResponseContourOverlay surfacesData={surfacesData} />)
    expect(screen.getAllByRole('img')).toHaveLength(2)  // 2 plots
  })

  it('toggles response visibility', () => {
    // ... test show/hide functionality
  })

  it('applies normalization correctly', () => {
    // ... verify normalized z-values
  })
})
```

### 11.2 Integration Tests

**Test Case 1: End-to-End Multi-Response Workflow**

```javascript
test('User can fit and visualize 2 responses', async () => {
  // 1. Generate design
  await user.click(screen.getByText('Generate Design'))

  // 2. Add second response column
  await user.click(screen.getByText('Manage Responses'))
  await user.click(screen.getByText('Add Response'))
  await user.type(screen.getByPlaceholderText('Response name'), 'Y2')
  await user.click(screen.getByText('Save'))

  // 3. Enter data for both responses
  // ... (fill table)

  // 4. Fit models
  await user.click(screen.getByText('Fit Multi-Model'))
  await waitFor(() => {
    expect(screen.getByText('Y1 Model: R² = 0.95')).toBeInTheDocument()
    expect(screen.getByText('Y2 Model: R² = 0.88')).toBeInTheDocument()
  })

  // 5. View overlay
  await user.click(screen.getByText('Visualize'))
  await user.click(screen.getByText('Overlay Responses'))

  // 6. Verify both contours are displayed
  expect(screen.getByText('Y1')).toBeInTheDocument()
  expect(screen.getByText('Y2')).toBeInTheDocument()
})
```

**Test Case 2: Normalization Verification**

```javascript
test('Responses with different scales are normalized correctly', async () => {
  const surfacesData = {
    Y1: createSurface(10, 100),  // Range: 10-100
    Y2: createSurface(0.01, 0.5) // Range: 0.01-0.5
  }

  render(<MultiResponseContourOverlay
    surfacesData={surfacesData}
    normalize={true}
  />)

  // Both should now be on [-3, 3] scale (z-scores)
  const normalizedRanges = getNormalizedDataRanges()
  expect(normalizedRanges.Y1.min).toBeCloseTo(-3, 0.5)
  expect(normalizedRanges.Y1.max).toBeCloseTo(3, 0.5)
  expect(normalizedRanges.Y2.min).toBeCloseTo(-3, 0.5)
  expect(normalizedRanges.Y2.max).toBeCloseTo(3, 0.5)
})
```

### 11.3 Performance Tests

**Test Case 3: Rendering Speed**

```javascript
test('Multi-response overlay renders in <2 seconds', async () => {
  const largeSurfacesData = {
    Y1: generateMockSurface(50),  // 50x50 grid = 2500 points
    Y2: generateMockSurface(50),
    Y3: generateMockSurface(50)
  }

  const startTime = performance.now()
  render(<MultiResponseContourOverlay surfacesData={largeSurfacesData} />)
  await waitFor(() => {
    expect(screen.getByTestId('contour-plot')).toBeInTheDocument()
  })
  const endTime = performance.now()

  expect(endTime - startTime).toBeLessThan(2000)  // < 2 seconds
})
```

### 11.4 Validation Test Cases

**Dataset 1: Chemical Process Optimization**
```
Factors: Temperature (X1), Pressure (X2)
Responses:
  - Y1: Yield (%) [Range: 60-95]
  - Y2: Purity (%) [Range: 85-99]
Goals: Maximize both (but trade-off exists)

Expected: Overlay shows optimal region balances both
```

**Dataset 2: Manufacturing Quality**
```
Factors: Speed (X1), Temperature (X2)
Responses:
  - Y1: Throughput (units/hr) [Range: 100-500]
  - Y2: Defect Rate (%) [Range: 0.1-5.0]
Goals: Maximize Y1, Minimize Y2

Expected: Clear visual conflict, desirability helps compromise
```

**Dataset 3: Pharmaceutical Formulation**
```
Factors: pH (X1), Concentration (X2)
Responses:
  - Y1: Dissolution Rate (%) [Range: 20-98]
  - Y2: Stability (days) [Range: 30-180]
  - Y3: Cost ($/unit) [Range: 0.50-3.00]
Goals: Maximize Y1, Maximize Y2, Minimize Y3

Expected: 3-response overlay (Phase 4 feature)
```

---

## 12. IMPLEMENTATION TIMELINE

### Week 1: Data Infrastructure (MVP Foundation)

**Backend (3 days)**:
- [ ] Create `MultiRSMRequest` model
- [ ] Implement `/fit-multi-model` endpoint
- [ ] Add validation for multiple responses
- [ ] Write unit tests for multi-model fitting
- [ ] Test with 2-response dataset

**Frontend (2 days)**:
- [ ] Extend `tableData` state to support multiple response columns
- [ ] Create `MultiResponseManager` component (modal)
- [ ] Update table rendering to show N response columns
- [ ] Add validation for multi-response data entry
- [ ] Test data entry workflow

**Deliverable**: User can enter data with 2 responses and fit 2 separate models

---

### Week 2: Side-by-Side Visualization (Safe MVP)

**Backend (1 day)**:
- [ ] Create `/generate-multi-surface` endpoint
- [ ] Implement normalization utilities
- [ ] Return normalization parameters with surface data

**Frontend (4 days)**:
- [ ] Create `MultiResponseSideBySide` component
- [ ] Grid layout for 2 contour plots
- [ ] Pass individual surface data to each `ContourPlot`
- [ ] Add response selector (which to display)
- [ ] Style and polish

**Deliverable**: User can view 2 response surfaces side-by-side for comparison

---

### Week 3: Basic Overlay (Core Feature)

**Frontend (5 days)**:
- [ ] Create `MultiResponseContourOverlay` component
- [ ] Implement z-score normalization in frontend
- [ ] Create multiple Plotly contour traces
- [ ] Add transparency controls
- [ ] Implement toggle visibility per response
- [ ] Add legend with color coding
- [ ] Test overlay with different color schemes
- [ ] Handle edge cases (overlapping contours)

**Testing (2 days)**:
- [ ] Integration tests for overlay
- [ ] Performance testing
- [ ] User acceptance testing with sample datasets

**Deliverable**: User can overlay 2 normalized response surfaces with transparency

---

### Week 4: Advanced Features & Polish

**Advanced Overlay (3 days)**:
- [ ] Support 3+ responses
- [ ] Multiple normalization options (z-score, min-max, desirability)
- [ ] Color scheme selector
- [ ] Line style options (solid, dashed, dotted)
- [ ] Animated transitions when toggling responses

**Desirability Integration (2 days)**:
- [ ] Link overlay to desirability optimization
- [ ] Show optimal point on overlay
- [ ] Display composite desirability surface
- [ ] Interactive goal adjustment

**Documentation & Examples (2 days)**:
- [ ] User guide for multi-response overlay
- [ ] Example datasets and case studies
- [ ] Tooltip explanations in UI
- [ ] Best practices guide

**Deliverable**: Full multi-response overlay feature with advanced options

---

## 13. RISK ANALYSIS & MITIGATION

### Risk 1: Performance Degradation
**Probability**: Medium
**Impact**: High
**Mitigation**:
- Limit grid resolution (max 30x30 per response)
- Lazy load surface data
- Implement caching for computed surfaces
- Add loading indicators

### Risk 2: Visual Confusion
**Probability**: High
**Impact**: Medium
**Mitigation**:
- Start with side-by-side (safer)
- Require user confirmation before overlay
- Provide clear visual legend
- Add tutorial/help tooltips

### Risk 3: Normalization Misinterpretation
**Probability**: Medium
**Impact**: High
**Mitigation**:
- Display original and normalized values in tooltips
- Add warning when scales differ by >10x
- Provide "Raw Values" toggle
- Clear documentation on normalization methods

### Risk 4: Scope Creep
**Probability**: High
**Impact**: Medium
**Mitigation**:
- Stick to 2-response MVP for first release
- Create backlog for advanced features
- Get user feedback before adding complexity

---

## 14. OPEN QUESTIONS & DECISIONS NEEDED

### Question 1: Default Normalization Method
**Options**:
- A) Z-score (mean=0, std=1)
- B) Min-max (0 to 1)
- C) Desirability (goal-based, 0 to 1)

**Recommendation**: Z-score for overlay (preserves shape), with option to switch

### Question 2: Maximum Number of Responses
**Options**:
- A) 2 responses only (simplest)
- B) 3 responses (moderate complexity)
- C) Unlimited (user discretion)

**Recommendation**: MVP = 2, Phase 2 = 5, warn user if >5

### Question 3: Overlay vs Side-by-Side Default
**Options**:
- A) Default to side-by-side (safer)
- B) Default to overlay (more advanced)
- C) User preference setting

**Recommendation**: Start with side-by-side, "Try Overlay" button

### Question 4: Handle Missing Response Data
**Options**:
- A) Require complete data for all responses
- B) Allow partial data, fit models independently
- C) Impute missing values

**Recommendation**: Option B (most flexible), with clear warnings

---

## 15. CONCLUSION & NEXT STEPS

### Summary

The Multi-Response Overlay feature is **technically feasible** and builds on existing infrastructure (desirability functions already handle multiple responses). The main challenges are:

1. **UI/UX Design**: Making overlays interpretable without confusion
2. **Normalization**: Handling different response scales appropriately
3. **Performance**: Rendering multiple surface datasets efficiently

### Recommended Immediate Actions

1. **Prototype Side-by-Side View** (1 week)
   - Fastest path to value
   - Lower risk of user confusion
   - Foundation for overlay feature

2. **User Feedback Session**
   - Show side-by-side prototype
   - Ask: "Would overlay be useful?"
   - Gather normalization preference

3. **Create Test Datasets**
   - 3 real-world multi-response scenarios
   - Validate assumptions about scales and correlations

### Long-Term Vision

**Phase 5-6 Possibilities**:
- **3D Multi-Surface Visualization**: Stack response surfaces in 3D space
- **Interactive Slicing**: Slice through multi-dimensional response space
- **Pareto Front Visualization**: For conflicting objectives
- **Response Correlation Matrix**: Heatmap showing which responses move together
- **Automated Multi-Response Design**: Wizard suggests design based on N responses

### Final Recommendation

**Start with 2-Response Side-by-Side (Week 1-2), then iterate based on user feedback.** This de-risks the project while delivering immediate value. The overlay feature can be added incrementally once users validate the core workflow.

---

**Document Prepared By**: Claude (Sonnet 4.5)
**Date**: 2025-11-23
**Version**: 1.0
**Status**: Ready for Review & Implementation
