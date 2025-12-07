from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import numpy as np

router = APIRouter()


class DataValidationRequest(BaseModel):
    """Request model for data validation"""
    data: List[List[Any]]
    headers: List[str]
    expected_types: Optional[Dict[str, str]] = None  # {"column_name": "numeric" or "categorical"}


class DataValidationResponse(BaseModel):
    """Response model for data validation"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    column_types: Dict[str, str]
    row_count: int
    column_count: int


class ParsedDataResponse(BaseModel):
    """Response model for parsed data"""
    data: List[Dict[str, Any]]
    headers: List[str]
    column_types: Dict[str, str]
    row_count: int
    column_count: int
    sheet_names: Optional[List[str]] = None  # For Excel files


def detect_column_type(series: pd.Series) -> str:
    """
    Detect if a column is numeric or categorical

    Args:
        series: pandas Series to analyze

    Returns:
        "numeric" or "categorical"
    """
    # Remove null values for analysis
    non_null = series.dropna()

    if len(non_null) == 0:
        return "unknown"

    # Check if column is already numeric type
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Try to convert to numeric
    try:
        pd.to_numeric(non_null)
        numeric_count = len(non_null)
    except (ValueError, TypeError):
        # Count how many values can be converted to numeric
        numeric_count = 0
        for val in non_null:
            try:
                float(val)
                numeric_count += 1
            except (ValueError, TypeError):
                pass

    # If > 80% of values are numeric, consider it numeric
    numeric_ratio = numeric_count / len(non_null)
    return "numeric" if numeric_ratio > 0.8 else "categorical"


def validate_data_structure(
    df: pd.DataFrame,
    expected_types: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Validate data structure and detect issues

    Args:
        df: pandas DataFrame to validate
        expected_types: Optional dict of expected column types

    Returns:
        Dictionary with validation results
    """
    errors = []
    warnings = []
    column_types = {}

    # Check for empty dataframe
    if df.empty:
        errors.append("Data is empty")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "column_types": {},
            "row_count": 0,
            "column_count": 0
        }

    # Check for duplicate column names
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        errors.append(f"Duplicate column names found: {duplicates}")

    # Detect column types
    for col in df.columns:
        column_types[col] = detect_column_type(df[col])

    # Check against expected types if provided
    if expected_types:
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                warnings.append(f"Expected column '{col}' not found in data")
            elif column_types[col] != expected_type:
                warnings.append(
                    f"Column '{col}' is {column_types[col]} but expected {expected_type}"
                )

    # Check for missing values
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            pct = (count / len(df)) * 100
            if pct > 50:
                errors.append(f"Column '{col}' has {pct:.1f}% missing values")
            elif pct > 10:
                warnings.append(f"Column '{col}' has {pct:.1f}% missing values")

    # Check for minimum data requirements
    if len(df) < 3:
        warnings.append("Data has fewer than 3 rows - may be insufficient for analysis")

    # Check for constant columns
    for col in df.columns:
        if df[col].nunique() == 1:
            warnings.append(f"Column '{col}' has only one unique value")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "column_types": column_types,
        "row_count": len(df),
        "column_count": len(df.columns)
    }


@router.post("/csv", response_model=ParsedDataResponse)
async def parse_csv(file: UploadFile = File(...)):
    """
    Parse a CSV file and return structured data

    Args:
        file: Uploaded CSV file

    Returns:
        Parsed data with headers, types, and metadata
    """
    try:
        # Validate file extension
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="File must be a CSV file (.csv extension)"
            )

        # Read file content
        content = await file.read()

        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise HTTPException(
                status_code=400,
                detail="Unable to decode CSV file. Please ensure it's properly encoded."
            )

        # Detect column types
        column_types = {col: detect_column_type(df[col]) for col in df.columns}

        # Convert DataFrame to list of dictionaries
        # Replace NaN with None for JSON serialization
        data = df.replace({np.nan: None}).to_dict('records')

        return ParsedDataResponse(
            data=data,
            headers=df.columns.tolist(),
            column_types=column_types,
            row_count=len(df),
            column_count=len(df.columns)
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing CSV: {str(e)}")


@router.post("/excel", response_model=ParsedDataResponse)
async def parse_excel(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = None
):
    """
    Parse an Excel file and return structured data

    Args:
        file: Uploaded Excel file (.xlsx or .xls)
        sheet_name: Optional sheet name (defaults to first sheet)

    Returns:
        Parsed data with headers, types, and metadata
    """
    try:
        # Validate file extension
        if not file.filename.lower().endswith(('.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="File must be an Excel file (.xlsx or .xls extension)"
            )

        # Read file content
        content = await file.read()

        # Read Excel file
        excel_file = pd.ExcelFile(io.BytesIO(content))
        sheet_names = excel_file.sheet_names

        # Use first sheet if not specified
        if sheet_name is None:
            sheet_name = sheet_names[0]
        elif sheet_name not in sheet_names:
            raise HTTPException(
                status_code=400,
                detail=f"Sheet '{sheet_name}' not found. Available sheets: {sheet_names}"
            )

        # Parse the specified sheet
        df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)

        # Detect column types
        column_types = {col: detect_column_type(df[col]) for col in df.columns}

        # Convert DataFrame to list of dictionaries
        # Replace NaN with None for JSON serialization
        data = df.replace({np.nan: None}).to_dict('records')

        return ParsedDataResponse(
            data=data,
            headers=df.columns.tolist(),
            column_types=column_types,
            row_count=len(df),
            column_count=len(df.columns),
            sheet_names=sheet_names
        )

    except ValueError as e:
        if "Excel file format" in str(e):
            raise HTTPException(
                status_code=400,
                detail="Invalid Excel file format"
            )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing Excel: {str(e)}")


@router.post("/validate", response_model=DataValidationResponse)
async def validate_data(request: DataValidationRequest):
    """
    Validate data structure and detect potential issues

    Args:
        request: Data validation request with data, headers, and optional expected types

    Returns:
        Validation results with errors, warnings, and detected column types
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data, columns=request.headers)

        # Validate structure
        result = validate_data_structure(df, request.expected_types)

        return DataValidationResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error validating data: {str(e)}"
        )
