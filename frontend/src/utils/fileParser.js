import Papa from 'papaparse';
import * as XLSX from 'xlsx';

/**
 * Parse CSV file using PapaParse
 * @param {File} file - CSV file to parse
 * @returns {Promise<{data: Array, headers: Array, error: string|null}>}
 */
export const parseCSV = (file) => {
  return new Promise((resolve) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (results.errors.length > 0) {
          resolve({
            data: [],
            headers: [],
            error: `CSV parsing errors: ${results.errors.map(e => e.message).join(', ')}`
          });
        } else {
          resolve({
            data: results.data,
            headers: results.meta.fields || [],
            error: null
          });
        }
      },
      error: (error) => {
        resolve({
          data: [],
          headers: [],
          error: `Failed to parse CSV: ${error.message}`
        });
      }
    });
  });
};

/**
 * Parse Excel file using xlsx library
 * @param {File} file - Excel file to parse
 * @returns {Promise<{data: Array, headers: Array, sheetNames: Array, error: string|null}>}
 */
export const parseExcel = (file) => {
  return new Promise((resolve) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { type: 'array' });

        // Get first sheet by default
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];

        // Convert to JSON
        const jsonData = XLSX.utils.sheet_to_json(worksheet, {
          header: 1,
          defval: null,
          blankrows: false
        });

        if (jsonData.length === 0) {
          resolve({
            data: [],
            headers: [],
            sheetNames: workbook.SheetNames,
            error: 'Excel file is empty'
          });
          return;
        }

        // First row is headers
        const headers = jsonData[0];

        // Remaining rows are data
        const rows = jsonData.slice(1).map(row => {
          const obj = {};
          headers.forEach((header, index) => {
            obj[header] = row[index] !== undefined ? row[index] : null;
          });
          return obj;
        });

        resolve({
          data: rows,
          headers: headers,
          sheetNames: workbook.SheetNames,
          error: null
        });
      } catch (error) {
        resolve({
          data: [],
          headers: [],
          sheetNames: [],
          error: `Failed to parse Excel: ${error.message}`
        });
      }
    };

    reader.onerror = () => {
      resolve({
        data: [],
        headers: [],
        sheetNames: [],
        error: 'Failed to read file'
      });
    };

    reader.readAsArrayBuffer(file);
  });
};

/**
 * Validate file type and size
 * @param {File} file - File to validate
 * @param {Array<string>} allowedTypes - Allowed MIME types
 * @param {number} maxSizeMB - Maximum file size in MB
 * @returns {{valid: boolean, error: string|null}}
 */
export const validateFile = (file, allowedTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'], maxSizeMB = 10) => {
  // Check file size
  const maxSizeBytes = maxSizeMB * 1024 * 1024;
  if (file.size > maxSizeBytes) {
    return {
      valid: false,
      error: `File size exceeds ${maxSizeMB}MB limit`
    };
  }

  // Check file type
  const fileExtension = file.name.split('.').pop().toLowerCase();
  const isCSV = fileExtension === 'csv' || file.type === 'text/csv';
  const isExcel = fileExtension === 'xlsx' || fileExtension === 'xls' ||
                  file.type === 'application/vnd.ms-excel' ||
                  file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';

  if (!isCSV && !isExcel) {
    return {
      valid: false,
      error: 'Only CSV and Excel (.xlsx, .xls) files are supported'
    };
  }

  return {
    valid: true,
    error: null
  };
};

/**
 * Auto-detect column types (numeric vs categorical)
 * @param {Array} data - Array of data objects
 * @param {Array} headers - Column headers
 * @returns {Object} - Map of column names to types {columnName: 'numeric'|'categorical'}
 */
export const detectColumnTypes = (data, headers) => {
  const types = {};

  headers.forEach(header => {
    const values = data.map(row => row[header]).filter(v => v !== null && v !== undefined && v !== '');

    if (values.length === 0) {
      types[header] = 'unknown';
      return;
    }

    // Check if all values are numbers
    const numericCount = values.filter(v => typeof v === 'number' || !isNaN(parseFloat(v))).length;
    const numericRatio = numericCount / values.length;

    // If > 80% of values are numeric, consider it numeric
    types[header] = numericRatio > 0.8 ? 'numeric' : 'categorical';
  });

  return types;
};

/**
 * Convert data to table format for MasterStat
 * @param {Array} data - Parsed data
 * @param {Array} factorColumns - Column names to use as factors
 * @param {string} responseColumn - Column name to use as response
 * @returns {Array} - Table data in MasterStat format
 */
export const convertToTableFormat = (data, factorColumns, responseColumn) => {
  return data.map(row => {
    const tableRow = [];

    // Add factor values
    factorColumns.forEach(col => {
      tableRow.push(row[col]);
    });

    // Add response value
    tableRow.push(row[responseColumn]);

    return tableRow;
  });
};
