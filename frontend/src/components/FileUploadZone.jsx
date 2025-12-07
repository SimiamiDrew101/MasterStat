import React, { useState, useRef } from 'react';
import { Upload, FileText, X, AlertCircle, CheckCircle2 } from 'lucide-react';
import { parseCSV, parseExcel, validateFile, detectColumnTypes, convertToTableFormat } from '../utils/fileParser';

const FileUploadZone = ({ onDataImport, expectedColumns = [] }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const [parsing, setParsing] = useState(false);
  const [parsedData, setParsedData] = useState(null);
  const [columnMapping, setColumnMapping] = useState({});
  const [showPreview, setShowPreview] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileSelect(droppedFile);
    }
  };

  const handleFileSelect = async (selectedFile) => {
    setError(null);
    setFile(null);
    setParsedData(null);

    // Validate file
    const validation = validateFile(selectedFile);
    if (!validation.valid) {
      setError(validation.error);
      return;
    }

    setFile(selectedFile);
    setParsing(true);

    try {
      // Parse file based on type
      const fileExtension = selectedFile.name.split('.').pop().toLowerCase();
      let result;

      if (fileExtension === 'csv') {
        result = await parseCSV(selectedFile);
      } else if (fileExtension === 'xlsx' || fileExtension === 'xls') {
        result = await parseExcel(selectedFile);
      }

      if (result.error) {
        setError(result.error);
        setParsing(false);
        return;
      }

      // Detect column types
      const columnTypes = detectColumnTypes(result.data, result.headers);

      setParsedData({
        ...result,
        columnTypes
      });

      // Auto-map columns if expected columns are provided
      if (expectedColumns.length > 0) {
        const autoMapping = {};
        expectedColumns.forEach((expected, index) => {
          if (result.headers[index]) {
            autoMapping[expected] = result.headers[index];
          }
        });
        setColumnMapping(autoMapping);
      }

      setShowPreview(true);
      setParsing(false);
    } catch (err) {
      setError(`Unexpected error: ${err.message}`);
      setParsing(false);
    }
  };

  const handleFileInputClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileInputChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      handleFileSelect(selectedFile);
    }
  };

  const handleColumnMappingChange = (mappedColumn, selectedColumn) => {
    setColumnMapping(prev => ({
      ...prev,
      [mappedColumn]: selectedColumn
    }));
  };

  const handleImport = () => {
    if (!parsedData || !onDataImport) return;

    // Get mapped columns
    const factorColumns = [];
    let responseColumn = null;

    Object.entries(columnMapping).forEach(([key, value]) => {
      if (key === 'response') {
        responseColumn = value;
      } else {
        factorColumns.push(value);
      }
    });

    if (!responseColumn) {
      setError('Please map a response column');
      return;
    }

    // Convert to table format
    const tableData = convertToTableFormat(parsedData.data, factorColumns, responseColumn);

    // Call parent callback
    onDataImport({
      tableData,
      factorColumns,
      responseColumn,
      rawData: parsedData.data
    });

    // Reset state
    setShowPreview(false);
    setFile(null);
    setParsedData(null);
    setColumnMapping({});
  };

  const handleCancel = () => {
    setFile(null);
    setParsedData(null);
    setShowPreview(false);
    setError(null);
    setColumnMapping({});
  };

  return (
    <div className="space-y-4">
      {/* Upload Zone */}
      {!showPreview && (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleFileInputClick}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
            transition-all duration-200
            ${isDragging
              ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
              : 'border-gray-300 dark:border-gray-600 hover:border-blue-400 dark:hover:border-blue-500'
            }
            ${parsing ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileInputChange}
            className="hidden"
            disabled={parsing}
          />

          <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />

          {parsing ? (
            <div>
              <p className="text-lg font-medium text-gray-700 dark:text-gray-300">
                Parsing file...
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                Please wait
              </p>
            </div>
          ) : (
            <div>
              <p className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">
                Drop file here or click to browse
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Supports CSV, Excel (.xlsx, .xls) up to 10MB
              </p>
            </div>
          )}

          {file && !parsing && (
            <div className="mt-4 flex items-center justify-center gap-2 text-sm text-blue-600 dark:text-blue-400">
              <FileText className="w-4 h-4" />
              <span>{file.name}</span>
            </div>
          )}
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="flex items-start gap-3 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-900 dark:text-red-200">
              Import Error
            </p>
            <p className="text-sm text-red-700 dark:text-red-300 mt-1">
              {error}
            </p>
          </div>
          <button
            onClick={() => setError(null)}
            className="ml-auto text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-200"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Preview and Column Mapping */}
      {showPreview && parsedData && (
        <div className="space-y-4">
          {/* Success message */}
          <div className="flex items-center gap-3 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
            <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400" />
            <div>
              <p className="text-sm font-medium text-green-900 dark:text-green-200">
                File parsed successfully
              </p>
              <p className="text-sm text-green-700 dark:text-green-300">
                {parsedData.data.length} rows, {parsedData.headers.length} columns
              </p>
            </div>
          </div>

          {/* Column Mapping */}
          <div className="bg-white dark:bg-slate-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Map Columns
            </h3>

            <div className="space-y-3">
              {expectedColumns.map((expectedCol) => (
                <div key={expectedCol} className="flex items-center gap-4">
                  <label className="w-32 text-sm font-medium text-gray-700 dark:text-gray-300 capitalize">
                    {expectedCol}:
                  </label>
                  <select
                    value={columnMapping[expectedCol] || ''}
                    onChange={(e) => handleColumnMappingChange(expectedCol, e.target.value)}
                    className="flex-1 px-3 py-2 bg-white dark:bg-slate-700 border border-gray-300 dark:border-gray-600 rounded-md text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">-- Select Column --</option>
                    {parsedData.headers.map((header) => (
                      <option key={header} value={header}>
                        {header} ({parsedData.columnTypes[header]})
                      </option>
                    ))}
                  </select>
                </div>
              ))}
            </div>
          </div>

          {/* Data Preview */}
          <div className="bg-white dark:bg-slate-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Data Preview (first 10 rows)
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    {parsedData.headers.map((header) => (
                      <th
                        key={header}
                        className="px-4 py-2 text-left font-medium text-gray-900 dark:text-white"
                      >
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {parsedData.data.slice(0, 10).map((row, rowIndex) => (
                    <tr
                      key={rowIndex}
                      className="border-b border-gray-100 dark:border-gray-700 last:border-0"
                    >
                      {parsedData.headers.map((header) => (
                        <td
                          key={header}
                          className="px-4 py-2 text-gray-700 dark:text-gray-300"
                        >
                          {row[header] !== null && row[header] !== undefined
                            ? String(row[header])
                            : '-'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleImport}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md font-medium transition-colors"
            >
              Import Data
            </button>
            <button
              onClick={handleCancel}
              className="px-6 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-900 dark:text-white rounded-md font-medium transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUploadZone;
