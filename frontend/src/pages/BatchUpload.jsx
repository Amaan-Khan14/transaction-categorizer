import { useState } from "react";
import { Doughnut } from "react-chartjs-2";
import Chart from "chart.js/auto";
import { predictBatch } from "../services/api";
import "./BatchUpload.css";

function BatchUpload() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const downloadCSV = () => {
    if (!result || !result.results) return;

    // Create CSV content
    const headers = [
      "transaction_id",
      "transaction",
      "predicted_category",
      "confidence",
    ];
    const rows = result.results.map((r) => [
      r.transaction_id,
      `"${r.transaction.replace(/"/g, '""')}"`, // Escape quotes in transaction
      r.predicted_category,
      r.confidence.toFixed(4),
    ]);

    const csvContent = [
      headers.join(","),
      ...rows.map((row) => row.join(",")),
    ].join("\n");

    // Create blob and download
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute(
      "download",
      `batch_results_${new Date().getTime()}.csv`
    );
    link.style.visibility = "hidden";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictBatch(file);
      setResult(data);
    } catch (err) {
      setError(
        err.response?.data?.detail || "Upload failed. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="batch-upload">
      <div className="batch-header">
        <h2>Batch Processing</h2>
        <p className="batch-subtitle">
          Upload a CSV file with transactions for bulk categorization
        </p>
      </div>

      <div className="batch-container">
        <div className="upload-section card">
          <h3>Upload File</h3>
          <p className="text-gray text-sm mb-4">
            CSV format: transaction_id, transaction
          </p>

          <form onSubmit={handleUpload}>
            <div className="file-input-wrapper">
              <input
                type="file"
                accept=".csv,.xlsx"
                onChange={handleFileChange}
                disabled={loading}
                id="file-upload"
                className="file-input"
              />
              <label htmlFor="file-upload" className="file-label">
                {file ? file.name : "Choose CSV or Excel file..."}
              </label>
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !file}
            >
              {loading ? "Processing..." : "Process Batch"}
            </button>
          </form>
        </div>

        {error && (
          <div className="alert alert-error card">
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <div className="result-section card">
            <h3>Batch Results</h3>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "30px",
                marginTop: "20px",
              }}
            >
              {/* Left side: Chart */}
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "center",
                  alignItems: "center",
                  padding: "20px",
                }}
              >
                {result.summary.category_distribution &&
                Object.keys(result.summary.category_distribution).length > 0 ? (
                  <div style={{ width: "100%" }}>
                    <div style={{ width: "100%", height: "280px" }}>
                      <Doughnut
                        data={{
                          labels: Object.entries(
                            result.summary.category_distribution,
                          )
                            .sort(([, a], [, b]) => b - a)
                            .map(([category]) => category),
                          datasets: [
                            {
                              label: "Transactions",
                              data: Object.entries(
                                result.summary.category_distribution,
                              )
                                .sort(([, a], [, b]) => b - a)
                                .map(([, count]) => count),
                              backgroundColor: [
                                "#8B5CF6",
                                "#EC4899",
                                "#F59E0B",
                                "#10B981",
                                "#3B82F6",
                                "#6366F1",
                                "#14B8A6",
                                "#F97316",
                              ],
                              borderColor: "#fff",
                              borderWidth: 3,
                            },
                          ],
                        }}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          plugins: {
                            legend: {
                              display: false,
                            },
                            tooltip: {
                              callbacks: {
                                label: function (context) {
                                  const label = context.label || "";
                                  const value = context.parsed || 0;
                                  const total = context.dataset.data.reduce(
                                    (a, b) => a + b,
                                    0,
                                  );
                                  const percentage = (
                                    (value / total) *
                                    100
                                  ).toFixed(1);
                                  return `${label}: ${value} (${percentage}%)`;
                                },
                              },
                              backgroundColor: "rgba(0, 0, 0, 0.8)",
                              padding: 12,
                              titleFont: { size: 13 },
                              bodyFont: { size: 12 },
                            },
                          },
                        }}
                      />
                    </div>
                    <div
                      style={{
                        display: "flex",
                        flexWrap: "wrap",
                        gap: "10px",
                        justifyContent: "center",
                        marginTop: "16px",
                      }}
                    >
                      {Object.entries(result.summary.category_distribution)
                        .sort(([, a], [, b]) => b - a)
                        .map(([category, count], index) => {
                          const colors = [
                            "#8B5CF6",
                            "#EC4899",
                            "#F59E0B",
                            "#10B981",
                            "#3B82F6",
                            "#6366F1",
                            "#14B8A6",
                            "#F97316",
                          ];
                          return (
                            <div
                              key={category}
                              style={{
                                display: "flex",
                                alignItems: "center",
                                gap: "8px",
                                fontSize: "12px",
                              }}
                            >
                              <div
                                style={{
                                  width: "12px",
                                  height: "12px",
                                  borderRadius: "3px",
                                  backgroundColor:
                                    colors[index % colors.length],
                                }}
                              />
                              <span style={{ color: "var(--text-secondary)" }}>
                                {category}
                              </span>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                ) : (
                  <p style={{ color: "var(--text-secondary)" }}>
                    No category distribution data
                  </p>
                )}
              </div>

              {/* Right side: Results */}
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                }}
              >
                <div>
                  <div
                    style={{
                      marginBottom: "20px",
                    }}
                  >
                    <p
                      style={{
                        color: "var(--text-secondary)",
                        fontSize: "14px",
                        marginBottom: "8px",
                      }}
                    >
                      Total Processed
                    </p>
                    <div
                      style={{
                        fontSize: "36px",
                        fontWeight: "600",
                        color: "var(--primary-purple)",
                      }}
                    >
                      {result.total_transactions}
                    </div>
                  </div>

                  {result.summary.category_distribution &&
                    Object.keys(result.summary.category_distribution).length >
                      0 && (
                      <div style={{ marginTop: "20px", width: "100%" }}>
                        <h4
                          style={{
                            marginBottom: "16px",
                            fontSize: "13px",
                            fontWeight: "600",
                            textTransform: "uppercase",
                            letterSpacing: "0.5px",
                            color: "var(--text-secondary)",
                          }}
                        >
                          Distribution
                        </h4>
                        <div
                          style={{
                            display: "grid",
                            gridTemplateColumns: "1fr 1fr",
                            gap: "14px",
                          }}
                        >
                          {Object.entries(result.summary.category_distribution)
                            .sort(([, a], [, b]) => b - a)
                            .map(([category, count]) => {
                              const percentage = (
                                (count / result.total_transactions) *
                                100
                              ).toFixed(1);
                              return (
                                <div
                                  key={category}
                                  style={{
                                    padding: "14px",
                                    backgroundColor: "var(--bg-tertiary)",
                                    borderRadius: "8px",
                                    border: "1px solid var(--bg-secondary)",
                                  }}
                                >
                                  <div
                                    style={{
                                      fontSize: "20px",
                                      color: "var(--text-secondary)",
                                      marginBottom: "8px",
                                      fontWeight: "500",
                                    }}
                                  >
                                    {category}
                                  </div>
                                  <div
                                    style={{
                                      fontSize: "26px",
                                      fontWeight: "700",
                                      color: "var(--primary-purple)",
                                      marginBottom: "8px",
                                    }}
                                  >
                                    {count}
                                  </div>
                                  <div
                                    style={{
                                      fontSize: "11px",
                                      color: "var(--text-tertiary)",
                                      marginBottom: "10px",
                                    }}
                                  >
                                    {percentage}%
                                  </div>
                                  <div
                                    style={{
                                      width: "100%",
                                      height: "6px",
                                      backgroundColor: "var(--bg-secondary)",
                                      borderRadius: "3px",
                                      overflow: "hidden",
                                    }}
                                  >
                                    <div
                                      style={{
                                        height: "100%",
                                        width: `${percentage}%`,
                                        backgroundColor:
                                          "var(--primary-purple)",
                                        transition: "width 0.3s ease",
                                      }}
                                    />
                                  </div>
                                </div>
                              );
                            })}
                        </div>
                      </div>
                    )}
                </div>

                <button
                  onClick={downloadCSV}
                  className="btn btn-success"
                  style={{
                    marginTop: "20px",
                    width: "100%",
                    textAlign: "center",
                  }}
                >
                  Download Results (CSV)
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default BatchUpload;
