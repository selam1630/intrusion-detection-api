import React, { useState } from "react";

function App() {
  const [features, setFeatures] = useState(Array(41).fill(""));
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeModel, setActiveModel] = useState(null);

  const exampleNormal = [0,0,1,1,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,511,511,0.0,0.0,0.0,0.0,1.0,0.0,0.0,255,255,1.0,0.0,0.0,0.0,0.0,0.0,0.0];
  const exampleAttack = [0,0,2,2,239,486,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1.0,1.0,0.0,0.0,0.0,1.0,1.0,255,255,0.0,0.0,0.0,0.0,1.0,1.0,0.0];

  const handleChange = (index, value) => {
    const newFeatures = [...features];
    newFeatures[index] = value;
    setFeatures(newFeatures);
  };

  const handlePreset = (preset) => {
    setFeatures(preset);
    setResult(null);
  };

  const handlePredict = async (model) => {
    setLoading(true);
    setActiveModel(model);
    setResult(null);
    try {
      const response = await fetch(`https://intrusion-detection-api-3.onrender.com/predict/${model}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features.map(Number)),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      setResult({ error: "Failed to fetch prediction" });
    }
    setLoading(false);
  };

  const handleClear = () => {
    setFeatures(Array(41).fill(""));
    setResult(null);
    setActiveModel(null);
  };

  // Styles object
  const styles = {
    // Global styles
    body: {
      margin: 0,
      padding: 0,
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif",
      background: "linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)",
      minHeight: "100vh",
      overflowX: "hidden",
    },
    
    // Container styles
    container: {
      minHeight: "100vh",
      background: "linear-gradient(to bottom right, #0f172a, #1e1b4b, #312e81)",
      display: "flex",
      flexDirection: "column" ,
      alignItems: "center",
      justifyContent: "center",
      padding: "1rem",
    },
    
    // Header styles
    header: {
      textAlign: "center" ,
      marginBottom: "2rem",
    },
    headerIcon: {
      width: "3rem",
      height: "3rem",
      background: "linear-gradient(to right, #06b6d4, #3b82f6)",
      borderRadius: "1rem",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      marginRight: "1rem",
      boxShadow: "0 10px 25px rgba(0, 0, 0, 0.3)",
    },
    title: {
      background: "linear-gradient(to right, #06b6d4, #3b82f6)",
      WebkitBackgroundClip: "text",
      WebkitTextFillColor: "transparent",
      backgroundClip: "text",
      fontSize: "2.5rem",
      fontWeight: "bold",
      margin: 0,
    },
    subtitle: {
      color: "#d1d5db",
      fontSize: "1rem",
      maxWidth: "32rem",
      margin: "0 auto",
    },
    
    // Main card
    mainCard: {
      background: "linear-gradient(to bottom right, rgba(31, 41, 55, 0.8), rgba(17, 24, 39, 0.8))",
      backdropFilter: "blur(10px)",
      padding: "1.5rem",
      borderRadius: "1.5rem",
      boxShadow: "0 20px 25px rgba(0, 0, 0, 0.5)",
      border: "1px solid rgba(55, 65, 81, 0.5)",
      width: "100%",
      maxWidth: "72rem",
    },
    
    // Button styles
    buttonGroup: {
      display: "flex",
      flexDirection: "column",
      gap: "1rem",
      marginBottom: "1.5rem",
    },
    buttonRow: {
      display: "flex",
      gap: "1rem",
    },
    button: {
      flex: 1,
      padding: "1rem 1.5rem",
      borderRadius: "0.75rem",
      border: "none",
      fontWeight: "600",
      fontSize: "1rem",
      cursor: "pointer",
      transition: "all 0.3s ease",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      gap: "0.75rem",
      boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
    },
    buttonHover: {
      transform: "translateY(-2px)",
    },
    normalButton: {
      background: "linear-gradient(to right, #10b981, #059669)",
      color: "white",
    },
    attackButton: {
      background: "linear-gradient(to right, #f43f5e, #dc2626)",
      color: "white",
    },
    clearButton: {
      background: "linear-gradient(to right, #4b5563, #374151)",
      color: "white",
    },
    predictButton: {
      background: "linear-gradient(to right, #7c3aed, #4f46e5)",
      color: "white",
      padding: "1.25rem 2rem",
      fontSize: "1.125rem",
    },
    treeButton: {
      background: "linear-gradient(to right, #2563eb, #06b6d4)",
      color: "white",
      padding: "1.25rem 2rem",
      fontSize: "1.125rem",
    },
    
    // Feature grid
    featuresGrid: {
      display: "grid",
      gridTemplateColumns: "repeat(2, 1fr)",
      gap: "1rem",
      maxHeight: "400px",
      overflowY: "auto",
      marginBottom: "2rem",
      padding: "1rem",
      backgroundColor: "rgba(17, 24, 39, 0.5)",
      borderRadius: "1rem",
      border: "1px solid rgba(55, 65, 81, 0.3)",
    },
    featureInputContainer: {
      position: "relative",
    },
    featureLabel: {
      display: "block",
      fontSize: "0.75rem",
      color: "#9ca3af",
      marginBottom: "0.25rem",
      fontWeight: "500",
    },
    featureInput: {
      width: "100%",
      backgroundColor: "rgba(31, 41, 55, 0.7)",
      border: "2px solid #374151",
      borderRadius: "0.75rem",
      padding: "0.75rem 1rem",
      color: "white",
      fontSize: "0.875rem",
      outline: "none",
      transition: "all 0.2s",
    },
    featureInputFocus: {
      borderColor: "#06b6d4",
      boxShadow: "0 0 0 3px rgba(6, 182, 212, 0.1), 0 0 20px rgba(6, 182, 212, 0.2)",
    },
    featureNumber: {
      position: "absolute",
      top: "-0.5rem",
      right: "-0.5rem",
      width: "1.5rem",
      height: "1.5rem",
      backgroundColor: "#1f2937",
      borderRadius: "50%",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: "0.75rem",
      fontWeight: "bold",
      color: "#06b6d4",
      border: "1px solid rgba(6, 182, 212, 0.3)",
    },
    
    // Loading
    loadingSpinner: {
      width: "4rem",
      height: "4rem",
      border: "4px solid rgba(6, 182, 212, 0.3)",
      borderTop: "4px solid #06b6d4",
      borderRadius: "50%",
      animation: "spin 1s linear infinite",
      marginBottom: "1rem",
    },
    
    // Results
    resultCard: {
      marginTop: "2rem",
      padding: "2rem",
      borderRadius: "1rem",
      boxShadow: "0 10px 15px rgba(0, 0, 0, 0.3)",
      transition: "all 0.5s ease",
      animation: "fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1)",
    },
    normalResult: {
      background: "linear-gradient(to right, rgba(5, 150, 105, 0.4), rgba(16, 185, 129, 0.4))",
      border: "1px solid rgba(5, 150, 105, 0.5)",
    },
    attackResult: {
      background: "linear-gradient(to right, rgba(220, 38, 38, 0.4), rgba(244, 63, 94, 0.4))",
      border: "1px solid rgba(220, 38, 38, 0.5)",
    },
    errorResult: {
      background: "linear-gradient(to right, rgba(220, 38, 38, 0.4), rgba(244, 63, 94, 0.4))",
      border: "1px solid rgba(220, 38, 38, 0.5)",
    },
    resultIcon: {
      width: "3rem",
      height: "3rem",
      borderRadius: "0.75rem",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
    
    // Scrollbar
    scrollbar: {
      scrollbarWidth: "thin",
      scrollbarColor: "#3b82f6 rgba(30, 41, 59, 0.5)",
    },
    
    // Animations
    fadeIn: {
      animation: "fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1)",
    },
    pulse: {
      animation: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
    },
  };

  // Media queries for responsive design
  const isMobile = window.innerWidth < 640;
  const isTablet = window.innerWidth < 1024;

  // Adjust grid based on screen size
  const gridStyle = {
    ...styles.featuresGrid,
    gridTemplateColumns: isMobile ? "repeat(2, 1fr)" : 
                         isTablet ? "repeat(3, 1fr)" : "repeat(6, 1fr)",
  };

  return (
    <>
      {/* Global styles as a style tag */}
      <style>
        {`
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
          
          * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
          }
          
          body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            min-height: 100vh;
            overflow-x: hidden;
          }
          
          /* Custom Scrollbar */
          .custom-scrollbar::-webkit-scrollbar {
            width: 10px;
            height: 10px;
          }
          
          .custom-scrollbar::-webkit-scrollbar-track {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 10px;
            margin: 4px;
          }
          
          .custom-scrollbar::-webkit-scrollbar-thumb {
            background: linear-gradient(to bottom, #06b6d4, #3b82f6);
            border-radius: 10px;
            border: 2px solid rgba(30, 41, 59, 0.8);
          }
          
          .custom-scrollbar::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(to bottom, #0891b2, #2563eb);
          }
          
          .custom-scrollbar::-webkit-scrollbar-corner {
            background: transparent;
          }
          
          /* Animations */
          @keyframes fadeIn {
            from {
              opacity: 0;
              transform: translateY(20px) scale(0.95);
            }
            to {
              opacity: 1;
              transform: translateY(0) scale(1);
            }
          }
          
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          
          @keyframes pulse {
            0%, 100% {
              opacity: 1;
            }
            50% {
              opacity: 0.5;
            }
          }
          
          .animate-fadeIn {
            animation: fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
          }
          
          .animate-pulse {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
          }
          
          /* Responsive */
          @media (min-width: 640px) {
            .features-grid {
              grid-template-columns: repeat(3, 1fr);
            }
          }
          
          @media (min-width: 768px) {
            .features-grid {
              grid-template-columns: repeat(4, 1fr);
            }
            .button-group {
              flex-direction: row;
            }
          }
          
          @media (min-width: 1024px) {
            .features-grid {
              grid-template-columns: repeat(6, 1fr);
            }
          }
          
          /* Glass effect */
          .glass {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
          }
        `}
      </style>

      <div style={styles.container}>
        {/* Header */}
        <div style={styles.header}>
          <div style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", marginBottom: "1rem" }}>
            <div style={styles.headerIcon}>
              <svg style={{ width: "1.5rem", height: "1.5rem", color: "white" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <h1 style={styles.title}>Intrusion Detection System</h1>
          </div>
          <p style={styles.subtitle}>
            Advanced machine learning models to detect network intrusions in real-time
          </p>
        </div>

        {/* Main Card */}
        <div style={styles.mainCard}>
          {/* Quick Actions */}
          <div style={{ marginBottom: "2rem" }}>
            <div style={styles.buttonGroup}>
              <div style={styles.buttonRow}>
                <button
                  onClick={() => handlePreset(exampleNormal)}
                  style={styles.button}
                  onMouseOver={(e) => e.currentTarget.style.transform = "translateY(-2px)"}
                  onMouseOut={(e) => e.currentTarget.style.transform = "translateY(0)"}
                >
                  <svg style={{ width: "1.25rem", height: "1.25rem" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Load Normal Traffic Example
                </button>
                <button
                  onClick={() => handlePreset(exampleAttack)}
                  style={{ ...styles.button, ...styles.attackButton }}
                  onMouseOver={(e) => e.currentTarget.style.transform = "translateY(-2px)"}
                  onMouseOut={(e) => e.currentTarget.style.transform = "translateY(0)"}
                >
                  <svg style={{ width: "1.25rem", height: "1.25rem" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.104 16.5c-.77.833.192 2.5 1.732 2.5z" />
                  </svg>
                  Load Attack Traffic Example
                </button>
                <button
                  onClick={handleClear}
                  style={{ ...styles.button, ...styles.clearButton }}
                  onMouseOver={(e) => e.currentTarget.style.transform = "translateY(-2px)"}
                  onMouseOut={(e) => e.currentTarget.style.transform = "translateY(0)"}
                >
                  <svg style={{ width: "1.25rem", height: "1.25rem" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  Clear All
                </button>
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
              <h2 style={{ fontSize: "1.5rem", fontWeight: "bold", color: "white" }}>
                Network Features <span style={{ color: "#06b6d4" }}>(41 Parameters)</span>
              </h2>
              <span style={{ fontSize: "0.875rem", color: "#9ca3af", backgroundColor: "#1f2937", padding: "0.25rem 0.75rem", borderRadius: "9999px" }}>
                Scroll to view all
              </span>
            </div>
          </div>

          {/* Features Grid */}
          <div className="custom-scrollbar features-grid" style={gridStyle}>
            {features.map((value, index) => (
              <div key={index} style={styles.featureInputContainer}>
                <label style={styles.featureLabel}>
                  Feature {index + 1}
                </label>
                <input
                  type="number"
                  value={value}
                  onChange={(e) => handleChange(index, e.target.value)}
                  style={styles.featureInput}
                  onFocus={(e) => e.target.style.borderColor = "#06b6d4"}
                  onBlur={(e) => e.target.style.borderColor = "#374151"}
                  placeholder={`F${index + 1}`}
                />
                <div style={styles.featureNumber}>
                  {index + 1}
                </div>
              </div>
            ))}
          </div>

          {/* Prediction Buttons */}
          <div style={{ marginBottom: "2rem" }}>
            <h3 style={{ fontSize: "1.25rem", fontWeight: "bold", color: "white", textAlign: "center", marginBottom: "1rem" }}>
              Select Prediction Model
            </h3>
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              <button
                onClick={() => handlePredict("logistic")}
                disabled={loading}
                style={{
                  ...styles.button,
                  ...styles.predictButton,
                  opacity: loading && activeModel === "logistic" ? 0.7 : 1,
                  animation: loading && activeModel === "logistic" ? "pulse 2s infinite" : "none",
                  border: activeModel === "logistic" ? "2px solid #a855f7" : "none",
                }}
                onMouseOver={(e) => !loading && (e.currentTarget.style.transform = "translateY(-2px)")}
                onMouseOut={(e) => e.currentTarget.style.transform = "translateY(0)"}
              >
                <svg style={{ width: "1.5rem", height: "1.5rem" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                Logistic Regression
              </button>
              <button
                onClick={() => handlePredict("tree")}
                disabled={loading}
                style={{
                  ...styles.button,
                  ...styles.treeButton,
                  opacity: loading && activeModel === "tree" ? 0.7 : 1,
                  animation: loading && activeModel === "tree" ? "pulse 2s infinite" : "none",
                  border: activeModel === "tree" ? "2px solid #06b6d4" : "none",
                }}
                onMouseOver={(e) => !loading && (e.currentTarget.style.transform = "translateY(-2px)")}
                onMouseOut={(e) => e.currentTarget.style.transform = "translateY(0)"}
              >
                <svg style={{ width: "1.5rem", height: "1.5rem" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                </svg>
                Decision Tree
              </button>
            </div>
          </div>

          {/* Loading State */}
          {loading && (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", marginBottom: "2rem" }}>
              <div style={styles.loadingSpinner}></div>
              <p style={{ color: "#06b6d4", fontWeight: "600", fontSize: "1.125rem" }}>
                Analyzing network patterns...
              </p>
              <p style={{ color: "#9ca3af", fontSize: "0.875rem", marginTop: "0.5rem" }}>
                Processing 41 features with {activeModel} model
              </p>
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="animate-fadeIn" style={{
              ...styles.resultCard,
              ...(result.error ? styles.errorResult : 
                  result.attack_detected ? styles.attackResult : styles.normalResult)
            }}>
              <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.75rem" }}>
                  {result.error ? (
                    <div style={{ ...styles.resultIcon, backgroundColor: "#dc2626" }}>
                      <svg style={{ width: "1.5rem", height: "1.5rem", color: "white" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                  ) : result.attack_detected ? (
                    <div style={{ ...styles.resultIcon, backgroundColor: "#dc2626", animation: "pulse 2s infinite" }}>
                      <svg style={{ width: "1.5rem", height: "1.5rem", color: "white" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.104 16.5c-.77.833.192 2.5 1.732 2.5z" />
                      </svg>
                    </div>
                  ) : (
                    <div style={{ ...styles.resultIcon, backgroundColor: "#059669" }}>
                      <svg style={{ width: "1.5rem", height: "1.5rem", color: "white" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                  )}
                  <h3 style={{ fontSize: "1.5rem", fontWeight: "bold", color: "white", margin: 0 }}>
                    {result.error ? "Error" : result.attack_detected ? "Threat Detected" : "All Clear"}
                  </h3>
                </div>
                
                {!result.error && (
                  <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "1rem" }}>
                    <div style={{ backgroundColor: "rgba(31, 41, 55, 0.5)", padding: "0.5rem 1rem", borderRadius: "0.5rem" }}>
                      <span style={{ fontSize: "0.875rem", color: "#9ca3af" }}>Model:</span>
                      <span style={{ marginLeft: "0.5rem", fontWeight: "600", color: "#67e8f9", textTransform: "capitalize" }}>
                        {result.model}
                      </span>
                    </div>
                    <div style={{ backgroundColor: "rgba(31, 41, 55, 0.5)", padding: "0.5rem 1rem", borderRadius: "0.5rem" }}>
                      <span style={{ fontSize: "0.875rem", color: "#9ca3af" }}>Confidence:</span>
                      <span style={{ marginLeft: "0.5rem", fontWeight: "600", color: "#d8b4fe" }}>High</span>
                    </div>
                  </div>
                )}

                <div style={{ textAlign: "center" }}>
                  {result.error ? (
                    <p style={{ color: "#fca5a5", fontWeight: "600", fontSize: "1.125rem" }}>{result.error}</p>
                  ) : (
                    <p style={{
                      fontSize: "2.25rem",
                      fontWeight: "900",
                      margin: 0,
                      color: result.attack_detected ? "#fca5a5" : "#86efac"
                    }}>
                      {result.attack_detected ? "🚨 ATTACK DETECTED!" : "✅ NORMAL TRAFFIC"}
                    </p>
                  )}
                  <p style={{ color: "#9ca3af", fontSize: "0.875rem", marginTop: "0.5rem" }}>
                    Timestamp: {new Date().toLocaleTimeString()}
                  </p>
                </div>
                
                {!result.error && (
                  <div style={{ marginTop: "1.5rem", paddingTop: "1.5rem", borderTop: "1px solid rgba(55, 65, 81, 0.5)" }}>
                    <p style={{ color: "#d1d5db", textAlign: "center" }}>
                      {result.attack_detected 
                        ? "Immediate action recommended. The system detected suspicious network activity patterns."
                        : "Network traffic appears normal. No security threats detected at this moment."
                      }
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Footer */}
          <div style={{ marginTop: "2rem", paddingTop: "1.5rem", borderTop: "1px solid rgba(55, 65, 81, 0.5)" }}>
            <p style={{ textAlign: "center", color: "#9ca3af", fontSize: "0.875rem" }}>
              System Status: <span style={{ color: "#10b981", fontWeight: "600" }}>● Operational</span> | 
              Last Updated: Just now | 
              API: <span style={{ color: "#06b6d4", fontWeight: "600" }}>Connected</span>
            </p>
          </div>
        </div>

        {/* Watermark */}
        <div style={{ marginTop: "2rem", textAlign: "center" }}>
          <p style={{ color: "#6b7280", fontSize: "0.875rem" }}>
            Powered by Advanced ML Models | Secure Network Monitoring
          </p>
        </div>
      </div>
    </>
  );
}

export default App;