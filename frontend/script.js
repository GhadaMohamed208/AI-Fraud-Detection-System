(() => {
    "use strict";

    const API_BASE = window.location.origin;

    const fileInput = document.getElementById("csvFile");
    const fileName = document.getElementById("fileName");
    const analyzeButton = document.getElementById("analyzeCsvBtn");
    const buttonText = document.getElementById("csvButtonText");
    const loader = document.getElementById("csvLoader");
    const errorBox = document.getElementById("csvErrorBox");
    const resultSection = document.getElementById("csvResultSection");
    const totalEl = document.getElementById("csvTotal");
    const fraudEl = document.getElementById("csvFraud");
    const legitimateEl = document.getElementById("csvLegitimate");
    const fraudRateEl = document.getElementById("csvFraudRate");
    const metricsEl = document.getElementById("csvMetrics");
    const tableBody = document.querySelector("#csvResultsTable tbody");
    const tableNote = document.getElementById("csvTableNote");
    const downloadButton = document.getElementById("downloadCsvBtn");

    let latestResults = null;

    function showError(message) {
        if (!errorBox) return;
        errorBox.textContent = message;
        errorBox.classList.remove("hidden");
    }

    function clearError() {
        if (!errorBox) return;
        errorBox.textContent = "";
        errorBox.classList.add("hidden");
    }

    function escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function number(value, digits = 4) {
        const n = Number(value);
        return Number.isFinite(n) ? n.toFixed(digits) : "-";
    }

    function setLoading(loading) {
        if (analyzeButton) analyzeButton.disabled = loading;
        if (buttonText) buttonText.textContent = loading ? "Analyzing..." : "Analyze CSV";
        if (loader) loader.classList.toggle("hidden", !loading);
    }

    function renderResults(data) {
        latestResults = data;
        const summary = data.summary || {};
        const rows = Array.isArray(data.rows) ? data.rows : [];

        if (totalEl) totalEl.textContent = Number(summary.total_transactions ?? rows.length).toLocaleString();
        if (fraudEl) fraudEl.textContent = Number(summary.fraud_count ?? 0).toLocaleString();
        if (legitimateEl) legitimateEl.textContent = Number(summary.legitimate_count ?? 0).toLocaleString();
        if (fraudRateEl) fraudRateEl.textContent = `${(Number(summary.fraud_rate ?? 0) * 100).toFixed(2)}%`;

        if (metricsEl) {
            const extra = [];
            if (summary.has_actual_labels) {
                extra.push(`Accuracy: <strong>${number(summary.accuracy * 100, 2)}%</strong>`);
                extra.push(`Correct: <strong>${summary.correct_predictions}</strong>`);
                extra.push(`False Positives: <strong>${summary.false_positives}</strong>`);
                extra.push(`False Negatives: <strong>${summary.false_negatives}</strong>`);
            }
            extra.push(`Threshold: <strong>${number(summary.threshold, 2)}</strong>`);
            extra.push(`Model: <strong>${escapeHtml(summary.model_version ?? "-")}</strong>`);
            metricsEl.innerHTML = extra.join(" &nbsp; | &nbsp; ");
            metricsEl.classList.toggle("hidden", extra.length === 0);
        }

        if (tableBody) {
            tableBody.innerHTML = rows.map((row, index) => {
                const prediction = row.prediction ?? row.status ?? "Unknown";
                const fraud = prediction === "Fraud";
                const reasons = Array.isArray(row.shap_explanation) ? row.shap_explanation : [];
                const reason = fraud
                    ? (row.reason || reasons.slice(0, 3).map(x => x.feature).join(", ") || "Model anomaly pattern")
                    : "No significant fraud indicators";

                return `
                    <tr class="${fraud ? "fraud-row" : ""}">
                        <td>${escapeHtml(row.transaction_id ?? `CSV-${String(index + 1).padStart(6, "0")}`)}</td>
                        <td>${number(row.amount, 2)}</td>
                        <td class="${fraud ? "prediction-fraud" : "prediction-legitimate"}">${escapeHtml(prediction)}</td>
                        <td>${number(row.fraud_score, 4)}</td>
                        <td>${number(row.ml_probability, 4)}</td>
                        <td>${number(row.anomaly_score, 4)}</td>
                        <td>${escapeHtml(reason)}</td>
                    </tr>
                `;
            }).join("");
        }

        if (tableNote) {
            const fraudRows = rows.filter(r => (r.prediction ?? r.status) === "Fraud").length;
            tableNote.textContent = fraudRows
                ? `SHAP explanations are generated only for the ${fraudRows} transaction(s) classified as Fraud.`
                : "No transactions were classified as Fraud, so no SHAP explanations were generated.";
        }

        if (resultSection) {
            resultSection.classList.remove("hidden");
            resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }

    async function analyzeCSV() {
        clearError();

        if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
            showError("Please choose a CSV file first.");
            return;
        }

        const file = fileInput.files[0];
        if (!file.name.toLowerCase().endsWith(".csv")) {
            showError("Please upload a CSV file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file, file.name);

        setLoading(true);
        if (resultSection) resultSection.classList.add("hidden");

        try {
            const response = await fetch(`${API_BASE}/predict-csv`, {
                method: "POST",
                body: formData,
                headers: { "Accept": "application/json" }
            });

            const contentType = response.headers.get("content-type") || "";
            let data;

            if (contentType.includes("application/json")) {
                data = await response.json();
            } else {
                const text = await response.text();
                throw new Error(`Backend returned HTTP ${response.status}: ${text.slice(0, 300)}`);
            }

            if (!response.ok) {
                throw new Error(data.detail || data.message || "CSV analysis failed.");
            }

            renderResults(data);
        } catch (error) {
            console.error("CSV analysis error:", error);
            showError(error.message || "Could not connect to the backend.");
        } finally {
            setLoading(false);
        }
    }

    function downloadCSVResults() {
        clearError();

        if (!latestResults || !Array.isArray(latestResults.rows) || latestResults.rows.length === 0) {
            showError("There are no results to download.");
            return;
        }

        const headers = [
            "transaction_id", "amount", "prediction", "fraud_score",
            "ml_probability", "anomaly_score", "reason"
        ];

        const escapeCsv = value => `"${String(value ?? "").replaceAll('"', '""')}"`;
        const lines = [headers.join(",")];

        for (const row of latestResults.rows) {
            const prediction = row.prediction ?? row.status ?? "Unknown";
            const reasons = Array.isArray(row.shap_explanation) ? row.shap_explanation : [];
            const reason = prediction === "Fraud"
                ? (row.reason || reasons.slice(0, 3).map(x => x.feature).join(" | ") || "Model anomaly pattern")
                : "No significant fraud indicators";

            lines.push([
                row.transaction_id ?? "",
                row.amount ?? "",
                prediction,
                row.fraud_score ?? "",
                row.ml_probability ?? "",
                row.anomaly_score ?? "",
                reason
            ].map(escapeCsv).join(","));
        }

        const blob = new Blob(["\uFEFF" + lines.join("\r\n")], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "fraud_analysis_results.csv";
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    }

    if (!fileInput || !analyzeButton) {
        console.error("CSV UI elements were not found in index.html.");
        return;
    }

    fileInput.addEventListener("change", () => {
        clearError();
        if (fileName) fileName.textContent = fileInput.files?.[0]?.name || "Choose CSV file";
    });

    analyzeButton.addEventListener("click", analyzeCSV);

    if (downloadButton) downloadButton.addEventListener("click", downloadCSVResults);

    // Expose functions for debugging/manual testing from the browser console.
    window.analyzeCSV = analyzeCSV;
    window.downloadCSVResults = downloadCSVResults;

    console.log("AI Fraud Detection CSV frontend loaded successfully.");
})();
